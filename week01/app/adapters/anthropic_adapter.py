"""真实 Anthropic Messages API 适配器（需 API Key，USE_MOCK=0 时启用）。

职责（适配器模式）：
    入站：把统一 ChatRequest 翻译成 Anthropic Messages API 请求体
          （system / messages / max_tokens / temperature / stream）
    出站：把 Anthropic 的响应（content[].text 或 content_block_delta 事件）归一化

与 OpenAI 原生结构的差异（这正是两种协议不同的体现）：
    - Anthropic 把 system 单独成字段（不是 messages 里的 role=system）
    - 非流式文本在 content 数组里（[{type:"text", text}]）
    - 流式用 content_block_delta，且 delta.text 承载增量
    - 必填 max_tokens；结构化输出没有原生 json 模式，这里用 system 指令约束
"""
import json
import time
import uuid

import httpx

from app.adapters.base import BaseAdapter
from app.core.models import ChatRequest, ChatResponse, Choice, Message, StreamChunk, Usage
from app.core.utils import estimate_tokens
from app.core.errors import ERR_PROVIDER_ERROR, ERR_RATE_LIMITED, ERR_TIMEOUT, GatewayError


class AnthropicAdapter(BaseAdapter):
    protocol = "anthropic-messages"

    def __init__(self, name: str, api_key: str, base_url: str):
        super().__init__(name)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    # ——————————————————— 入站翻译 ———————————————————
    def _build_body(self, req: ChatRequest) -> dict:
        system_texts = [m.content for m in req.messages if m.role == "system"]
        system = system_texts[0] if system_texts else None
        messages = [
            {"role": m.role, "content": m.content}
            for m in req.messages if m.role != "system"
        ]
        # Anthropic 必填 max_tokens
        body = {
            "model": req.model,
            "messages": messages,
            "max_tokens": req.max_tokens or 1024,
            "temperature": req.temperature,
            "stream": req.stream,
        }
        if system:
            # 结构化输出：Anthropic 无原生 JSON 模式，用 system 指令要求只输出 JSON
            if req.response_format:
                system = system + "\n\n请只输出合法的 JSON，不要包含任何额外说明文字。"
            body["system"] = system
        return body

    # ——————————————————— 出站翻译 ———————————————————
    @staticmethod
    def _usage_from_data(data: dict, fallback_text: str) -> Usage:
        u = data.get("usage") or {}
        pt = u.get("input_tokens") or estimate_tokens(fallback_text)
        ct = u.get("output_tokens") or estimate_tokens(fallback_text)
        return Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)

    async def chat(self, req: ChatRequest, stream: bool):
        body = self._build_body(req)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/messages"
        if stream:
            return self._stream_response(url, headers, body, req.model)
        return await self._non_stream_response(url, headers, body, req.model)

    async def _non_stream_response(self, url, headers, body, model) -> ChatResponse:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=body)
        except httpx.TimeoutException:
            raise GatewayError(ERR_TIMEOUT, "anthropic upstream timeout", 504)
        except httpx.HTTPError as e:
            raise GatewayError(ERR_PROVIDER_ERROR, f"anthropic upstream error: {e}", 502)

        if r.status_code == 429:
            raise GatewayError(ERR_RATE_LIMITED, "anthropic upstream rate limited", 429)
        if r.status_code >= 500:
            raise GatewayError(ERR_PROVIDER_ERROR, f"anthropic 5xx: {r.text[:200]}", 502)
        if r.status_code != 200:
            raise GatewayError(ERR_PROVIDER_ERROR, f"anthropic error {r.status_code}: {r.text[:200]}", 502)

        data = r.json()
        # 非流式：文本在 content 数组里
        content = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = self._usage_from_data(data, content)
        return ChatResponse(
            id=data.get("id", "msg_" + uuid.uuid4().hex[:8]),
            created=int(time.time()), model=data.get("model", model),
            choices=[Choice(message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=usage,
        )

    async def _stream_response(self, url, headers, body, model):
        """流式：解析 Anthropic 的 SSE，逐块归一化为 StreamChunk。"""
        stream_id = "msg_" + uuid.uuid4().hex[:8]
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            evt = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if evt.get("type") == "content_block_delta":
                            text = evt.get("delta", {}).get("text", "")
                            yield StreamChunk(
                                id=stream_id, created=int(time.time()), model=model,
                                choices=[{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                            )
                        elif evt.get("type") == "message_stop":
                            yield StreamChunk(
                                id=stream_id, created=int(time.time()), model=model,
                                choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            )
        except httpx.TimeoutException:
            raise GatewayError(ERR_TIMEOUT, "anthropic stream timeout", 504)
        except httpx.HTTPError as e:
            raise GatewayError(ERR_PROVIDER_ERROR, f"anthropic stream error: {e}", 502)
