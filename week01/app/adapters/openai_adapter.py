"""真实 OpenAI Responses API 适配器（需 API Key，USE_MOCK=0 时启用）。

职责（适配器模式）：
    入站：把统一 ChatRequest 翻译成 OpenAI Responses API 的请求体
          （input / instructions / temperature / max_output_tokens / text.format / stream）
    出站：把 OpenAI 的响应（非流式 JSON 或流式 SSE 事件）归一化成统一 ChatResponse / StreamChunk

与 Mock 的关键区别：
    这里真的通过 httpx 调用远端 {base_url}/responses；协议差异（尤其是 Responses API
    用 output_text、用 instructions 承载 system、用 max_output_tokens）都在此翻译。

错误处理策略（与可重试集合对齐）：
    - 网络超时 → ERR_TIMEOUT（可重试）
    - 上游 5xx / 其它网络错误 → ERR_PROVIDER_ERROR（可重试）
    - 上游 429 → ERR_RATE_LIMITED（不重试，直接透给调用方）
"""
import asyncio
import json
import time
import uuid

import httpx

from app.adapters.base import BaseAdapter
from app.core.models import ChatRequest, ChatResponse, Choice, Message, StreamChunk, Usage
from app.core.utils import estimate_tokens
from app.core.errors import ERR_PROVIDER_ERROR, ERR_RATE_LIMITED, ERR_TIMEOUT, GatewayError


class OpenAIAdapter(BaseAdapter):
    protocol = "openai-responses"

    def __init__(self, name: str, api_key: str, base_url: str):
        super().__init__(name)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    # ——————————————————— 入站翻译 ———————————————————
    def _build_body(self, req: ChatRequest) -> dict:
        """把统一请求翻译成 OpenAI Responses API 请求体。"""
        # Responses API 用 instructions 承载 system 指令，其余消息进 input 数组
        system_texts = [m.content for m in req.messages if m.role == "system"]
        instructions = system_texts[0] if system_texts else None
        input_items = [
            {"role": m.role, "content": m.content}
            for m in req.messages if m.role != "system"
        ]
        body = {
            "model": req.model,
            "input": input_items,
            "temperature": req.temperature,
            "stream": req.stream,
        }
        if instructions:
            body["instructions"] = instructions
        # Responses API 用 max_output_tokens（而非 chat 接口的 max_tokens）
        if req.max_tokens:
            body["max_output_tokens"] = req.max_tokens
        # 结构化输出：翻译为 Responses API 的 text.format 约束
        if req.response_format:
            body["text"] = self._to_text_format(req.response_format)
        return body

    @staticmethod
    def _to_text_format(response_format: dict) -> dict:
        """把统一 response_format 翻成 OpenAI 的 text.format 结构。"""
        if response_format.get("type") == "json_schema":
            return {"format": {"type": "json_schema", "json_schema": response_format["json_schema"]}}
        # 默认 json_object
        return {"format": {"type": "json_object"}}

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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/responses"
        if stream:
            return self._stream_response(url, headers, body, req.model)
        return await self._non_stream_response(url, headers, body, req.model)

    async def _non_stream_response(self, url, headers, body, model) -> ChatResponse:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=body)
        except httpx.TimeoutException:
            raise GatewayError(ERR_TIMEOUT, "openai upstream timeout", 504)
        except httpx.HTTPError as e:
            raise GatewayError(ERR_PROVIDER_ERROR, f"openai upstream error: {e}", 502)

        if r.status_code == 429:
            raise GatewayError(ERR_RATE_LIMITED, "openai upstream rate limited", 429)
        if r.status_code >= 500:
            raise GatewayError(ERR_PROVIDER_ERROR, f"openai 5xx: {r.text[:200]}", 502)
        if r.status_code != 200:
            raise GatewayError(ERR_PROVIDER_ERROR, f"openai error {r.status_code}: {r.text[:200]}", 502)

        data = r.json()
        content = data.get("output_text", "")
        usage = self._usage_from_data(data, content)
        return ChatResponse(
            id=data.get("id", "resp_" + uuid.uuid4().hex[:8]),
            created=int(time.time()), model=data.get("model", model),
            choices=[Choice(message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=usage,
        )

    async def _stream_response(self, url, headers, body, model):
        """流式：解析 OpenAI Responses API 的 SSE，逐块归一化为 StreamChunk。"""
        stream_id = "resp_" + uuid.uuid4().hex[:8]
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
                        if evt.get("type") == "response.output_text.delta":
                            yield StreamChunk(
                                id=stream_id, created=int(time.time()), model=model,
                                choices=[{"index": 0, "delta": {"content": evt.get("delta", "")}, "finish_reason": None}],
                            )
                        elif evt.get("type") == "response.completed":
                            yield StreamChunk(
                                id=stream_id, created=int(time.time()), model=model,
                                choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            )
        except httpx.TimeoutException:
            raise GatewayError(ERR_TIMEOUT, "openai stream timeout", 504)
        except httpx.HTTPError as e:
            raise GatewayError(ERR_PROVIDER_ERROR, f"openai stream error: {e}", 502)
