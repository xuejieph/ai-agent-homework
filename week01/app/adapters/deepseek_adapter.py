"""真实 DeepSeek Chat Completions 适配器（OpenAI 兼容，USE_MOCK=0 时启用）。

为什么单独写一个适配器，而不是复用 OpenAIAdapter？
    DeepSeek 提供的是「OpenAI 兼容的 Chat Completions」接口
    （/v1/chat/completions，请求体用 messages、返回用 choices[].message.content），
    而本项目 OpenAIAdapter 打的是 OpenAI **Responses API** 方言
    （/responses，用 input / instructions / output_text）。两者协议形状不同，
    这正是作业要求「两种不同协议」的落脚点之一：

        deepseek-v4-pro  → DeepSeek Chat Completions（本适配器，protocol=deepseek-chat）
        deepseek-v4-flash → Anthropic Messages API（anthropic_adapter）

    因此 DeepSeek 不能简单把 OpenAIAdapter 的 base_url 指过去（会打到不存在的
    /responses 端点），必须按 chat-completions 方言重新翻译。

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


class DeepSeekAdapter(BaseAdapter):
    protocol = "deepseek-chat"

    def __init__(self, name: str, api_key: str, base_url: str):
        super().__init__(name)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    # ——————————————————— 入站翻译（Chat Completions 方言） ———————————————————
    def _build_body(self, req: ChatRequest) -> dict:
        """把统一请求翻译成 OpenAI 兼容的 chat/completions 请求体。

        与 Responses API 的差别：system 直接作为 messages 里的一条 role=system，
        而非 instructions；token 上限字段是 max_tokens，而非 max_output_tokens。
        """
        messages = [
            {"role": m.role, "content": m.content}
            for m in req.messages
        ]
        body = {
            "model": req.model,
            "messages": messages,
            "temperature": req.temperature,
            "stream": req.stream,
        }
        if req.max_tokens:
            body["max_tokens"] = req.max_tokens
        # 结构化输出：DeepSeek 支持 response_format（json_object / json_schema）
        if req.response_format:
            body["response_format"] = req.response_format
        return body

    @staticmethod
    def _usage_from_data(data: dict, fallback_text: str) -> Usage:
        u = data.get("usage") or {}
        pt = u.get("prompt_tokens") or estimate_tokens(fallback_text)
        ct = u.get("completion_tokens") or estimate_tokens(fallback_text)
        return Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)

    # ——————————————————— 出站翻译 ———————————————————
    async def chat(self, req: ChatRequest, stream: bool):
        body = self._build_body(req)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/chat/completions"
        if stream:
            return self._stream_response(url, headers, body, req.model)
        return await self._non_stream_response(url, headers, body, req.model)

    async def _non_stream_response(self, url, headers, body, model) -> ChatResponse:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=body)
        except httpx.TimeoutException:
            raise GatewayError(ERR_TIMEOUT, "deepseek upstream timeout", 504)
        except httpx.HTTPError as e:
            raise GatewayError(ERR_PROVIDER_ERROR, f"deepseek upstream error: {e}", 502)

        if r.status_code == 429:
            raise GatewayError(ERR_RATE_LIMITED, "deepseek upstream rate limited", 429)
        if r.status_code >= 500:
            raise GatewayError(ERR_PROVIDER_ERROR, f"deepseek 5xx: {r.text[:200]}", 502)
        if r.status_code != 200:
            raise GatewayError(ERR_PROVIDER_ERROR, f"deepseek error {r.status_code}: {r.text[:200]}", 502)

        data = r.json()
        content = data["choices"][0]["message"]["content"]
        usage = self._usage_from_data(data, content)
        return ChatResponse(
            id=data.get("id", "chatcmpl_" + uuid.uuid4().hex[:8]),
            created=int(time.time()), model=data.get("model", model),
            choices=[Choice(message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=usage,
        )

    async def _stream_response(self, url, headers, body, model):
        """流式：解析 OpenAI 兼容 SSE（choices[].delta.content + [DONE]）。"""
        stream_id = "chatcmpl_" + uuid.uuid4().hex[:8]
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
                        choices = evt.get("choices") or []
                        if not choices:
                            # 一些事件（如仅含 usage 的块）没有 choices，忽略
                            continue
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        finish = choices[0].get("finish_reason")
                        if content is not None:
                            yield StreamChunk(
                                id=stream_id, created=int(time.time()), model=model,
                                choices=[{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                            )
                        if finish:
                            yield StreamChunk(
                                id=stream_id, created=int(time.time()), model=model,
                                choices=[{"index": 0, "delta": {}, "finish_reason": finish}],
                            )
        except httpx.TimeoutException:
            raise GatewayError(ERR_TIMEOUT, "deepseek stream timeout", 504)
        except httpx.HTTPError as e:
            raise GatewayError(ERR_PROVIDER_ERROR, f"deepseek stream error: {e}", 502)
