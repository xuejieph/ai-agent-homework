"""离线 Mock 适配器：无需任何 API Key，模拟两种不同协议的返回。

为什么需要 Mock 适配器？
    作业要求「两种不同 API 协议」，真实协议需要 Key 且联网。为了能在任意环境离线
    验证全部功能点，我们提供：
        - MockOpenAIAdapter    模拟 OpenAI Responses API
        - MockAnthropicAdapter 模拟 Anthropic Messages API
        - FaultyAdapter        前 N 次失败（用于演示重试后恢复）

设计要点（适配器模式的核心体现）：
    每个 Mock 适配器内部先生成「该协议原生的响应结构」——两种协议的原生形状是
    互不相同的（OpenAI 用 output_text / delta 事件；Anthropic 用 content[].text /
    content_block_delta 事件）。适配器再把原生结构「翻译」成统一的 ChatResponse /
    StreamChunk。因此即便内部表示不同，网关看到的永远是同一套契约，协议差异被封死。
"""
import asyncio
import json
import time
import uuid

from app.adapters.base import BaseAdapter
from app.core.models import ChatRequest, ChatResponse, Choice, Message, StreamChunk, Usage
from app.core.utils import estimate_tokens
from app.core.errors import ERR_PROVIDER_ERROR, GatewayError

PROTOCOL_OPENAI = "openai-responses"
PROTOCOL_ANTHROPIC = "anthropic-messages"
PROTOCOL_DEEPSEEK = "deepseek-chat"


def _last_user_text(req: ChatRequest) -> str:
    """取出最后一条 user 消息文本，作为 Mock 回复的依据（模拟「看到用户输入」）。"""
    for m in reversed(req.messages):
        if m.role == "user":
            return m.content
    return ""


def _mock_answer(model: str, text: str, structured: bool) -> str:
    """生成 Mock 回答。

    结构化输出时返回可被 json.loads 解析的 JSON 串（内容带上模型名与回显输入），
    方便测试断言「返回是合法 JSON」；否则返回普通文本。
    """
    if structured:
        return json.dumps({"model": model, "echo": text, "ok": True}, ensure_ascii=False)
    return f"[{model}] {text}"


def _usage_from(req: ChatRequest, answer_text: str) -> Usage:
    """根据请求消息与回答文本估算用量（Mock 无真模型，用估算口径）。"""
    prompt = estimate_tokens(" ".join(m.content for m in req.messages))
    completion = estimate_tokens(answer_text)
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)


class MockOpenAIAdapter(BaseAdapter):
    """模拟 OpenAI Responses API 协议。"""

    protocol = PROTOCOL_OPENAI

    def __init__(self, name: str = "openai"):
        super().__init__(name)

    async def chat(self, req: ChatRequest, stream: bool):
        text = _last_user_text(req)
        structured = req.has_structured_output()
        answer = _mock_answer(req.model, text, structured)
        usage = _usage_from(req, answer)
        if stream:
            return self._stream_openai(answer, req.model)
        return self._openai_response(answer, req.model, usage)

    # —— 以下方法刻意先构造「OpenAI 原生结构」，再归一化，体现翻译过程 ——
    def _openai_response(self, answer: str, model: str, usage: Usage) -> ChatResponse:
        # OpenAI Responses API 的顶层用 output_text 承载文本
        native = {"id": "resp_" + uuid.uuid4().hex[:8], "model": model, "output_text": answer}
        return self._from_openai_native(native, usage)

    @staticmethod
    def _from_openai_native(native: dict, usage: Usage) -> ChatResponse:
        content = native["output_text"]
        return ChatResponse(
            id=native["id"], created=int(time.time()), model=native["model"],
            choices=[Choice(message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=usage,
        )

    async def _stream_openai(self, answer: str, model: str):
        # 模拟 Responses API 的流式事件：response.output_text.delta
        stream_id = "resp_" + uuid.uuid4().hex[:8]
        for ch in answer:
            # 原生事件形状：{type, delta}
            native_evt = {"type": "response.output_text.delta", "delta": ch}
            yield self._chunk_from_openai_delta(native_evt, model, stream_id)
            await asyncio.sleep(0)  # 让出事件循环，模拟逐块到达
        # 结束事件：原生 response.completed → 归一化成 finish_reason=stop
        yield self._chunk_from_openai_delta({"type": "response.completed", "delta": ""}, model, stream_id, finish=True)

    @staticmethod
    def _chunk_from_openai_delta(native_evt: dict, model: str, stream_id: str, finish: bool = False) -> StreamChunk:
        delta = {"content": native_evt.get("delta", "")} if not finish else {}
        return StreamChunk(
            id=stream_id, created=int(time.time()), model=model,
            choices=[{"index": 0, "delta": delta, "finish_reason": "stop" if finish else None}],
        )


class MockAnthropicAdapter(BaseAdapter):
    """模拟 Anthropic Messages API 协议。"""

    protocol = PROTOCOL_ANTHROPIC

    def __init__(self, name: str = "anthropic"):
        super().__init__(name)

    async def chat(self, req: ChatRequest, stream: bool):
        text = _last_user_text(req)
        structured = req.has_structured_output()
        answer = _mock_answer(req.model, text, structured)
        usage = _usage_from(req, answer)
        if stream:
            return self._stream_anthropic(answer, req.model)
        return self._anthropic_response(answer, req.model, usage)

    # —— 以下方法构造「Anthropic 原生结构」（与 OpenAI 形状不同），再归一化 ——
    def _anthropic_response(self, answer: str, model: str, usage: Usage) -> ChatResponse:
        # Anthropic Messages API 的非流式返回：content 是 [{type:"text", text}]
        native = {"id": "msg_" + uuid.uuid4().hex[:8], "model": model,
                  "content": [{"type": "text", "text": answer}]}
        return self._from_anthropic_native(native, usage)

    @staticmethod
    def _from_anthropic_native(native: dict, usage: Usage) -> ChatResponse:
        content = "".join(b.get("text", "") for b in native.get("content", []) if b.get("type") == "text")
        return ChatResponse(
            id=native["id"], created=int(time.time()), model=native["model"],
            choices=[Choice(message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=usage,
        )

    async def _stream_anthropic(self, answer: str, model: str):
        # 模拟 Messages API 的流式事件：content_block_delta + 最终的 message_stop
        stream_id = "msg_" + uuid.uuid4().hex[:8]
        for ch in answer:
            native_evt = {"type": "content_block_delta",
                          "delta": {"type": "text_delta", "text": ch}}
            yield self._chunk_from_anthropic_delta(native_evt, model, stream_id)
            await asyncio.sleep(0)
        yield self._chunk_from_anthropic_delta({"type": "message_stop", "delta": {}}, model, stream_id, finish=True)

    @staticmethod
    def _chunk_from_anthropic_delta(native_evt: dict, model: str, stream_id: str, finish: bool = False) -> StreamChunk:
        text = native_evt.get("delta", {}).get("text", "") if not finish else ""
        delta = {"content": text} if not finish else {}
        return StreamChunk(
            id=stream_id, created=int(time.time()), model=model,
            choices=[{"index": 0, "delta": delta, "finish_reason": "stop" if finish else None}],
        )


class MockDeepSeekAdapter(MockOpenAIAdapter):
    """模拟 DeepSeek Chat Completions 协议（OpenAI 兼容）。

    直接复用 MockOpenAIAdapter 的全部逻辑（同样是「补全式」原生结构 → 归一化），
    仅把 protocol 标识为 deepseek-chat，以体现它与 Responses API 是不同协议。
    """

    protocol = PROTOCOL_DEEPSEEK

    def __init__(self, name: str = "deepseek"):
        super().__init__(name)


class FaultyAdapter(BaseAdapter):
    """故障适配器：前 fail_times 次抛可重试错误，之后恢复正常。

    用途：在测试中演示「网关指数退避重试」——把该适配器注册给某个测试模型，
    即可观察到失败后重试、最终成功，且 metrics 记录 retries>0。
    """

    protocol = "faulty"

    def __init__(self, name: str = "faulty", fail_times: int = 2):
        super().__init__(name)
        self.fail_times = fail_times
        self.calls = 0

    async def chat(self, req: ChatRequest, stream: bool):
        self.calls += 1
        if self.calls <= self.fail_times:
            # 抛「可重试」错误（PROVIDER_ERROR 在 RETRYABLE 集合内）
            raise GatewayError(ERR_PROVIDER_ERROR, f"simulated upstream failure #{self.calls}", http_status=502)

        text = _last_user_text(req)
        answer = _mock_answer(req.model, text, req.has_structured_output())
        usage = _usage_from(req, answer)
        return ChatResponse(
            id="rec_" + uuid.uuid4().hex[:8], created=int(time.time()), model=req.model,
            choices=[Choice(message=Message(role="assistant", content=answer), finish_reason="stop")],
            usage=usage,
        )
