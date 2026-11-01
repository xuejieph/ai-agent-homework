"""Step 5 验证：适配器层（两种协议 + 离线 Mock）。

验证点：
    1. 两个模型分别路由到「协议不同」的适配器（openai-responses vs anthropic-messages）。
    2. 两个 Mock 适配器都返回合法统一 ChatResponse（非流式）。
    3. 流式：逐块产出，拼接内容正确，且末块 finish_reason=stop。
    4. 结构化输出：带 response_format 时返回内容可被 json.loads 解析（两种协议均验证）。
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings
from app.routing.router import build_registry
from app.gateway import Gateway
from app.core.models import ChatRequest, Message


def _gw():
    """用默认 mock 配置构建 Gateway（含 Mock 双适配器）。"""
    settings = Settings(use_mock=True, model_routing={
        "deepseek-v4-pro": "openai", "deepseek-v4-flash": "anthropic",
    })
    return Gateway(settings, registry=build_registry(settings))


def test_two_models_use_different_protocols():
    gw = _gw()
    a1 = gw.registry.resolve("deepseek-v4-pro")
    a2 = gw.registry.resolve("deepseek-v4-flash")
    assert a1.protocol == "openai-responses"
    assert a2.protocol == "anthropic-messages"
    assert a1.protocol != a2.protocol


def test_non_stream_returns_unified_response():
    gw = _gw()

    async def run():
        r1 = await gw.complete(ChatRequest(model="deepseek-v4-pro", messages=[Message(role="user", content="你好")]))
        r2 = await gw.complete(ChatRequest(model="deepseek-v4-flash", messages=[Message(role="user", content="你好")]))
        return r1, r2

    r1, r2 = asyncio.run(run())
    # 模型名正确回显（证明走对了适配器）
    assert r1.model == "deepseek-v4-pro"
    assert r2.model == "deepseek-v4-flash"
    # 两者都是统一 ChatResponse，且都有非零 usage
    assert r1.usage.total_tokens > 0 and r2.usage.total_tokens > 0
    assert r1.choices[0].message.content  # 有内容


def test_stream_yields_chunks_and_done():
    gw = _gw()

    async def run():
        # 经 gateway.stream 走完整编排（路由/限流/重试/观测）的流式分支
        chunks = []
        async for c in gw.stream(ChatRequest(model="deepseek-v4-pro", messages=[Message(role="user", content="abc")], stream=True)):
            chunks.append(c)
        return chunks

    chunks = asyncio.run(run())
    texts = [c.choices[0]["delta"].get("content", "") for c in chunks if c.choices[0]["delta"]]
    joined = "".join(texts)
    assert joined == "[deepseek-v4-pro] abc"  # Mock 非结构化回答形状
    assert chunks[-1].choices[0]["finish_reason"] == "stop"  # 末块结束


def test_structured_output_is_valid_json_both_protocols():
    gw = _gw()

    async def run():
        r_pro = await gw.complete(ChatRequest(
            model="deepseek-v4-pro",
            messages=[Message(role="user", content="x")],
            response_format={"type": "json_object"},
        ))
        r_flash = await gw.complete(ChatRequest(
            model="deepseek-v4-flash",
            messages=[Message(role="user", content="x")],
            response_format={"type": "json_object"},
        ))
        return r_pro, r_flash

    r_pro, r_flash = asyncio.run(run())
    # 两种协议下，结构化输出都应返回可被 json.loads 解析的内容
    json.loads(r_pro.choices[0].message.content)
    json.loads(r_flash.choices[0].message.content)


if __name__ == "__main__":
    test_two_models_use_different_protocols()
    test_non_stream_returns_unified_response()
    test_stream_yields_chunks_and_done()
    test_structured_output_is_valid_json_both_protocols()
    print("ADAPTERS_OK: 两种协议适配器（Mock）路由/非流式/流式/结构化 均通过")
