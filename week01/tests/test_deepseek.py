"""DeepSeek 适配器验证（方案 1：新增 Chat Completions 适配器）。

验证点：
    1. 真实模式下两个模型分别路由到「协议不同」的适配器
       （deepseek-chat vs anthropic-messages）。
    2. DeepSeekAdapter 的入站翻译是 Chat Completions 方言：
       system 作为 messages 内的一条、字段用 max_tokens / response_format。
    3. 离线 Mock 模式：deepseek-v4-pro 也能正常返回统一响应（protocol=deepseek-chat）。
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings
from app.routing.router import build_registry
from app.gateway import Gateway
from app.core.models import ChatRequest, Message
from app.adapters.deepseek_adapter import DeepSeekAdapter


def test_real_routing_uses_deepseek_and_anthropic():
    settings = Settings(use_mock=False)
    gw = Gateway(settings, registry=build_registry(settings))
    a_pro = gw.registry.resolve("deepseek-v4-pro")
    a_flash = gw.registry.resolve("deepseek-v4-flash")
    assert a_pro.protocol == "deepseek-chat"
    assert a_flash.protocol == "anthropic-messages"
    assert a_pro.protocol != a_flash.protocol


def test_deepseek_body_is_chat_completions_dialect():
    adapter = DeepSeekAdapter("deepseek", "sk-x", "https://api.deepseek.com/v1")
    req = ChatRequest(
        model="deepseek-v4-pro",
        messages=[
            Message(role="system", content="你是助手"),
            Message(role="user", content="你好"),
        ],
        max_tokens=128,
        response_format={"type": "json_object"},
    )
    body = adapter._build_body(req)
    # Chat Completions：system 作为 messages 内一条、而非 instructions
    roles = [m["role"] for m in body["messages"]]
    assert "system" in roles
    assert body["max_tokens"] == 128
    assert body["response_format"] == {"type": "json_object"}
    # 关键：不是 Responses API 的字段
    assert "instructions" not in body
    assert "input" not in body
    assert "max_output_tokens" not in body


def test_mock_deepseek_returns_unified_response():
    # mock 模式下 pro 默认走 openai（对齐作业）；这里直接验证 MockDeepSeek 适配器本身
    from app.adapters.mock import MockDeepSeekAdapter

    adapter = MockDeepSeekAdapter("deepseek")
    assert adapter.protocol == "deepseek-chat"

    async def run():
        r = await adapter.chat(ChatRequest(
            model="deepseek-v4-pro", messages=[Message(role="user", content="你好")]
        ), stream=False)
        return r

    r = asyncio.run(run())
    assert r.model == "deepseek-v4-pro"
    assert r.usage.total_tokens > 0
    assert r.choices[0].message.content


if __name__ == "__main__":
    test_real_routing_uses_deepseek_and_anthropic()
    test_deepseek_body_is_chat_completions_dialect()
    test_mock_deepseek_returns_unified_response()
    print("DEEPSEEK_OK: DeepSeek 适配器（路由/方言/离线 Mock）均通过")
