"""Step 2 验证：统一抽象层 + 动态路由。

验证点：
    1. 两个模型分别落到不同协议的适配器（deepseek-v4-pro → openai-responses，
       deepseek-v4-flash → anthropic-messages），且实例不同。
    2. 未知模型名解析时抛 MODEL_NOT_FOUND（HTTP 404）。

说明：本测试用两个最小桩适配器（Stub）验证「路由解析」这一纯逻辑，
不依赖真实的 OpenAI/Anthropic 适配器（它们在 Step 5 才实现）。
"""
import sys
from pathlib import Path

# 让 `app` 包可从仓库根（homework/week01）导入。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 兼容处理：本环境可能没装 pytest。用 try 守护导入，保证文件在「无 pytest」时
# 也能被 import；真正的 pytest 测试函数只在 pytest 存在时由 pytest 收集执行。
try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    pytest = None  # type: ignore[assignment]
    HAS_PYTEST = False

from app.adapters.base import BaseAdapter
from app.routing.router import AdapterRegistry
from app.core.errors import ERR_MODEL_NOT_FOUND, GatewayError


class _StubOpenAI(BaseAdapter):
    """桩适配器：模拟 OpenAI Responses API 协议。"""
    protocol = "openai-responses"

    async def chat(self, req, stream):
        return None  # 桩实现，仅用于路由验证


class _StubAnthropic(BaseAdapter):
    """桩适配器：模拟 Anthropic Messages API 协议。"""
    protocol = "anthropic-messages"

    async def chat(self, req, stream):
        return None


def _build_registry() -> AdapterRegistry:
    reg = AdapterRegistry()
    reg.register("openai", _StubOpenAI("openai"), ["deepseek-v4-pro"])
    reg.register("anthropic", _StubAnthropic("anthropic"), ["deepseek-v4-flash"])
    return reg


def test_two_models_resolve_to_different_adapters():
    reg = _build_registry()
    adapter_pro = reg.resolve("deepseek-v4-pro")
    adapter_flash = reg.resolve("deepseek-v4-flash")

    # 协议不同：证明两个模型确实走了两种不同协议适配器
    assert adapter_pro.protocol == "openai-responses"
    assert adapter_flash.protocol == "anthropic-messages"
    # 实例不同：证明是各自独立的适配器对象
    assert adapter_pro is not adapter_flash


def test_unknown_model_raises_404():
    reg = _build_registry()
    with pytest.raises(GatewayError) as exc_info:
        reg.resolve("not-a-real-model")
    assert exc_info.value.code == ERR_MODEL_NOT_FOUND
    assert exc_info.value.http_status == 404


if __name__ == "__main__":
    # 无 pytest 时的简易运行器：直接 `python3 tests/test_routing.py` 即可看到结果。
    # 在有 pytest 的环境，请用 `python3 -m pytest tests/ -v` 运行（含本文件）。
    reg = _build_registry()

    adapter_pro = reg.resolve("deepseek-v4-pro")
    adapter_flash = reg.resolve("deepseek-v4-flash")
    assert adapter_pro.protocol == "openai-responses", "pro 应路由到 openai 适配器"
    assert adapter_flash.protocol == "anthropic-messages", "flash 应路由到 anthropic 适配器"
    assert adapter_pro is not adapter_flash, "两个模型应是不同适配器实例"

    try:
        reg.resolve("not-a-real-model")
        raise AssertionError("未知模型应抛 MODEL_NOT_FOUND")
    except GatewayError as e:
        assert e.code == ERR_MODEL_NOT_FOUND and e.http_status == 404

    print("ROUTING_OK: 两个模型路由到不同适配器，未知模型返回 404")
