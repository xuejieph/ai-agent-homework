"""Step 4 验证：网关编排骨架（模板/路由/限流/重试/观测 串联）。

用桩适配器（不依赖真实或 Mock 协议实现）端到端驱动 Gateway，验证：
    1. 两个模型分别路由到各自适配器，且 /v1/metrics 记录了分类 Token。
    2. 网关层限流：超容量抛 RATE_LIMITED(429)。
    3. 网关层重试已接入：适配器前两次失败后第三次成功，metrics 记 retries=2。

运行方式：
    - 本沙箱：python3 tests/test_orchestration.py
    - 正常环境：python3 -m pytest tests/ -v
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.gateway import Gateway
from app.core.config import Settings
from app.routing.router import AdapterRegistry
from app.adapters.base import BaseAdapter
from app.core.models import ChatRequest, ChatResponse, Message, Choice, Usage
from app.capabilities.observability import MetricsStore
from app.core.errors import ERR_PROVIDER_ERROR, ERR_RATE_LIMITED, GatewayError


# —— 桩适配器：直接回声，便于验证编排而非协议 ——
class _EchoOpenAI(BaseAdapter):
    protocol = "openai-responses"

    async def chat(self, req, stream):
        text = " ".join(m.content for m in req.messages)
        return ChatResponse(
            id="r-openai", created=0, model=req.model,
            choices=[Choice(message=Message(role="assistant", content=text), finish_reason="stop")],
            usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )


class _EchoAnthropic(BaseAdapter):
    protocol = "anthropic-messages"

    async def chat(self, req, stream):
        text = " ".join(m.content for m in req.messages)
        return ChatResponse(
            id="r-anthropic", created=0, model=req.model,
            choices=[Choice(message=Message(role="assistant", content=text), finish_reason="stop")],
            usage=Usage(prompt_tokens=2, completion_tokens=1, total_tokens=3),
        )


class _Faulty(BaseAdapter):
    """前 fail_times 次抛可重试错误，之后成功——用于验证网关重试。"""
    protocol = "openai-responses"

    def __init__(self, name, fail_times=2):
        super().__init__(name)
        self.fail_times = fail_times
        self.calls = 0

    async def chat(self, req, stream):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise GatewayError(ERR_PROVIDER_ERROR, "boom", http_status=502)
        return ChatResponse(
            id="r-recovered", created=0, model=req.model,
            choices=[Choice(message=Message(role="assistant", content="recovered"), finish_reason="stop")],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


def test_complete_routes_both_models_and_records_metrics():
    settings = Settings(
        use_mock=True,
        model_routing={"deepseek-v4-pro": "openai", "deepseek-v4-flash": "anthropic"},
    )
    reg = AdapterRegistry()
    reg.register("openai", _EchoOpenAI("openai"), ["deepseek-v4-pro"])
    reg.register("anthropic", _EchoAnthropic("anthropic"), ["deepseek-v4-flash"])
    metrics = MetricsStore()
    gw = Gateway(settings, registry=reg, metrics=metrics)

    async def run():
        r1 = await gw.complete(ChatRequest(model="deepseek-v4-pro", messages=[Message(role="user", content="hi")]))
        r2 = await gw.complete(ChatRequest(model="deepseek-v4-flash", messages=[Message(role="user", content="hi")]))
        return r1, r2

    r1, r2 = asyncio.run(run())
    assert r1.model == "deepseek-v4-pro" and r1.choices[0].message.content == "hi"
    assert r2.model == "deepseek-v4-flash" and r2.choices[0].message.content == "hi"

    s = metrics.summary()
    assert s["by_model"]["deepseek-v4-pro"]["calls"] == 1
    assert s["by_model"]["deepseek-v4-flash"]["calls"] == 1
    assert s["total_tokens"] == 6  # 两个调用各 3 token


def test_gateway_rate_limit_returns_429():
    # 容量=1：第二次调用应被网关限流
    settings = Settings(
        use_mock=True, model_routing={"m": "openai"},
        rate_limit_capacity=1, rate_limit_window=60.0,
    )
    reg = AdapterRegistry()
    reg.register("openai", _EchoOpenAI("openai"), ["m"])
    gw = Gateway(settings, registry=reg)

    async def run():
        await gw.complete(ChatRequest(model="m", messages=[Message(role="user", content="a")]))
        try:
            await gw.complete(ChatRequest(model="m", messages=[Message(role="user", content="b")]))
            return False
        except GatewayError as e:
            return e.code == ERR_RATE_LIMITED and e.http_status == 429

    assert asyncio.run(run()) is True


def test_gateway_retry_wired():
    settings = Settings(
        use_mock=True, model_routing={"faulty": "faulty"},
        max_retries=3, retry_base_delay=0.001,
    )
    reg = AdapterRegistry()
    reg.register("faulty", _Faulty("faulty", fail_times=2), ["faulty"])
    metrics = MetricsStore()
    gw = Gateway(settings, registry=reg, metrics=metrics)

    async def run():
        return await gw.complete(ChatRequest(model="faulty", messages=[Message(role="user", content="x")]))

    resp = asyncio.run(run())
    assert resp.choices[0].message.content == "recovered"

    s = metrics.summary()
    assert s["by_model"]["faulty"]["retries"] == 2  # 重试 2 次后成功


if __name__ == "__main__":
    test_complete_routes_both_models_and_records_metrics()
    test_gateway_rate_limit_returns_429()
    test_gateway_retry_wired()
    print("ORCHESTRATION_OK: 网关编排骨架（路由/限流/重试/观测）端到端通过")
