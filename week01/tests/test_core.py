"""Step 3 验证：限流 / 重试 / 可观测 三个独立模块。

这些模块不依赖 fastapi，可在任意环境直接验证：
    - 本沙箱：python3 tests/test_core.py   （走 __main__ 运行器）
    - 正常环境：python3 -m pytest tests/ -v

测试函数均为同步函数，内部用 asyncio.run 驱动异步逻辑，
因此不需要 pytest-asyncio 这类插件即可被 pytest 收集执行。
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.capabilities.rate_limiter import RateLimiter
from app.capabilities.retry import with_retries
from app.capabilities.observability import MetricsStore, CallRecord
from app.core.errors import ERR_RATE_LIMITED, ERR_MODEL_NOT_FOUND, ERR_PROVIDER_ERROR, GatewayError


# ——————————————— 限流 ———————————————
def test_rate_limit_triggers_429():
    async def scenario():
        # 容量=2：前两次放行，第三次应被限流
        limiter = RateLimiter(capacity=2, window=60.0)
        await limiter.acquire()
        await limiter.acquire()
        try:
            await limiter.acquire()
            return False  # 没抛异常说明限流失效
        except GatewayError as e:
            return e.code == ERR_RATE_LIMITED and e.http_status == 429

    assert asyncio.run(scenario()) is True


# ——————————————— 重试 ———————————————
def test_retry_succeeds_after_failures():
    async def flaky(attempt):
        # 前两次抛可重试错误，第三次成功
        if attempt < 2:
            raise GatewayError(ERR_PROVIDER_ERROR, "boom", http_status=502)
        return "ok"

    result, attempts = asyncio.run(with_retries(flaky, max_retries=3, base_delay=0.001))
    assert result == "ok"
    assert attempts == 3  # 首次 + 2 次重试


def test_retry_skips_non_retryable():
    async def bad(_):  # 不可重试错误（模型不存在），不应重试
        raise GatewayError(ERR_MODEL_NOT_FOUND, "x", http_status=404)

    try:
        asyncio.run(with_retries(bad, max_retries=3, base_delay=0.001))
        assert False, "不可重试错误不应被重试"
    except GatewayError as e:
        assert e.code == ERR_MODEL_NOT_FOUND
        assert e.attempts == 1  # 只尝试了 1 次


def test_retry_exhausted_after_max():
    async def always(_):  # 永远失败（可重试），应在 max_retries 后放弃
        raise GatewayError(ERR_PROVIDER_ERROR, "boom", http_status=502)

    try:
        asyncio.run(with_retries(always, max_retries=3, base_delay=0.001))
        assert False, "应耗尽重试后抛出"
    except GatewayError as e:
        assert e.attempts == 4  # 首次 + 3 次重试


# ——————————————— 可观测 ———————————————
def test_metrics_summary_categorizes():
    store = MetricsStore()
    store.record(CallRecord(
        id="1", model="deepseek-v4-pro", protocol="openai-responses", status="ok",
        prompt_tokens=10, completion_tokens=5, total_tokens=15,
        latency_ms=100.0, ttft_ms=50.0, retries=0, timestamp=1.0,
    ))
    store.record(CallRecord(
        id="2", model="deepseek-v4-flash", protocol="anthropic-messages", status="ok",
        prompt_tokens=20, completion_tokens=10, total_tokens=30,
        latency_ms=200.0, ttft_ms=80.0, retries=1, timestamp=2.0,
    ))

    s = store.summary()
    assert s["total_calls"] == 2
    assert s["total_tokens"] == 45
    assert s["total_prompt_tokens"] == 30
    assert s["total_completion_tokens"] == 15
    assert s["total_retries"] == 1
    # 按模型分类正确
    assert s["by_model"]["deepseek-v4-pro"]["calls"] == 1
    assert s["by_model"]["deepseek-v4-flash"]["retries"] == 1
    assert s["by_model"]["deepseek-v4-pro"]["ttft_ms"] == [50.0]
    assert s["by_model"]["deepseek-v4-flash"]["latency_ms"] == [200.0]


if __name__ == "__main__":
    test_rate_limit_triggers_429()
    test_retry_succeeds_after_failures()
    test_retry_skips_non_retryable()
    test_retry_exhausted_after_max()
    test_metrics_summary_categorizes()
    print("CORE_OK: 限流/重试/可观测 三模块单测通过")
