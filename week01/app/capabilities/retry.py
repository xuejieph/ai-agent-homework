"""指数退避重试（Exponential Backoff with Jitter）。

为什么需要重试？
    上游 LLM 服务偶发 5xx / 超时 / 上游 429，这些「暂时性」故障值得再试，
    能显著提升网关成功率。但普通 4xx（如模型不存在）是确定失败，重试无意义。

本模块的设计要点：
    1. 只对 RETRYABLE 集合里的错误重试（见 errors.RETRYABLE：PROVIDER_ERROR/TIMEOUT）。
    2. 退避公式：delay = min(base * 2^attempt, max_delay) + 随机抖动。
       - 指数增长避免「失败—立即重试—再失败」的密集冲击（惊群）。
       - max_delay 封顶，防止等待时间过长。
       - 抖动（jitter）让一批同时失败的重试错开，进一步削峰。
    3. 最多重试 max_retries 次（不含首次），达到上限或遇不可重试错误立即抛出。

返回：
    (result, attempts) —— attempts 含首次，方便上层把「重试次数」记入可观测数据。
"""
import asyncio
import random

from app.core.errors import RETRYABLE, GatewayError


async def with_retries(fn, max_retries: int = 3, base_delay: float = 0.2, max_delay: float = 5.0):
    """对异步函数 fn(attempt) 执行指数退避重试。

    参数：
        fn           被重试的异步可调用对象，接收当前尝试序号（从 0 开始）
        max_retries  最大重试次数（不含首次）
        base_delay   退避基数（秒）
        max_delay    单次等待上限（秒）

    返回：
        (result, attempts)，attempts 为实际尝试总次数（含首次）

    抛出：
        最后一次仍失败时的 GatewayError（其 .attempts 会被写入尝试次数）
    """
    last: GatewayError | None = None
    attempts = 0

    # 循环范围 [0, max_retries]，即「首次 + max_retries 次重试」
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        try:
            result = await fn(attempt)
            return result, attempts
        except GatewayError as e:
            # 不可重试的错误，或已达最大次数 → 直接抛出，不再等待
            if e.code not in RETRYABLE or attempt >= max_retries:
                e.attempts = attempts
                raise
            last = e
            # 指数退避 + 抖动
            delay = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, base_delay)
            await asyncio.sleep(delay)

    # 理论不会走到这里（循环内必抛），但保险起见把最后错误补上尝试次数再抛出
    last.attempts = attempts  # type: ignore[union-attr]
    raise last
