"""按模型独立限流（滑动时间窗算法）。

为什么用「滑动窗口」而不是「固定窗口」？
    固定窗口在窗口边界会出现「双倍突发」（如 0s 末和 60s 初各放一波）。
    滑动窗口只统计「过去 window 秒内」的请求数，限流更平滑、更贴近真实速率。

限流粒度：
    每个模型拥有自己的 RateLimiter 实例（由网关在初始化时按 model_routing 创建），
    因此「deepseek-v4-pro 被限流」不会影响「deepseek-v4-flash」——满足作业要求的
    「按模型独立限流」。

超限行为：
    直接抛 GatewayError(RATE_LIMITED, 429)。注意：限流是网关对调用方的约束，
    不在 RETRYABLE 集合内，调用方应自行退避，网关不会重试（避免加剧拥塞）。
"""
import asyncio
import time

from app.core.errors import ERR_RATE_LIMITED, GatewayError


class RateLimiter:
    def __init__(self, capacity: int, window: float):
        """初始化一个限流器。

        参数：
            capacity  时间窗内允许的最大请求数（容量）
            window    时间窗长度（秒）
        """
        self.capacity = capacity
        self.window = window
        self._timestamps: list[float] = []  # 记录窗口内每次请求的到达时间
        self._lock = asyncio.Lock()         # 异步安全：并发请求串行化判定

    async def acquire(self) -> None:
        """尝试获取一个配额；获取不到则抛 429。

        算法步骤：
            1. 加锁，保证并发下计数准确。
            2. 丢弃所有「超过 window 秒」的旧时间戳（滑出窗口）。
            3. 若窗口内请求数 >= 容量，计算还需等待多久（retry_after）并拒绝。
            4. 否则记录当前时间，放行。
        """
        async with self._lock:
            now = time.time()
            # 滑动：只保留 [now - window, now) 内的请求
            self._timestamps = [t for t in self._timestamps if now - t < self.window]

            if len(self._timestamps) >= self.capacity:
                # 最早那条何时滑出窗口，即为可重试的等待时间（给调用方参考）
                retry_after = self.window - (now - self._timestamps[0])
                raise GatewayError(
                    ERR_RATE_LIMITED,
                    f"rate limit exceeded (capacity={self.capacity}/{self.window}s)",
                    http_status=429,
                    details={"retry_after_s": round(max(0.0, retry_after), 2)},
                )

            self._timestamps.append(now)
