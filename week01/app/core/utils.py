"""公共小工具。

目前仅提供 token 估算。之所以放在独立模块，是因为 Mock 适配器和网关观测层
都会用到同一套估算口径，避免各处实现不一致。
"""
from typing import Union


def estimate_tokens(text: Union[str, None]) -> int:
    """粗略估算一段文本的 token 数量。

    为什么需要估算？
        真实 provider 会在响应里返回精确 token 数；但 Mock 适配器没有真模型，
        网关在「流式场景下也可能拿不到上游 usage」，此时需要一个近似口径来保证
        可观测数据不为空、且横向可比。

    算法说明：
        以「每 4 个字符 ≈ 1 个 token」做近似（中英文混排下的经验值）。
        最少返回 1，避免空串统计成 0 干扰汇总。
    """
    if not text:
        return 0
    return max(1, len(text) // 4)
