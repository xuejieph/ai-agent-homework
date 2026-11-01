"""可观测性：调用指标存储（MetricsStore）。

作业要求记录：
    - 每次调用的 Token 消耗（含分类统计：prompt / completion / total）
    - 延迟（latency_ms），以及首 Token 延迟（ttft_ms，流式场景尤其重要）

设计说明：
    - 用线程锁保证并发写入安全（FastAPI 多线程/异步下都可能写）。
    - summary() 在内存里做聚合，按模型分类，方便 /v1/metrics 直接返回。
    - 本作业只做内存聚合，不持久化（生产可换 PostgreSQL/ClickHouse，见 1-7 思路）。
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CallRecord:
    """单次调用的观测记录。"""

    id: str
    model: str                 # 模型名（用于分类）
    protocol: str              # 走的协议（调试用）
    status: str                # "ok" | "error"
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float          # 端到端延迟（毫秒）
    ttft_ms: Optional[float]   # 首 Token 延迟；非流式时为总延迟，失败时为 None
    retries: int               # 本次调用的重试次数
    timestamp: float
    error: Optional[str] = None  # 失败时的错误码


class MetricsStore:
    """内存指标仓库：记录 + 按模型聚合汇总。"""

    def __init__(self):
        self._records: List[CallRecord] = []
        self._lock = threading.Lock()

    def record(self, rec: CallRecord) -> None:
        """写入一条调用记录。"""
        with self._lock:
            self._records.append(rec)

    def summary(self) -> Dict:
        """聚合出可观测汇总。

        返回结构（与 /v1/metrics 对接）：
            total_calls / total_tokens / total_prompt_tokens /
            total_completion_tokens / total_retries
            by_model[model] = {
                calls, errors, prompt_tokens, completion_tokens,
                total_tokens, retries, latency_ms[], ttft_ms[]
            }
        latency_ms / ttft_ms 保留原始列表，便于后续算 min/max/avg/p99。
        """
        with self._lock:
            records = list(self._records)

        by_model: Dict[str, dict] = {}
        total_tokens = total_prompt = total_completion = total_retries = 0

        for r in records:
            total_tokens += r.total_tokens
            total_prompt += r.prompt_tokens
            total_completion += r.completion_tokens
            total_retries += r.retries

            m = by_model.setdefault(r.model, {
                "calls": 0, "errors": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "total_tokens": 0,
                "retries": 0, "latency_ms": [], "ttft_ms": [],
            })
            m["calls"] += 1
            m["prompt_tokens"] += r.prompt_tokens
            m["completion_tokens"] += r.completion_tokens
            m["total_tokens"] += r.total_tokens
            m["retries"] += r.retries
            if r.status != "ok":
                m["errors"] += 1
            m["latency_ms"].append(round(r.latency_ms, 2))
            if r.ttft_ms is not None:
                m["ttft_ms"].append(round(r.ttft_ms, 2))

        return {
            "total_calls": len(records),
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_retries": total_retries,
            "by_model": by_model,
        }

    def reset(self) -> None:
        """清空所有记录（测试隔离用）。"""
        with self._lock:
            self._records.clear()
