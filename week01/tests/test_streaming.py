"""Step 6 验证（改名）：SSE 文本格式化（纯函数，无需 fastapi）。

验证点：
    1. sse_chunk_payload 输出形如 'data: <合法JSON>\n\n'，且 JSON 可被解析还原。
    2. sse_done 输出 'data: [DONE]\n\n'。
    3. sse_error 输出 'data: {"error": {...}}\n\n' 且字段正确。

注：真正的「fastapi StreamingResponse 推送 SSE」由 server.py 负责，需在你本机
（已装 fastapi）运行服务后，用 requests/httpx 读取流体验证；本文件只验证格式本身。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

from app.transport.streaming import sse_chunk_payload, sse_done, sse_error
from app.core.models import StreamChunk


def test_sse_chunk_payload_is_valid_sse():
    chunk = StreamChunk(
        id="c1", created=123, model="deepseek-v4-pro",
        choices=[{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
    )
    out = sse_chunk_payload(chunk)
    assert out.startswith("data: ")
    assert out.endswith("\n\n")
    payload = out[len("data: "):-2]
    parsed = json.loads(payload)
    assert parsed["model"] == "deepseek-v4-pro"
    assert parsed["choices"][0]["delta"]["content"] == "hi"


def test_sse_done_sentinel():
    assert sse_done() == "data: [DONE]\n\n"


def test_sse_error_event():
    out = sse_error("RATE_LIMITED", "too many")
    assert out.startswith("data: ")
    assert out.endswith("\n\n")
    parsed = json.loads(out[len("data: "):-2])
    assert parsed["error"]["code"] == "RATE_LIMITED"
    assert parsed["error"]["message"] == "too many"


if __name__ == "__main__":
    test_sse_chunk_payload_is_valid_sse()
    test_sse_done_sentinel()
    test_sse_error_event()
    print("STREAMING_OK: SSE 文本格式化（chunk/DONE/error）格式正确（由 fastapi StreamingResponse 推送）")
