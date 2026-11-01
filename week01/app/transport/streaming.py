"""SSE 文本格式化（纯函数，不依赖 Web 框架）。

说明：作业要求流式以「SSE 格式」逐块返回，SSE 是一种简单的文本协议
（每个事件 `data: <负载>\n\n`，结束发 `data: [DONE]\n\n`），并非第三方库。
本文件只负责把统一 StreamChunk 序列化成 SSE 文本字符串，由 server.py 用
fastapi 的 StreamingResponse 逐段推送给客户端。拆成 fastapi 无关的纯函数，
便于离线单测。
"""
import json

from app.core.models import StreamChunk


def sse_chunk_payload(chunk: StreamChunk) -> str:
    """把一个统一 StreamChunk 编码成一行 SSE 文本。"""
    # 复用 pydantic 的 model_dump_json，保证字段与对外契约一致
    return "data: " + chunk.model_dump_json() + "\n\n"


def sse_done() -> str:
    """流式结束哨兵事件。"""
    return "data: [DONE]\n\n"


def sse_error(code: str, message: str) -> str:
    """流式途中出错时，发送一个错误事件（仍走 SSE 通道，避免连接半截中断）。"""
    payload = json.dumps({"error": {"code": code, "message": message}})
    return "data: " + payload + "\n\n"
