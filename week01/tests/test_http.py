"""Step 8（收尾加强）：真实 HTTP 层端到端验证。

环境装好 fastapi 后，本文件用 FastAPI 的 TestClient 把六大功能点打到「真实运行的
server.app」上验证——比网关层（test_full.py）更贴近作业验收的真实场景
（HTTP 接口 + SSE 流式 + 统一错误码）。

覆盖：
    ① 两模型非流式调用（不同协议）
    ② 流式 SSE（data: 分块 + [DONE]）
    ③ 结构化输出（HTTP 返回合法 JSON）
    ④ 提示词模板发布/引用/列出（变量替换逻辑见 test_prompts/test_full，此处验证 HTTP 链路）
    ⑤ /v1/metrics 端点（分类 Token 统计）
    ⑥ 重试（FaultyAdapter 经 HTTP 恢复）+ 限流（容量=1 返回 429）

可离线 `python3 tests/test_http.py` 运行，也可 `python3 -m pytest tests/test_http.py -v`。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    pytest = None  # type: ignore[assignment]
    HAS_PYTEST = False

from fastapi.testclient import TestClient

import app.transport.server as server
from app.adapters.mock import FaultyAdapter
from app.capabilities.rate_limiter import RateLimiter


def _client() -> TestClient:
    """每个用例拿一个全新的 TestClient（共享同一 app 实例，便于读 metrics）。"""
    return TestClient(server.app)


# ————————————————————————————————————————————————
# ① 两模型非流式调用（不同协议）
# ————————————————————————————————————————————————
def test_two_models_non_stream():
    c = _client()
    r1 = c.post("/v1/chat/completions", json={
        "model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}]})
    r2 = c.post("/v1/chat/completions", json={
        "model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]})
    assert r1.status_code == 200 and r1.json()["model"] == "deepseek-v4-pro"
    assert r2.status_code == 200 and r2.json()["model"] == "deepseek-v4-flash"


# ————————————————————————————————————————————————
# ② 流式 SSE
# ————————————————————————————————————————————————
def test_stream_sse_over_http():
    c = _client()
    with c.stream("POST", "/v1/chat/completions", json={
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "abc"}],
            "stream": True}) as resp:
        body = resp.read().decode()

    assert "[DONE]" in body                      # 结束哨兵
    # 解析每个 data: 行，拼接 content，验证逐块正确
    texts = []
    for line in body.split("\n"):
        if not line.startswith("data: ") or "[DONE]" in line:
            continue
        obj = json.loads(line[len("data: "):])
        delta = obj["choices"][0]["delta"]
        if delta.get("content"):
            texts.append(delta["content"])
    assert "".join(texts) == "[deepseek-v4-pro] abc"


# ————————————————————————————————————————————————
# ③ 结构化输出（HTTP 返回合法 JSON）
# ————————————————————————————————————————————————
def test_structured_output_over_http():
    c = _client()
    for model in ("deepseek-v4-pro", "deepseek-v4-flash"):
        r = c.post("/v1/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": "x"}],
            "response_format": {"type": "json_object"}})
        assert r.status_code == 200
        # 两种协议下结构化输出都应可被 json.loads 解析
        json.loads(r.json()["choices"][0]["message"]["content"])


# ————————————————————————————————————————————————
# ④ 提示词模板：发布 / 引用 / 列出（HTTP 链路）
# ————————————————————————————————————————————————
def test_prompt_template_endpoints():
    c = _client()
    # 发布模板
    r = c.post("/v1/prompts", json={
        "id": "greet", "version": "v1",
        "content": "你好 {{ name }}，你是{{ role }}。", "description": "demo"})
    assert r.status_code == 200
    # 引用模板（HTTP 链路不报错即证明模板解析→注入→调用 全流程通畅；
    # 变量替换的精确内容证据见 test_prompts / test_full ④）
    r2 = c.post("/v1/chat/completions", json={
        "model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}],
        "prompt_template": "greet", "prompt_version": "v1",
        "prompt_variables": {"name": "Alice", "role": "助手"}})
    assert r2.status_code == 200
    # 列出 / 详情
    assert any(t["id"] == "greet" for t in c.get("/v1/prompts").json()["data"])
    assert c.get("/v1/prompts/greet").status_code == 200


# ————————————————————————————————————————————————
# ⑤ /v1/metrics 端点（分类 Token 统计）
# ————————————————————————————————————————————————
def test_metrics_endpoint():
    c = _client()
    c.post("/v1/chat/completions", json={
        "model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "m"}]})
    m = c.get("/v1/metrics").json()
    assert "by_model" in m and "deepseek-v4-pro" in m["by_model"]
    assert m["by_model"]["deepseek-v4-pro"]["total_tokens"] > 0


# ————————————————————————————————————————————————
# ⑥ 重试 + 限流（经真实 HTTP）
# ————————————————————————————————————————————————
def test_retry_over_http():
    # 把 FaultyAdapter 注册给 faulty-model（默认 model_routing 不含它，需手动注入）
    server.gateway.registry.register("faulty", FaultyAdapter("faulty", fail_times=2), ["faulty-model"])
    server.gateway.rate_limiters["faulty-model"] = RateLimiter(server.settings.rate_limit_capacity, 60.0)

    c = _client()
    r = c.post("/v1/chat/completions", json={
        "model": "faulty-model", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200                       # 重试后恢复
    m = c.get("/v1/metrics").json()
    assert m["by_model"]["faulty-model"]["retries"] >= 2  # metrics 记录重试


def test_rate_limit_429_over_http():
    # 把该模型限流容量临时调成 1，验证超限返回 429
    server.gateway.rate_limiters["deepseek-v4-pro"] = RateLimiter(1, 60.0)
    try:
        c = _client()
        c.post("/v1/chat/completions", json={
            "model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "a"}]})
        r = c.post("/v1/chat/completions", json={
            "model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "b"}]})
        assert r.status_code == 429
        assert r.json()["error"]["code"] == "RATE_LIMITED"
    finally:
        # 还原，避免影响其它用例
        server.gateway.rate_limiters["deepseek-v4-pro"] = RateLimiter(
            server.settings.rate_limit_capacity, server.settings.rate_limit_window)


ALL_HTTP_TESTS = [
    test_two_models_non_stream, test_stream_sse_over_http, test_structured_output_over_http,
    test_prompt_template_endpoints, test_metrics_endpoint, test_retry_over_http,
    test_rate_limit_429_over_http,
]


def run_all():
    print("=== HTTP 层端到端验证（六大功能点）===")
    for fn in ALL_HTTP_TESTS:
        fn()
        print(f"  [OK] {fn.__name__}")
    print("HTTP_ALL_OK: 六大功能点经真实 FastAPI 服务验证通过")


if __name__ == "__main__":
    run_all()
