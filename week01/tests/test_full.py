"""Step 8 统一验证脚本：串联「六大功能点」的可验证证据。

本文件把作业验收标准拆成 6 个检查，全部走真实编排（Gateway + Mock 双协议适配器），
不依赖 fastapi / pytest 也能直接 `python3 tests/test_full.py` 看结果；在有 pytest 的
环境也可用 `python3 -m pytest tests/test_full.py -v` 跑。

六大功能点：
    ① 两个模型调用正常（不同协议）
    ② 流式输出（SSE 分块 + 结束）
    ③ 结构化输出（返回合法 JSON）
    ④ 提示词模板引用（变量替换后内容到达适配器）
    ⑤ 可观测数据（/v1/metrics 的分类 Token / 延迟 / 首 Token 延迟 / 重试）
    ⑥ 重试 + 限流（失败重试恢复；超限返回 429）

每个检查只做「断言 + 打印证据」，便于对照验收标准。
"""
import asyncio
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

from app.core.config import Settings
from app.routing.router import AdapterRegistry, build_registry
from app.gateway import Gateway
from app.core.models import ChatRequest, ChatResponse, Message, Choice, Usage
from app.core.errors import ERR_RATE_LIMITED, GatewayError
from app.adapters.base import BaseAdapter
from app.prompts.store import PromptStore


# ————————————————————————————————————————————————
# ⑥ 用到的辅助适配器/存储
# ————————————————————————————————————————————————
class _CapturingAdapter(BaseAdapter):
    """测试专用：把收到的 system 消息内容原样回显到回答，用于证明「模板变量替换后
    内容确实到达了适配器」（模板经网关注入 system，再被本适配器捕获）。"""
    protocol = "capture"

    def __init__(self):
        super().__init__("capture")
        self.last_messages = None

    async def chat(self, req: ChatRequest, stream: bool):
        self.last_messages = list(req.messages)
        system = next((m.content for m in req.messages if m.role == "system"), "")
        return ChatResponse(
            id="cap", created=0, model=req.model,
            choices=[Choice(message=Message(role="assistant", content=system), finish_reason="stop")],
            usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )


# ————————————————————————————————————————————————
# ① 两个模型调用正常（不同协议）
# ————————————————————————————————————————————————
def check_two_models_work():
    settings = Settings(use_mock=True, model_routing={
        "deepseek-v4-pro": "openai", "deepseek-v4-flash": "anthropic"})
    gw = Gateway(settings, registry=build_registry(settings))

    async def run():
        r1 = await gw.complete(ChatRequest(model="deepseek-v4-pro", messages=[Message(role="user", content="hi")]))
        r2 = await gw.complete(ChatRequest(model="deepseek-v4-flash", messages=[Message(role="user", content="hi")]))
        return r1, r2

    r1, r2 = asyncio.run(run())
    assert r1.model == "deepseek-v4-pro" and r1.choices[0].message.content
    assert r2.model == "deepseek-v4-flash" and r2.choices[0].message.content
    # 两个模型协议不同（核心：两种 API 协议被统一）
    assert gw.registry.resolve("deepseek-v4-pro").protocol == "openai-responses"
    assert gw.registry.resolve("deepseek-v4-flash").protocol == "anthropic-messages"
    print("  [①] 两模型调用正常：pro→openai-responses，flash→anthropic-messages，均返回内容")


# ————————————————————————————————————————————————
# ② 流式输出
# ————————————————————————————————————————————————
def check_streaming():
    settings = Settings(use_mock=True, model_routing={"deepseek-v4-pro": "openai"})
    gw = Gateway(settings, registry=build_registry(settings))

    async def run():
        chunks = []
        async for c in gw.stream(ChatRequest(
                model="deepseek-v4-pro", messages=[Message(role="user", content="abc")], stream=True)):
            chunks.append(c)
        return chunks

    chunks = asyncio.run(run())
    texts = [c.choices[0]["delta"].get("content", "") for c in chunks if c.choices[0]["delta"]]
    joined = "".join(texts)
    assert joined == "[deepseek-v4-pro] abc"          # 流式逐块拼接正确
    assert chunks[-1].choices[0]["finish_reason"] == "stop"  # 结束块
    print(f"  [②] 流式输出：拼接='{joined}'，末块 finish_reason=stop")


# ————————————————————————————————————————————————
# ③ 结构化输出
# ————————————————————————————————————————————————
def check_structured_output():
    settings = Settings(use_mock=True, model_routing={
        "deepseek-v4-pro": "openai", "deepseek-v4-flash": "anthropic"})
    gw = Gateway(settings, registry=build_registry(settings))

    async def run():
        r1 = await gw.complete(ChatRequest(model="deepseek-v4-pro", messages=[Message(role="user", content="x")],
                                          response_format={"type": "json_object"}))
        r2 = await gw.complete(ChatRequest(model="deepseek-v4-flash", messages=[Message(role="user", content="x")],
                                          response_format={"type": "json_object"}))
        return r1, r2

    r1, r2 = asyncio.run(run())
    # 两种协议下，结构化输出都应返回可被 json.loads 解析的内容
    json.loads(r1.choices[0].message.content)
    json.loads(r2.choices[0].message.content)
    print("  [③] 结构化输出：pro/flash 返回内容均 json.loads 通过")


# ————————————————————————————————————————————————
# ④ 提示词模板引用（变量替换后内容到达适配器）
# ————————————————————————————————————————————————
def check_prompt_template():
    import tempfile
    d = tempfile.mkdtemp()
    store = PromptStore(d)
    store.add_template("greet", "v1", "你好 {{ name }}，你是{{ role }}。", "demo")

    settings = Settings(use_mock=True, model_routing={"cap-model": "capture"})
    reg = AdapterRegistry()
    cap = _CapturingAdapter()
    reg.register("capture", cap, ["cap-model"])
    gw = Gateway(settings, registry=reg, prompts=store)

    async def run():
        return await gw.complete(ChatRequest(
            model="cap-model", messages=[Message(role="user", content="hi")],
            prompt_template="greet", prompt_version="v1", prompt_variables={"name": "Alice", "role": "助手"}))

    resp = asyncio.run(run())
    # 变量替换后内容到达了适配器，并被回显到回答
    assert "Alice" in resp.choices[0].message.content
    assert cap.last_messages[0].role == "system"
    assert "Alice" in cap.last_messages[0].content
    print("  [④] 提示词模板引用：变量替换后 system 内容='%s'" % cap.last_messages[0].content.strip())


# ————————————————————————————————————————————————
# ⑤ 可观测数据（分类 Token / 延迟 / TTFT / 重试）
# ————————————————————————————————————————————————
def check_observability():
    settings = Settings(use_mock=True, model_routing={"deepseek-v4-pro": "openai"})
    gw = Gateway(settings, registry=build_registry(settings))

    async def run():
        await gw.complete(ChatRequest(model="deepseek-v4-pro", messages=[Message(role="user", content="观测")]))
        # 再跑一次流式，产生 ttft 记录
        async for _ in gw.stream(ChatRequest(
                model="deepseek-v4-pro", messages=[Message(role="user", content="观测2")], stream=True)):
            pass

    asyncio.run(run())
    summary = gw.metrics.summary()
    by = summary["by_model"].get("deepseek-v4-pro")
    assert by is not None
    assert by["total_tokens"] > 0            # 分类 Token 统计
    assert by["calls"] == 2
    assert len(by["latency_ms"]) == 2         # 延迟记录（保留原始列表）
    assert any(t is not None for t in by["ttft_ms"])  # 首 Token 延迟（流式那次非空）
    print(f"  [⑤] 可观测数据：by_model['deepseek-v4-pro'] 总token={by['total_tokens']}，"
          f"调用={by['calls']}，含TTFT记录")


# ————————————————————————————————————————————————
# ⑥ 重试 + 限流（网关级证据）
# ————————————————————————————————————————————————
def check_resilience():
    # 6a 重试：FaultyAdapter 前 2 次失败，第 3 次（含重试）成功
    s1 = Settings(use_mock=True,
                  model_routing={"faulty-model": "faulty", "deepseek-v4-pro": "openai"},
                  retry_base_delay=0.001)
    gw1 = Gateway(s1, registry=build_registry(s1))

    async def run_retry():
        return await gw1.complete(ChatRequest(model="faulty-model", messages=[Message(role="user", content="x")]))

    resp = asyncio.run(run_retry())
    assert resp.choices[0].message.content
    by = gw1.metrics.summary()["by_model"].get("faulty-model")
    assert by["retries"] >= 2, "重试后应在 metrics 记录 retries>0"
    print(f"  [⑥a] 重试恢复：faulty-model 重试后成功，metrics.retries={by['retries']}")

    # 6b 限流：容量=1，第二次同模型调用返回 429
    s2 = Settings(use_mock=True, model_routing={"deepseek-v4-pro": "openai"},
                  rate_limit_capacity=1, rate_limit_window=60.0)
    gw2 = Gateway(s2, registry=build_registry(s2))

    async def run_limit():
        await gw2.complete(ChatRequest(model="deepseek-v4-pro", messages=[Message(role="user", content="a")]))
        try:
            await gw2.complete(ChatRequest(model="deepseek-v4-pro", messages=[Message(role="user", content="b")]))
            return None
        except GatewayError as e:
            return e

    err = asyncio.run(run_limit())
    assert err is not None and err.code == ERR_RATE_LIMITED and err.http_status == 429
    print("  [⑥b] 限流：同模型第二次调用返回 429（RATE_LIMITED）")


ALL_CHECKS = [
    check_two_models_work,
    check_streaming,
    check_structured_output,
    check_prompt_template,
    check_observability,
    check_resilience,
]


def run_all():
    print("=== 统一验证：六大功能点证据 ===")
    for fn in ALL_CHECKS:
        fn()
    print("ALL_FEATURES_OK: 六大功能点（含重试/限流）均有可验证证据")


if __name__ == "__main__":
    run_all()
