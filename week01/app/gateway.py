"""网关编排层（Gateway）。

这是整个服务的「中枢」，把多个横切能力按固定顺序编排到每一次模型调用上：

    请求进入
      │
      ├─ ① 模板解析  _apply_template   （若请求引用了提示词模板，渲染后注入 system）
      ├─ ② 动态路由  registry.resolve   （按 model 找到具体适配器）
      ├─ ③ 限流      _rate_limit       （按模型独立滑动窗口，超限 429）
      ├─ ④ 重试      with_retries      （指数退避，仅可重试错误，最多 3 次）
      └─ ⑤ 观测      metrics.record    （记录 Token 分类、延迟、首 Token 延迟、重试数）

非流式 complete() 与流式 stream() 共用 ①②③⑤ 这四步；差别只在第④步之后如何
消费适配器返回（一次性拿 ChatResponse vs 逐块迭代 StreamChunk）。

设计原则：
    - 网关只依赖 BaseAdapter 抽象接口与统一数据契约，不感知具体协议。
    - 各能力都是独立模块（见 rate_limiter / retry / observability / prompts），
      这里只做「编排」，不做协议翻译。
"""
import time
import uuid

from app.adapters.base import BaseAdapter
from app.core.config import Settings
from app.core.errors import GatewayError
from app.core.models import ChatRequest, ChatResponse, Message, StreamChunk, Usage, Choice
from app.capabilities.observability import CallRecord, MetricsStore
from app.prompts.store import PromptStore
from app.capabilities.rate_limiter import RateLimiter
from app.capabilities.retry import with_retries
from app.routing.router import AdapterRegistry, build_registry
from app.core.utils import estimate_tokens


class Gateway:
    """统一调用编排器。"""

    def __init__(
        self,
        settings: Settings,
        registry: AdapterRegistry | None = None,
        prompts: PromptStore | None = None,
        metrics: MetricsStore | None = None,
    ):
        self.settings = settings
        # 适配器注册表：测试时可注入自定义 registry；否则按配置构建（Step 5 填适配器）
        self.registry = registry or build_registry(settings)
        self.prompts = prompts or PromptStore(settings.templates_dir)
        self.metrics = metrics or MetricsStore()
        # 按模型独立限流器；额外保留一个 __default__ 兜底未知模型
        self.rate_limiters = self._build_limiters(settings)

    # ——————————————————— 初始化辅助 ———————————————————
    def _build_limiters(self, settings: Settings) -> dict[str, RateLimiter]:
        """为路由表中的每个模型各建一个限流器，并加一个默认兜底。"""
        limiters: dict[str, RateLimiter] = {}
        for model in settings.model_routing:
            limiters[model] = RateLimiter(settings.rate_limit_capacity, settings.rate_limit_window)
        limiters["__default__"] = RateLimiter(settings.rate_limit_capacity, settings.rate_limit_window)
        return limiters

    # ——————————————————— 横切能力 ———————————————————
    def _apply_template(self, req: ChatRequest) -> None:
        """① 模板解析：若请求引用了提示词模板，渲染后插到消息最前作为 system。

        注意：实际的渲染（Jinja2 变量替换 + 版本定位）由 PromptStore.render 完成，
        该方法在 Step 7 实现；本步若未设置 prompt_template，则直接跳过，不受影响。
        """
        if not req.has_prompt_template():
            return
        version, rendered = self.prompts.render(
            req.prompt_template, req.prompt_version, req.prompt_variables
        )
        # 插到最前，使模板内容作为系统指令优先于用户消息
        req.messages.insert(0, Message(role="system", content=rendered))

    async def _rate_limit(self, model: str) -> None:
        """③ 限流：按模型取到对应限流器并获取配额（超限抛 429）。"""
        limiter = self.rate_limiters.get(model, self.rate_limiters["__default__"])
        await limiter.acquire()

    # ——————————————————— 非流式入口 ———————————————————
    async def complete(self, req: ChatRequest) -> ChatResponse:
        """执行一次非流式调用，返回完整 ChatResponse。

        编排顺序：模板 → 路由 → 限流 → 重试(调用适配器) → 观测。
        """
        self._apply_template(req)
        adapter: BaseAdapter = self.registry.resolve(req.model)
        await self._rate_limit(req.model)

        start = time.time()
        try:
            # ④ 重试包裹适配器调用；attempts 含首次
            response, attempts = await with_retries(
                lambda _: adapter.chat(req, stream=False),
                self.settings.max_retries,
                self.settings.retry_base_delay,
            )
            latency_ms = (time.time() - start) * 1000
            # ⑤ 观测：非流式没有「首 token」之分，TTFT 即总延迟
            self.metrics.record(CallRecord(
                id=response.id,
                model=req.model,
                protocol=adapter.protocol,
                status="ok",
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=latency_ms,
                ttft_ms=latency_ms,
                retries=attempts - 1,
                timestamp=time.time(),
            ))
            return response
        except GatewayError as e:
            latency_ms = (time.time() - start) * 1000
            self.metrics.record(CallRecord(
                id=uuid.uuid4().hex[:8],
                model=req.model,
                protocol=adapter.protocol,
                status="error",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=latency_ms,
                ttft_ms=None,
                retries=0,
                timestamp=time.time(),
                error=e.code,
            ))
            raise

    # ——————————————————— 流式入口 ———————————————————
    async def stream(self, req: ChatRequest):
        """执行一次流式调用，产出归一化后的 StreamChunk 异步迭代器。

        流式下的重试只覆盖「建立连接/生成器创建前」的失败（避免半截流重复）；
        一旦开始产出内容，中途错误不再重试，由上层发送 SSE 错误事件结束。
        """
        self._apply_template(req)
        adapter: BaseAdapter = self.registry.resolve(req.model)
        await self._rate_limit(req.model)

        start = time.time()
        ttft_ms = None
        completion_text = ""
        prompt_text = " ".join(m.content for m in req.messages)

        # ④ 重试包裹「生成器创建」；真正的内容迭代在下面进行，不再重试
        gen, attempts = await with_retries(
            lambda _: adapter.chat(req, stream=True),
            self.settings.max_retries,
            self.settings.retry_base_delay,
        )
        try:
            async for chunk in gen:
                # 累计补全文本，用于流式场景下估算 token（上游可能不返回 usage）
                for choice in chunk.choices:
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        completion_text += delta["content"]
                # 首块到达即打点首 Token 延迟
                if ttft_ms is None:
                    ttft_ms = (time.time() - start) * 1000
                yield chunk

            latency_ms = (time.time() - start) * 1000
            comp_tokens = estimate_tokens(completion_text)
            self.metrics.record(CallRecord(
                id=uuid.uuid4().hex[:8],
                model=req.model,
                protocol=adapter.protocol,
                status="ok",
                prompt_tokens=estimate_tokens(prompt_text),
                completion_tokens=comp_tokens,
                total_tokens=estimate_tokens(prompt_text) + comp_tokens,
                latency_ms=latency_ms,
                ttft_ms=ttft_ms,
                retries=attempts - 1,
                timestamp=time.time(),
            ))
        except GatewayError as e:
            latency_ms = (time.time() - start) * 1000
            self.metrics.record(CallRecord(
                id=uuid.uuid4().hex[:8],
                model=req.model,
                protocol=adapter.protocol,
                status="error",
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                latency_ms=latency_ms, ttft_ms=None, retries=0,
                timestamp=time.time(), error=e.code,
            ))
            raise
