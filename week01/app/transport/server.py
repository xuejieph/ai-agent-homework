"""FastAPI 服务入口：把 Gateway 编排层暴露为 HTTP 接口。

端点（本步实现的部分）：
    POST /v1/chat/completions  统一调用入口（stream=true → SSE 流式）
    GET  /v1/models            列出已注册路由的模型
    GET  /v1/metrics           观测汇总（Token 分类 / 延迟 / TTFT / 重试）
    GET  /health               健康检查

提示词管理端点（GET/POST /v1/prompts 等）在 Step 7 接入。

设计要点：
    - 所有横切能力（路由/限流/重试/观测/模板）都在 Gateway 内部完成，server 只做
      「HTTP ↔ 统一契约」的转换与 SSE 包装，保持薄。
    - 统一异常处理器：GatewayError 自动转成标准错误码 JSON（见 errors.error_response）。
    - 流式：用 StreamingResponse 包裹 gateway.stream()，逐个 chunk 编码为 SSE；
      中途若抛 GatewayError，则发送一个 SSE 错误事件再补 [DONE]，保证连接优雅结束。
"""
import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.gateway import Gateway
from app.core.errors import GatewayError, error_response, ERR_PROMPT_NOT_FOUND
from app.core.models import ChatRequest, PromptCreate
from app.capabilities.observability import MetricsStore
from app.prompts.store import PromptStore
from app.routing.router import build_registry
from app.transport.streaming import sse_chunk_payload, sse_done, sse_error

# ——— 应用级单例：配置 / 注册表 / 网关 / 指标 / 模板库 ———
settings = get_settings()
registry = build_registry(settings)
metrics = MetricsStore()
prompt_store = PromptStore(settings.templates_dir)
gateway = Gateway(settings, registry=registry, prompts=prompt_store, metrics=metrics)

app = FastAPI(title="LLM Unified Gateway", version="1.0.0")


@app.exception_handler(GatewayError)
async def _handle_gateway_error(request: Request, exc: GatewayError):
    """统一异常处理器：把网关异常转成标准错误码响应。"""
    return error_response(exc)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """统一调用入口。

    - stream=false：等待完整响应，直接返回 ChatResponse（JSON）。
    - stream=true ：返回 text/event-stream，逐块推送 SSE，结尾 [DONE]。
    """
    if req.stream:
        async def event_stream():
            try:
                async for chunk in gateway.stream(req):
                    yield sse_chunk_payload(chunk)
            except GatewayError as e:
                yield sse_error(e.code, e.message)
            yield sse_done()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # 非流式：完整返回
    return await gateway.complete(req)


@app.get("/v1/models")
async def list_models():
    """列出当前路由表中的所有模型。"""
    return {"data": [{"id": m} for m in registry.list_models()]}


@app.get("/v1/metrics")
async def get_metrics():
    """返回观测汇总（按模型分类的 Token / 延迟 / TTFT / 重试）。"""
    return metrics.summary()


@app.get("/health")
async def health():
    return {"status": "ok"}


# ——————————————— 提示词版本管理（Step 7） ———————————————
@app.get("/v1/prompts")
async def list_prompts():
    """列出所有模板及其版本。"""
    return {"data": prompt_store.list_templates()}


@app.post("/v1/prompts")
async def create_prompt(p: PromptCreate):
    """新增 / 发布一个模板版本（持久化到 YAML）。"""
    prompt_store.add_template(p.id, p.version, p.content, p.description)
    return {"id": p.id, "version": p.version, "status": "created"}


@app.get("/v1/prompts/{template_id}")
async def get_prompt(template_id: str):
    """获取单个模板详情（含全部版本）。"""
    tpl = prompt_store.get_template(template_id)
    if not tpl:
        raise GatewayError(ERR_PROMPT_NOT_FOUND, f"template '{template_id}' not found", 404)
    return tpl


if __name__ == "__main__":
    import uvicorn

    # 仅本地直接运行时生效；生产可用 gunicorn + uvicorn worker。
    uvicorn.run(app, host="0.0.0.0", port=8000)
