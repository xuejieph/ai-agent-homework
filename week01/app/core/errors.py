"""统一错误码层（Unified Error Codes）。

为什么需要统一错误码？
    网关之下接了多种上游协议，各家错误格式千差万别。如果把这些原始错误直接
    抛给调用方，调用方就得为每种供应商写不同的错误处理逻辑。因此我们在网关边界
    统一收敛成一套「错误码 + 消息 + HTTP 状态」，调用方只看这套即可。

用法约定：
    1. 网关/适配器内部统一抛出 GatewayError(code, message, http_status, details)。
    2. FastAPI 的异常处理器（见 server.py）捕获后，调用 error_response() 转成
       标准 JSON：{"error": {"code": ..., "message": ..., "details": ...}}。
    3. RETRYABLE 集合标记「哪些错误值得重试」，供 retry.py 判断。

错误码语义（与后续章节对应）：
    INVALID_REQUEST      入参非法（400）——如缺字段、类型错
    MODEL_NOT_FOUND      模型名未在路由表中（404）
    PROVIDER_ERROR       上游返回 5xx / 网关无法连接上游（502/504，可重试）
    TIMEOUT              上游超时（504，可重试）
    RATE_LIMITED         本网关对该模型限流（429，不重试，由调用方退避）
    PROMPT_NOT_FOUND     模板/版本不存在或渲染失败（404/400）
    STRUCTURED_OUTPUT    结构化输出不合规（422）
"""
# 注意：这里【不】在模块顶层 import fastapi。原因有二：
#   1. 解耦——错误码/异常属于「领域核心」，不应依赖 Web 框架；
#   2. 可测性——没有 fastapi 的环境也能导入本模块、测试路由/重试等核心逻辑。
# fastapi 只在真正构造 HTTP 响应时才需要，因此改为函数内惰性导入。

# —— 错误码常量（集中管理，避免散落字符串）——
ERR_INVALID_REQUEST = "INVALID_REQUEST"
ERR_MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
ERR_PROVIDER_ERROR = "PROVIDER_ERROR"
ERR_TIMEOUT = "TIMEOUT"
ERR_RATE_LIMITED = "RATE_LIMITED"
ERR_PROMPT_NOT_FOUND = "PROMPT_NOT_FOUND"
ERR_STRUCTURED_OUTPUT = "STRUCTURED_OUTPUT_ERROR"

# 可重试错误集合：只有落在这个集合里、且未达最大次数的错误才会被 retry.py 重试。
# 注意 RATE_LIMITED 不在此列——限流是网关对调用方的约束，应由调用方自行退避，
# 网关若重试只会加剧拥塞。
RETRYABLE = {ERR_PROVIDER_ERROR, ERR_TIMEOUT}


class GatewayError(Exception):
    """网关统一异常。

    属性：
        code        错误码（取上面的 ERR_* 常量）
        message     给人看的错误描述
        http_status 对应的 HTTP 状态码
        details     可选的附加信息（如重试等待秒数、校验错误明细）
        attempts    重试模块会写入实际尝试次数（默认 None）
    """

    def __init__(self, code: str, message: str, http_status: int = 400, details=None):
        self.code = code
        self.message = message
        self.http_status = http_status
        self.details = details
        self.attempts = None  # 由 retry.with_retries 填充
        super().__init__(message)


def error_response(e: GatewayError) -> "JSONResponse":
    """把 GatewayError 转成统一的 JSON 错误响应。

    返回体格式固定为：
        {"error": {"code": "<错误码>", "message": "<描述>", "details": <附加信息>}}
    这样无论底层是哪家供应商出错，调用方拿到的结构都一致。
    """
    # 惰性导入：仅在需要构造 HTTP 响应时才引入 fastapi，避免模块级强耦合。
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=e.http_status,
        content={"error": {"code": e.code, "message": e.message, "details": e.details}},
    )
