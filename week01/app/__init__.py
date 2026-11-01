"""app 包：LLM 统一模型调用服务的全部源码。

本包的整体定位：
- 对外只暴露「一套统一请求/响应契约」（见 models.py），调用方无需关心底层是
  OpenAI Responses API 还是 Anthropic Messages API。
- 协议差异被完全封死在 adapters/ 下的各个适配器里。
- gateway.py 负责把「路由 / 限流 / 重试 / 模板 / 观测」这些横切能力编排起来。

目录分工（后续步骤逐步填充）：
    models.py     统一数据契约（请求/响应/流式块/用量）
    errors.py     统一错误码与异常响应
    config.py     运行时配置（可经环境变量覆盖）
    utils.py      公共小工具（如 token 估算）
    adapters/     协议适配器（OpenAI / Anthropic / Mock）
    gateway.py    编排层
    server.py     FastAPI 入口与路由
"""
