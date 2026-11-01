"""统一数据契约层（Unified Contract）。

这一层是整个网关的「对外语言」。无论底层接的是 OpenAI Responses API 还是
Anthropic Messages API，所有进出网关的数据都必须先翻译成这里的模型：

- 入参：ChatRequest（调用方看到的统一请求）
- 出参（非流式）：ChatResponse
- 出参（流式）：StreamChunk（逐块，形状对齐 OpenAI Chat Completions chunk）
- 用量统计：Usage

为什么要单独抽一层契约？
    1. 调用方只学一套接口，换模型/换供应商不用改代码。
    2. 适配器只负责「把外部协议 ⇄ 本契约」互转，互不污染。
    3. 网关的路由/限流/重试/观测都基于这套稳定结构，与具体协议解耦。

注意：流式块之所以采用 OpenAI chunk 形状，是因为它是业界最通用、调用方
（如 openai SDK）最容易直接消费的形态；这只是「对外格式选择」，并不表示
我们只在用 OpenAI 协议——Anthropic 适配器同样会把自身事件归一化成这个形状。
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """单条对话消息。

    与 OpenAI / Anthropic 的消息语义保持一致：role 通常为 system / user / assistant，
    content 为纯文本。适配器在翻译时会把本结构映射到各自协议的消息体。
    """

    role: str = Field(..., description="角色：system / user / assistant")
    content: str = Field(..., description="消息正文（纯文本）")


class ChatRequest(BaseModel):
    """统一调用请求——调用方唯一需要关心的入参结构。

    设计要点：
    - model 字段是「动态路由」的唯一依据：网关据此查路由表，决定交给哪个适配器。
    - stream / response_format / prompt_* 是三大能力的开关，缺省即关闭。
    """

    # —— 基础字段 ——
    model: str = Field(..., description="模型名，例如 deepseek-v4-pro / deepseek-v4-flash；用于动态路由")
    messages: List[Message] = Field(..., description="对话历史，按时间顺序排列")

    # —— 能力开关：流式输出 ——
    stream: bool = Field(default=False, description="是否流式返回；true 时以 SSE 逐块输出")

    # —— 采样参数（透传给适配器，再由适配器翻成对应协议的参数）——
    temperature: float = Field(default=1.0, description="采样温度，范围 0~2")
    max_tokens: Optional[int] = Field(default=None, description="单次生成最大 token 数；None 表示由适配器/模型决定")

    # —— 能力开关：结构化输出 ——
    # 形如 {"type": "json_object"} 或 {"type": "json_schema", "json_schema": {...}}
    # 网关只负责把它透传给适配器；适配器再翻译成各自协议的约束方式。
    response_format: Optional[Dict[str, Any]] = Field(
        default=None, description="结构化输出约束；存在时要求返回可被 json.loads 解析的合法 JSON"
    )

    # —— 能力开关：提示词版本管理 ——
    # 三者配合使用：用 id+version 定位模板，再用 variables 做变量替换。
    prompt_template: Optional[str] = Field(default=None, description="提示词模板 id，引用已存储的模板")
    prompt_version: Optional[str] = Field(default=None, description="模板版本；为空时取最新版本")
    prompt_variables: Optional[Dict[str, Any]] = Field(default=None, description="模板变量键值对，用于 Jinja2 渲染")

    def has_structured_output(self) -> bool:
        """便捷判断：本次请求是否要求结构化输出。"""
        return self.response_format is not None

    def has_prompt_template(self) -> bool:
        """便捷判断：本次请求是否引用了提示词模板。"""
        return self.prompt_template is not None


class Usage(BaseModel):
    """一次调用的 token 用量。

    用于可观测性：网关会按模型分类汇总这些数字。
    """

    prompt_tokens: int = Field(default=0, description="输入（提示）token 数")
    completion_tokens: int = Field(default=0, description="输出（补全）token 数")
    total_tokens: int = Field(default=0, description="总 token 数 = prompt + completion")


class Choice(BaseModel):
    """非流式响应里的一个候选项（本作业只返回 1 个）。"""

    index: int = Field(default=0, description="候选序号，固定为 0")
    message: Optional[Message] = Field(default=None, description="生成的完整消息")
    finish_reason: Optional[str] = Field(default=None, description="结束原因，如 stop")


class ChatResponse(BaseModel):
    """非流式统一响应。

    形状对齐 OpenAI chat.completion，便于调用方用通用 SDK 直接消费。
    """

    id: str = Field(..., description="本次响应唯一 id")
    object: str = Field(default="chat.completion", description="对象类型标识")
    created: int = Field(..., description="创建时间戳（秒）")
    model: str = Field(..., description="实际使用的模型名（回显路由结果）")
    choices: List[Choice] = Field(..., description="候选列表")
    usage: Usage = Field(default_factory=Usage, description="token 用量")


class StreamChunk(BaseModel):
    """流式统一块。

    形状对齐 OpenAI chat.completion.chunk：每个块携带 choices，其中 delta 是
    增量内容。最后一个块 delta 为空、finish_reason 为 stop。流式结束时网关
    再补一个 SSE 的 `data: [DONE]` 信号（不在本模型内，由 server 层拼接）。
    """

    id: str = Field(..., description="本次流式响应唯一 id（同一流内一致）")
    object: str = Field(default="chat.completion.chunk", description="对象类型标识")
    created: int = Field(..., description="创建时间戳（秒）")
    model: str = Field(..., description="实际使用的模型名")
    # choices 用 dict 而不是强类型，是为了灵活承载 delta/finish_reason 的增量语义。
    choices: List[dict] = Field(
        default_factory=list,
        description='每个元素形如 {"index":0,"delta":{"content":"..."},"finish_reason":null}',
    )


class PromptCreate(BaseModel):
    """创建/发布提示词模板的请求体。"""

    id: str = Field(..., description="模板唯一 id")
    version: str = Field(..., description="版本号，例如 v1；同 id 重复提交即产生新版本")
    content: str = Field(..., description="模板正文，支持 Jinja2 变量占位符 {{ var }}")
    description: str = Field(default="", description="模板用途说明")
