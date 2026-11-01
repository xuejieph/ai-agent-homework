"""适配器抽象基类（Adapter Pattern 的核心）。

设计说明：
    网关调用的永远是 `adapter.chat(req, stream)`，它不关心底下是 OpenAI 还是
    Anthropic。具体适配器负责两件翻译工作：
        1. 入站翻译：把统一 ChatRequest 变成「该协议专用的请求体」
           （例如 OpenAI 用 input 字段、Anthropic 用 messages 字段）。
        2. 出站翻译：把「该协议的响应」归一化成统一 ChatResponse / StreamChunk
           （例如把 Anthropic 的 content_block_delta 翻成 OpenAI chunk 形状）。

    这样，两种协议在请求体结构、返回格式、鉴权方式上的差异，全部被封死在
    各自的适配器里，对外完全不可见。

关于返回类型：
    非流式返回 ChatResponse；流式返回一个「异步迭代器」，逐个产出 StreamChunk。
    用 Union 表达这种二选一；调用方（网关）会根据 stream 参数决定如何消费。
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Union

from app.core.models import ChatRequest, ChatResponse, StreamChunk


class BaseAdapter(ABC):
    """所有协议适配器的抽象基类。

    子类必须：
        - 设置类属性 protocol（标识封装的是哪种协议，仅用于观测/调试）
        - 实现 chat() 方法完成实际调用与协议翻译
    """

    # protocol 仅作标识，例如 "openai-responses" / "anthropic-messages"。
    protocol: str = "base"

    def __init__(self, name: str):
        # name 是该适配器在注册表中的名字，如 "openai" / "anthropic"，
        # 路由表 model -> name 正是用这个名字定位适配器实例。
        self.name = name

    @abstractmethod
    async def chat(self, req: ChatRequest, stream: bool) -> Union[ChatResponse, AsyncIterator[StreamChunk]]:
        """执行一次模型调用并完成协议翻译。

        参数：
            req    统一请求（其中的 model 已由网关路由解析，可回显）
            stream 是否要求流式返回

        返回：
            stream=False → ChatResponse（完整响应）
            stream=True  → 异步迭代器，逐个产出 StreamChunk（调用方负责迭代）

        异常：
            上游/协议相关错误应抛 GatewayError（见 errors.py），由网关统一处理。
        """
        raise NotImplementedError
