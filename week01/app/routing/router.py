"""适配器注册表 + 动态路由解析。

这是「按 model 字段动态路由」的核心实现。

为什么需要注册表？
    网关收到请求时只知道 model 字符串（如 "deepseek-v4-pro"）。它需要从字符串
    找到具体适配器实例。注册表就是这张「模型名 → 适配器」的映射，外加一层
    「适配器名 → 适配器实例」的间接层，方便一个适配器服务多个模型。

路由流程：
    model ──(routing 表)──► adapter_name ──(adapters 表)──► BaseAdapter 实例
"""
from app.adapters.base import BaseAdapter
from app.core.errors import ERR_MODEL_NOT_FOUND, GatewayError


class AdapterRegistry:
    """管理适配器实例与模型路由表。"""

    def __init__(self):
        # adapter_name -> 适配器实例
        self.adapters: dict[str, BaseAdapter] = {}
        # model(模型名) -> adapter_name（路由表）
        self.routing: dict[str, str] = {}

    def register(self, name: str, adapter: BaseAdapter, models: list[str] | None = None) -> None:
        """注册一个适配器，并可把它绑定到一个或多个模型名。

        参数：
            name    适配器注册名（如 "openai"），用于 routing 表指向它
            adapter 适配器实例
            models  要绑定到该适配器的模型名列表；为空表示仅注册、暂不绑模型
        """
        self.adapters[name] = adapter
        for model in (models or []):
            self.routing[model] = name

    def resolve(self, model: str) -> BaseAdapter:
        """根据模型名动态解析出对应的适配器实例。

        这就是「动态路由」的落点：请求里的 model 字段在这里被翻译成具体适配器。
        找不到时抛 MODEL_NOT_FOUND（HTTP 404），因为对调用方而言就是「模型不存在」。
        """
        adapter_name = self.routing.get(model)
        if not adapter_name or adapter_name not in self.adapters:
            raise GatewayError(
                ERR_MODEL_NOT_FOUND,
                f"no adapter registered for model '{model}'",
                http_status=404,
            )
        return self.adapters[adapter_name]

    def list_models(self) -> list[str]:
        """返回当前已注册路由的所有模型名（用于 /v1/models 等接口）。"""
        return list(self.routing.keys())


def build_registry(settings) -> AdapterRegistry:
    """根据配置构建适配器注册表（按 settings.model_routing 数据驱动注册）。

    - use_mock=True：注册离线 Mock 适配器（MockOpenAI / MockAnthropic /
      MockDeepSeek），无需任何 Key 即可跑通全部功能点；额外注册 FaultyAdapter 供重试演示。
    - use_mock=False：注册真实 httpx 适配器（OpenAI / Anthropic / DeepSeek），
      需要对应 API Key 与联网。默认路由下 deepseek-v4-pro → DeepSeek（Chat Completions），
      deepseek-v4-flash → Anthropic（Messages API），两者协议不同。

    注册逻辑：先准备好「适配器名 → 实例」映射，再遍历 model_routing，
    把每个模型名绑定到其对应的适配器。这样路由表改模型名/加模型都不用动这里。
    """
    from app.adapters.mock import (
        MockOpenAIAdapter, MockAnthropicAdapter, MockDeepSeekAdapter, FaultyAdapter,
    )
    from app.adapters.openai_adapter import OpenAIAdapter
    from app.adapters.anthropic_adapter import AnthropicAdapter
    from app.adapters.deepseek_adapter import DeepSeekAdapter

    if settings.use_mock:
        adapters = {
            "openai": MockOpenAIAdapter("openai"),
            "anthropic": MockAnthropicAdapter("anthropic"),
            "deepseek": MockDeepSeekAdapter("deepseek"),
            "faulty": FaultyAdapter("faulty", fail_times=2),
        }
    else:
        adapters = {
            "openai": OpenAIAdapter("openai", settings.openai_api_key, settings.openai_base_url),
            "anthropic": AnthropicAdapter("anthropic", settings.anthropic_api_key, settings.anthropic_base_url),
            "deepseek": DeepSeekAdapter("deepseek", settings.deepseek_api_key, settings.deepseek_base_url),
        }

    reg = AdapterRegistry()
    for model, adapter_name in settings.model_routing.items():
        if adapter_name in adapters:
            reg.register(adapter_name, adapters[adapter_name], [model])
    return reg
