"""运行时配置层（Settings）。

配置来源：代码默认值 + 环境变量覆盖。这样本地开发（mock 模式）和真实部署
（填 Key）可以通过环境变量无缝切换，而不用改代码。

本作业的关键配置维度：
- use_mock：是否启用内置 Mock 适配器。默认开启，使作业在「无任何 API Key」
  的情况下也能跑通并验证全部功能；置为 0/False 后改用真实 provider。
- model_routing：模型名 → 适配器名的路由表。这是「动态路由」的数据源，
  也是本次作业「两种不同协议」的落脚点：两个模型分别指向 openai / anthropic 适配器。
- 限流/重试参数：容量、时间窗、最大重试次数、退避基数，都可调。
"""
import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """网关运行配置（全部字段均有默认值，可用环境变量覆盖）。"""

    # —— 模式开关 ——
    # True：使用内置 Mock 适配器（模拟两种协议），无需任何 Key。
    # 设置为 "0"/"false"/"False" 可关闭，转而连接真实 provider。
    use_mock: bool = True

    # —— 真实 provider 凭证（use_mock=False 时生效）——
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # —— 动态路由表：模型名 → 适配器名 ——
    # 作业要求两个模型走两种不同协议，因此这里分别指向 deepseek / anthropic 适配器：
    #   deepseek-v4-pro  → deepseek  （OpenAI 兼容的 Chat Completions 协议）
    #   deepseek-v4-flash → anthropic（Anthropic Messages API 协议）
    # 适配器名在 router.py 中注册，路由解析时据此找到具体适配器实例。
    # 若你持有 OpenAI Key，可把 deepseek-v4-pro 改回 "openai"（同时提供 OPENAI_API_KEY）。
    model_routing: dict = field(default_factory=lambda: {
        "deepseek-v4-pro": "deepseek",   # → DeepSeek Chat Completions 协议适配器
        "deepseek-v4-flash": "anthropic", # → Anthropic Messages API 协议适配器
    })

    # —— 按模型独立限流（滑动窗口算法，见 rate_limiter.py）——
    rate_limit_capacity: int = 1000   # 每个模型在时间窗内允许的最大请求数
    rate_limit_window: float = 60.0   # 时间窗长度（秒）

    # —— 指数退避重试（见 retry.py）——
    max_retries: int = 3              # 最多重试次数（不含首次）
    retry_base_delay: float = 0.2     # 退避基数（秒），第 n 次等待 ≈ base * 2^n

    # —— 提示词模板存放目录（见 prompts.py）——
    templates_dir: str = "templates"


def get_settings() -> Settings:
    """从环境变量构建 Settings。

    环境变量名（覆盖对应字段）：
        USE_MOCK, OPENAI_API_KEY, OPENAI_BASE_URL,
        ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL,
        DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
        RATE_LIMIT_CAPACITY, RATE_LIMIT_WINDOW,
        MAX_RETRIES, RETRY_BASE_DELAY, TEMPLATES_DIR
    """
    return Settings(
        # 仅当显式置为 "0"/"false"/"False" 时才关闭 mock。
        use_mock=os.environ.get("USE_MOCK", "1") not in ("0", "false", "False"),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        anthropic_base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        rate_limit_capacity=int(os.environ.get("RATE_LIMIT_CAPACITY", "1000")),
        rate_limit_window=float(os.environ.get("RATE_LIMIT_WINDOW", "60")),
        max_retries=int(os.environ.get("MAX_RETRIES", "3")),
        retry_base_delay=float(os.environ.get("RETRY_BASE_DELAY", "0.2")),
        templates_dir=os.environ.get("TEMPLATES_DIR", "templates"),
    )
