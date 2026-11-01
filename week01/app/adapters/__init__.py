"""adapters 包：协议适配器集合。

本包的每个模块封装「一种上游 API 协议」：
    base.py            抽象基类，定义所有适配器必须实现的接口
    mock.py            （Step 5）离线 Mock：模拟两种协议并归一化
    openai_adapter.py  （Step 5）真实 OpenAI Responses API
    anthropic_adapter.py（Step 5）真实 Anthropic Messages API

适配器存在的意义（适配器模式）：
    把「统一契约 ⇄ 具体协议」的翻译逻辑局限在一个类里。网关只依赖 BaseAdapter
    这个抽象接口，因此换协议/加供应商时，网关代码一行都不用改，只新增一个适配器。
"""
