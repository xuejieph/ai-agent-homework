# LLM 统一模型调用服务（第一周作业）

一套**统一抽象层 + 适配器模式**的模型网关：两个模型走**两种不同 API 协议**
（OpenAI Responses API / Anthropic Messages API），对外只暴露**一套统一接口**，
协议差异全部封死在适配器内部。

> 默认 `USE_MOCK=1`：**无需任何 API Key**，即可在离线环境跑通并验证全部功能点。
> 配置 Key 后（`USE_MOCK=0`）可切换为真实 provider 调用。

---

## 一、架构总览

```
客户端
  │  POST /v1/chat/completions  (统一请求: model / messages / stream / response_format / prompt_*)
  ▼
app/server.py (FastAPI)
  ├─ 异常处理器：GatewayError → 统一错误码 JSON
  ├─ 路由：/v1/chat/completions, /v1/models, /v1/prompts*, /v1/metrics, /health
  ▼
app/gateway.py (Gateway 编排)
  ├─ ① 模板解析  _apply_template   （prompt_template/version/variables → 注入 system）
  ├─ ② 动态路由  registry.resolve   （按 model 找到具体适配器）
  ├─ ③ 限流      _rate_limit       （按模型独立滑动窗口，超限 429）
  ├─ ④ 重试      with_retries      （指数退避，仅可重试错误，最多 3 次）
  └─ ⑤ 观测      metrics.record    （Token 分类 / 延迟 / 首 Token 延迟 / 重试数）
  ▼
app/adapters/ (BaseAdapter 接口)
  ├─ openai_adapter.py     → OpenAI Responses API 协议
  ├─ anthropic_adapter.py  → Anthropic Messages API 协议
  └─ mock.py (MockOpenAI / MockAnthropic / Faulty) → 离线模拟并归一化
```

**两个模型与协议映射**（见 `app/config.py` 的 `model_routing`）：

| 模型名 | 适配器 | 协议 |
|---|---|---|
| `deepseek-v4-pro` | `openai` | OpenAI Responses API |
| `deepseek-v4-flash` | `anthropic` | Anthropic Messages API |

---

## 二、目录结构

```
homework/week01/
├── app/
│   ├── gateway.py         # 编排层（根）：模板/路由/限流/重试/观测，聚合下方所有子包
│   ├── core/              # 领域核心：零框架依赖（设计主线要求的边界）
│   │   ├── models.py      # 统一数据契约（请求/响应/流式块/用量）
│   │   ├── errors.py      # 统一错误码 + 可重试集合（fastapi 惰性导入）
│   │   ├── config.py      # Settings（环境变量覆盖）
│   │   └── utils.py       # token 估算等纯工具
│   ├── capabilities/      # 横切能力：独立、可单测、被 gateway 调用
│   │   ├── rate_limiter.py    # 按模型独立滑动窗口限流
│   │   ├── retry.py           # 指数退避 + 抖动重试
│   │   └── observability.py   # 指标存储（分类 Token / 延迟 / TTFT / 重试）
│   ├── routing/           # 动态路由
│   │   └── router.py      # AdapterRegistry + build_registry
│   ├── prompts/           # 提示词版本管理
│   │   └── store.py       # Jinja2 渲染 + YAML 持久化
│   ├── transport/         # HTTP / 流式传输（fastapi 只在这里出现）
│   │   ├── server.py      # FastAPI 入口（SSE 流式、异常处理器、各端点）
│   │   └── streaming.py   # SSE 文本格式化（纯函数，由 StreamingResponse 推送）
│   └── adapters/          # 适配器（已存在）
│       ├── base.py        # BaseAdapter 抽象接口
│       ├── openai_adapter.py    # OpenAI Responses API（真实，可选启用）
│       ├── anthropic_adapter.py # Anthropic Messages API（真实）
│       ├── deepseek_adapter.py  # DeepSeek Chat Completions（真实，默认 pro 走它）
│       └── mock.py        # 离线三协议 Mock + Faulty（演示重试）
├── templates/
│   └── welcome.yaml      # 示例提示词模板（v1/v2 两版本）
├── tests/
│   ├── test_routing.py    # Step 2 路由
│   ├── test_core.py       # Step 3 限流/重试/可观测
│   ├── test_orchestration.py
│   ├── test_adapters.py   # Step 5 双协议
│   ├── test_streaming.py  # Step 6 SSE 格式
│   ├── test_prompts.py    # Step 7 模板
│   ├── test_http.py       # Step 8 HTTP 层端到端（六大功能点）
│   └── test_full.py       # Step 8 网关层统一验证（六大功能点证据）
├── PLAN.md                # 分步实施计划与学习笔记
├── requirements.txt
└── run.py                 # 启动入口（uvicorn app.transport.server:app）
```

> 依赖方向单向：`transport → gateway → {routing, capabilities, prompts, adapters} → core`，
> `core` 与 `capabilities` 不依赖 fastapi，可独立导入与测试。

---

## 三、启动指南

### 1. 安装依赖

> **路径说明**：仓库内 `week01/` 直接在仓库根目录，请先 `cd week01`（或 `cd <你的仓库根>/week01`）再继续。
> **Windows 用户**：用 `python` 代替 `python3`；若 `pip install` 因 `requirements.txt` 的中文注释报
> `UnicodeDecodeError: 'gbk' codec can't decode`，请先执行 `$env:PYTHONUTF8='1'` 以 UTF-8 模式安装。

```bash
cd week01
pip3 install -r requirements.txt
```

> 本作业核心逻辑（路由/限流/重试/观测/适配器/模板）**不依赖 fastapi/pytest**，
> 在无网环境也能 `python3 tests/*.py` 直接看结果（见第四节）。

### 2. 启动服务（默认 mock 模式）

```bash
python3 run.py
# 或： uvicorn app.transport.server:app --host 0.0.0.0 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

### 3. 切换真实 provider（USE_MOCK=0）

默认路由下，`deepseek-v4-pro` 走 **DeepSeek**（OpenAI 兼容的 Chat Completions），
`deepseek-v4-flash` 走 **Anthropic**（Messages API）——两者是不同的协议。

```bash
USE_MOCK=0 \
DEEPSEEK_API_KEY=sk-你的deepseek-key \
ANTHROPIC_API_KEY=sk-ant-xxx \
python3 run.py
```

> **Windows (PowerShell) 用户**：上面的 `KEY=val \` 续行是 bash 语法，PowerShell 不识别（会报
> `无法将“USE_MOCK=0”项识别为 cmdlet`）。请改用 `$env:` 设置环境变量，并用 `python` 代替 `python3`：
>
> ```powershell
> $env:USE_MOCK='0'
> $env:DEEPSEEK_API_KEY='sk-你的deepseek-key'
> $env:ANTHROPIC_API_KEY='sk-ant-xxx'
> python run.py
> ```
```

> - DeepSeek 通过 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`（默认 `https://api.deepseek.com/v1`）配置。
>   注意 DeepSeek 是 **Chat Completions** 兼容，不能用 OpenAI Responses API 的 `/responses` 端点，
>   所以本作业单独写了 `deepseek_adapter.py`，而不是把 `OPENAI_BASE_URL` 指向 DeepSeek。
> - 若你持有 OpenAI Key，可把 `model_routing` 里 `deepseek-v4-pro` 改回 `"openai"`，并改用 `OPENAI_API_KEY`。
> - 两个 provider 的 base_url 都可用 `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` / `DEEPSEEK_BASE_URL` 改写。

---

## 四、curl 示例

> 以下示例在 **mock 模式** 下即可运行（不需要 Key）。

### ① 普通调用（非流式）

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

返回（mock 会把模型名回显到回答，便于区分协议）：

```json
{"id":"...","created":...,"model":"deepseek-v4-pro",
 "choices":[{"index":0,"message":{"role":"assistant","content":"[deepseek-v4-pro] 你好"},
            "finish_reason":"stop"}],
 "usage":{"prompt_tokens":..,"completion_tokens":..,"total_tokens":..}}
```

把 `"model"` 换成 `deepseek-v4-flash` 即走 Anthropic 协议适配器。

### ② 流式输出（SSE）

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-pro",
    "stream": true,
    "messages": [{"role": "user", "content": "abc"}]
  }'
```

输出逐块（`data: <StreamChunk JSON>\n\n`），结尾 `data: [DONE]\n\n`：

```
data: {"id":"...","created":...,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"content":"["},"finish_reason":null}]}

data: {"id":"...","choices":[{"index":0,"delta":{"content":"}"},"finish_reason":null}]}

data: {"id":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### ③ 结构化输出

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "x"}],
    "response_format": {"type": "json_object"}
  }'
```

返回内容可被 `json.loads` 解析（mock 返回 `{"model":..., "echo":"x", "ok":true}`）。

### ④ 提示词模板引用

先发布一个模板（或直接使用 `templates/welcome.yaml`）：

```bash
curl http://127.0.0.1:8000/v1/prompts \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "greet", "version": "v1",
    "content": "你好 {{ name }}，你是{{ role }}。",
    "description": "greeting"
  }'
```

调用时引用并传变量：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "hi"}],
    "prompt_template": "greet",
    "prompt_version": "v1",
    "prompt_variables": {"name": "Alice", "role": "助手"}
  }'
```

模板渲染后会作为 `system` 消息注入（变量已替换）。查看已发布模板：

```bash
curl http://127.0.0.1:8000/v1/prompts          # 列出
curl http://127.0.0.1:8000/v1/prompts/greet    # 详情
```

### ⑤ 可观测数据

```bash
curl http://127.0.0.1:8000/v1/metrics
```

返回按模型分类的 Token 统计、调用数、错误数、重试数、延迟列表、首 Token 延迟列表：

```json
{"total_calls":3,"total_tokens":..,"total_prompt_tokens":..,"total_completion_tokens":..,
 "total_retries":0,
 "by_model":{
   "deepseek-v4-pro":{"calls":2,"errors":0,"prompt_tokens":..,"completion_tokens":..,
     "total_tokens":..,"retries":0,"latency_ms":[..],"ttft_ms":[..]}
 }}
```

### ⑥ 限流（429）

把单模型容量调小后，超出即返回 429：

```bash
RATE_LIMIT_CAPACITY=1 python3 run.py
# 连续两次同模型调用，第二次：
# {"error":{"code":"RATE_LIMITED","message":"rate limit exceeded ...","details":{"retry_after_s":..}}}
```

### 错误码示例

网关统一错误响应格式：

```json
{"error": {"code": "MODEL_NOT_FOUND", "message": "model 'x' not found", "details": null}}
```

---

## 五、验证脚本（六大功能点证据）

### 离线逐文件运行（无需 fastapi/pytest）

```bash
python3 tests/test_routing.py     # 路由：两模型→不同协议，未知→404
python3 tests/test_core.py        # 限流/重试/可观测 单测
python3 tests/test_adapters.py    # 双协议适配器（mock）路由/非流式/流式/结构化
python3 tests/test_streaming.py   # SSE 文本格式
python3 tests/test_prompts.py     # 提示词版本管理
python3 tests/test_full.py        # 统一验证：六大功能点证据
```

### pytest 回归（需安装 pytest）

```bash
python3 -m pytest tests/ -v
```

`tests/test_full.py` 的输出即作业验收的**可验证证据**：

```
[①] 两模型调用正常：pro→openai-responses，flash→anthropic-messages
[②] 流式输出：拼接正确，末块 finish_reason=stop
[③] 结构化输出：pro/flash 返回内容均 json.loads 通过
[④] 提示词模板引用：变量替换后 system 内容='你好 Alice，你是助手。'
[⑤] 可观测数据：by_model 分类 Token / 延迟 / TTFT 记录
[⑥a] 重试恢复：faulty-model 重试后成功，metrics.retries=2
[⑥b] 限流：同模型第二次调用返回 429（RATE_LIMITED）
```

---

## 六、验收对照表

| 功能点 | 要求 | 证据（见 test_full.py） |
|---|---|---|
| 两个模型调用正常 | 两种协议均可工作 | ① pro/flash 分别走 openai-responses / anthropic-messages 并返回内容 |
| 流式输出 | stream=true 以 SSE 逐块返回 | ② 拼接正确 + 末块 `[DONE]` 前的 `finish_reason=stop` |
| 结构化输出 | 返回合法 JSON | ③ 两协议返回均可 `json.loads` |
| 提示词版本管理 | 模板存储/变量替换/版本引用 | ④ 变量替换后内容注入 system 并到达适配器 |
| 可观测性 | Token 分类 + 延迟 + 首 Token 延迟 | ⑤ `/v1/metrics` 分类统计、latency、ttft 列表 |
| 重试 + 限流 | 指数退避最多 3 次 + 按模型独立限流 429 | ⑥a 重试恢复并记录 retries；⑥b 超限返回 429 |


