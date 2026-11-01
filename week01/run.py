"""启动入口。

等价于：`uvicorn app.transport.server:app --host 0.0.0.0 --port 8000`。

用法：
    python3 run.py                      # 默认 8000 端口，USE_MOCK 由环境变量决定
    USE_MOCK=0 OPENAI_API_KEY=sk-xxx python3 run.py   # 切真实 provider

注意：本文件依赖 fastapi + uvicorn，需先 `pip install -r requirements.txt`。
"""
import uvicorn


def main() -> None:
    # "app.transport.server:app" 让 uvicorn 以字符串导入，避免在当前进程提前 import app.transport.server
    # （从而把 fastapi 的加载推迟到真正起服务时，核心逻辑仍可在无 fastapi 环境测试）。
    uvicorn.run("app.transport.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
