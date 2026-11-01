"""Step 7 验证：提示词版本管理（模板存储 / 变量替换 / 版本引用）。

验证点：
    1. add_template + render：变量被正确替换。
    2. version=None 取最新版本（v2）。
    3. 指定 version 可取历史版本（v1）。
    4. 变量缺失 → 抛 ERR_PROMPT_NOT_FOUND（StrictUndefined 生效）。
    5. 模板/版本不存在 → 抛 ERR_PROMPT_NOT_FOUND。
    6. 网关 _apply_template：请求引用模板后，system 消息被注入渲染结果。
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.prompts.store import PromptStore
from app.core.config import Settings
from app.routing.router import build_registry
from app.gateway import Gateway
from app.core.models import ChatRequest, Message
from app.core.errors import ERR_PROMPT_NOT_FOUND, GatewayError


def _store():
    """用临时目录构建一个 PromptStore，避免依赖 cwd 下的 templates/。"""
    d = tempfile.mkdtemp()
    s = PromptStore(d)
    s.add_template("greet", "v1", "你是{{ name }}，欢迎{{ user }}。", "greet v1")
    s.add_template("greet", "v2", "Hi {{ user }}, I'm {{ name }}。", "greet v2")
    return s


def test_render_substitutes_variables():
    s = _store()
    ver, text = s.render("greet", "v1", {"name": "A", "user": "B"})
    assert ver == "v1"
    assert "A" in text and "B" in text


def test_render_defaults_to_latest_version():
    s = _store()
    ver, text = s.render("greet", None, {"name": "A", "user": "B"})  # version=None → 最新 = v2
    assert ver == "v2"
    assert "Hi B" in text


def test_render_specific_version():
    s = _store()
    ver, _ = s.render("greet", "v1", {"name": "A", "user": "B"})
    assert ver == "v1"


def test_render_missing_variable_raises():
    s = _store()
    try:
        s.render("greet", "v1", {"name": "A"})  # 缺 user
        assert False, "变量缺失应当抛错"
    except GatewayError as e:
        assert e.code == ERR_PROMPT_NOT_FOUND


def test_render_unknown_template_raises():
    s = _store()
    try:
        s.render("nope")
        assert False
    except GatewayError as e:
        assert e.code == ERR_PROMPT_NOT_FOUND and e.http_status == 404


def test_gateway_applies_template():
    store = _store()
    settings = Settings(use_mock=True, model_routing={"deepseek-v4-pro": "openai"})
    gw = Gateway(settings, registry=build_registry(settings), prompts=store)

    req = ChatRequest(
        model="deepseek-v4-pro", messages=[Message(role="user", content="hi")],
        prompt_template="greet", prompt_version="v1", prompt_variables={"name": "A", "user": "B"},
    )
    gw._apply_template(req)  # 直接验证模板注入逻辑
    assert req.messages[0].role == "system"
    assert "A" in req.messages[0].content and "B" in req.messages[0].content


if __name__ == "__main__":
    test_render_substitutes_variables()
    test_render_defaults_to_latest_version()
    test_render_specific_version()
    test_render_missing_variable_raises()
    test_render_unknown_template_raises()
    test_gateway_applies_template()
    print("PROMPTS_OK: 提示词版本管理（存储/变量替换/版本引用/网关注入）通过")
