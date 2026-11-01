"""提示词版本管理（完整实现）。

能力（作业要求三大能力之一）：模板存储、变量替换、版本引用。

存储结构：
    每个模板一个 YAML 文件，放在 templates_dir 下，文件名 = <id>.yaml：
        id: welcome
        description: 欢迎语模板
        versions:
          - version: v1
            content: |
              你是 {{ name }}，请欢迎用户 {{ user }}。
    内存中以 self.templates[id] = {"description", "versions": [{version, content}, ...]}
    保存（versions 用列表而非字典，以保留「版本加入顺序」，从而支持「version=None 取最新」）。

关键设计：
    - render(id, version, variables)：
        * version=None → 取最后加入的版本（最新）。
        * 用 Jinja2 的 StrictUndefined：变量缺失会直接报错，避免「模板渲染出空变量」的静默 bug。
        * 渲染失败 / 模板或版本不存在 → 统一抛 ERR_PROMPT_NOT_FOUND（含 404/400）。
    - add_template(...) ：合并写回 YAML（同名版本覆盖，新版本追加），并刷新内存。
"""
import os
import threading

import jinja2
import yaml

from app.core.errors import ERR_PROMPT_NOT_FOUND, GatewayError


class PromptStore:
    def __init__(self, templates_dir: str):
        self.dir = templates_dir
        self.templates: dict = {}          # id -> {"description", "versions": [{version, content}]}
        self._lock = threading.Lock()
        self._load_all()

    # ——————————————————— 加载 ———————————————————
    def _load_all(self) -> None:
        if not os.path.isdir(self.dir):
            return
        for fname in os.listdir(self.dir):
            if fname.endswith((".yaml", ".yml")):
                self._load_file(os.path.join(self.dir, fname))

    def _load_file(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        # 解析成有序版本列表，便于「取最新」
        versions = [{"version": v["version"], "content": v["content"]} for v in data.get("versions", [])]
        with self._lock:
            self.templates[data["id"]] = {"description": data.get("description", ""), "versions": versions}

    # ——————————————————— 查询 ———————————————————
    def list_templates(self) -> list:
        """列出所有模板及其版本（GET /v1/prompts）。"""
        with self._lock:
            return [
                {"id": tid, "description": t["description"],
                 "versions": [v["version"] for v in t["versions"]]}
                for tid, t in self.templates.items()
            ]

    def get_template(self, template_id: str) -> dict | None:
        """获取单个模板详情（GET /v1/prompts/{id}）；不存在返回 None。"""
        with self._lock:
            t = self.templates.get(template_id)
            if not t:
                return None
            return {"id": template_id, "description": t["description"], "versions": list(t["versions"])}

    # ——————————————————— 渲染（变量替换 + 版本引用） ———————————————————
    def render(self, template_id: str, version: str | None = None, variables: dict | None = None):
        """渲染模板，返回 (实际使用的版本号, 渲染后的文本)。

        异常：
            ERR_PROMPT_NOT_FOUND 在以下情况抛出：
                - 模板不存在 / 没有版本
                - 指定 version 找不到
                - 变量缺失或模板语法错误（Jinja2 渲染失败）
        """
        with self._lock:
            t = self.templates.get(template_id)
            if not t:
                raise GatewayError(ERR_PROMPT_NOT_FOUND, f"template '{template_id}' not found", 404)
            versions = t["versions"]
            if not versions:
                raise GatewayError(ERR_PROMPT_NOT_FOUND, f"template '{template_id}' has no versions", 404)
            if version is None:
                chosen = versions[-1]                      # 最新版本
            else:
                chosen = next((v for v in versions if v["version"] == version), None)
                if chosen is None:
                    raise GatewayError(ERR_PROMPT_NOT_FOUND,
                                      f"version '{version}' not found for '{template_id}'", 404)

        # 渲染放在锁外：Jinja2 渲染不依赖内部状态，且可能较慢
        try:
            env = jinja2.Environment(
                loader=jinja2.BaseLoader(),
                undefined=jinja2.StrictUndefined,   # 变量缺失即报错，防止静默空渲染
            )
            template = env.from_string(chosen["content"])
            rendered = template.render(**(variables or {}))
        except jinja2.TemplateError as e:
            raise GatewayError(ERR_PROMPT_NOT_FOUND, f"render failed: {e}", 400,
                              details={"template_id": template_id, "version": chosen["version"]})
        return chosen["version"], rendered

    # ——————————————————— 写入（持久化） ———————————————————
    def add_template(self, template_id: str, version: str, content: str, description: str = "") -> None:
        """新增 / 更新一个模板版本，并持久化到 YAML。

        同名版本覆盖，新版本追加；文件不存在则创建。写完后刷新内存。
        """
        os.makedirs(self.dir, exist_ok=True)
        path = os.path.join(self.dir, f"{template_id}.yaml")

        # 读取已有文件以合并版本
        existing = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                existing = yaml.safe_load(fh) or {}

        versions = [{"version": v["version"], "content": v["content"]} for v in existing.get("versions", [])]
        # 覆盖同名版本或追加新版本
        for i, v in enumerate(versions):
            if v["version"] == version:
                versions[i] = {"version": version, "content": content}
                break
        else:
            versions.append({"version": version, "content": content})

        data = {
            "id": template_id,
            "description": description or existing.get("description", ""),
            "versions": versions,
        }
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)

        # 刷新内存
        with self._lock:
            self.templates[template_id] = {"description": data["description"], "versions": versions}
