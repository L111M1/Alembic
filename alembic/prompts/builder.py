from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import jinja2

from alembic.core.types import SeedSample

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _split_filter(value: str, sep: str = ",") -> list[str]:
    return [s.strip() for s in value.split(sep) if s.strip()]


class PromptBuilder:
    def __init__(self, lang: str = "en"):
        self._messages: list[dict] = []
        self._lang = lang
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=False,
        )
        env.globals["random"] = random
        env.filters["split"] = _split_filter
        self._env = env

    def system(self, text: str) -> PromptBuilder:
        self._messages.append({"role": "system", "content": text})
        return self

    def user(self, text: str) -> PromptBuilder:
        self._messages.append({"role": "user", "content": text})
        return self

    def assistant(self, text: str) -> PromptBuilder:
        self._messages.append({"role": "assistant", "content": text})
        return self

    def from_template(self, template_name: str, **variables) -> PromptBuilder:
        name = self._resolve_template(template_name)
        tmpl = self._env.get_template(name)
        rendered = tmpl.render(**variables)
        parsed = self._parse_chat_messages(rendered)
        self._messages.extend(parsed)
        return self

    def _resolve_template(self, name: str) -> str:
        lang_specific = name.replace(".j2", f"_{self._lang}.j2")
        if (TEMPLATES_DIR / lang_specific).exists():
            return lang_specific
        return name

    def build(self) -> list[dict]:
        return list(self._messages)

    def _parse_chat_messages(self, text: str) -> list[dict]:
        result = []
        lines = text.strip().split("\n")
        current_role = None
        current_lines = []
        for line in lines:
            if line.startswith("system:"):
                if current_role:
                    result.append({"role": current_role, "content": "\n".join(current_lines).strip()})
                current_role = "system"
                current_lines = [line[len("system:"):].strip()]
            elif line.startswith("user:"):
                if current_role:
                    result.append({"role": current_role, "content": "\n".join(current_lines).strip()})
                current_role = "user"
                current_lines = [line[len("user:"):].strip()]
            elif line.startswith("assistant:"):
                if current_role:
                    result.append({"role": current_role, "content": "\n".join(current_lines).strip()})
                current_role = "assistant"
                current_lines = [line[len("assistant:"):].strip()]
            else:
                if current_role:
                    current_lines.append(line)
        if current_role and current_lines:
            result.append({"role": current_role, "content": "\n".join(current_lines).strip()})
        return result


def format_examples(examples: list[SeedSample], builder: Optional[PromptBuilder] = None) -> PromptBuilder:
    b = builder or PromptBuilder()
    for ex in examples:
        b.user(ex.instruction)
        b.assistant(ex.output)
    return b


def load_seeds(seed_file: str, field_map: Optional[dict] = None) -> list[SeedSample]:
    seeds = []
    with open(seed_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if field_map:
                    item = {v: item.get(k, "") for k, v in field_map.items()}
                if "messages" in item and isinstance(item["messages"], list):
                    seed = SeedSample(messages=item["messages"])
                    sys_msg = next((m["content"] for m in item["messages"] if m.get("role") == "system"), "")
                    seed.system = sys_msg
                    seeds.append(seed)
                elif "instruction" in item and "output" in item:
                    seeds.append(SeedSample(
                        instruction=item["instruction"],
                        output=item["output"],
                        system=item.get("system", ""),
                    ))
                elif "instruction" in item and "response" in item:
                    seeds.append(SeedSample(
                        instruction=item["instruction"],
                        output=item["response"],
                        system=item.get("system", ""),
                    ))
            except (json.JSONDecodeError, KeyError):
                continue
    return seeds
