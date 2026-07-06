"""prompts.py — versioned prompt registry.

Every prompt the agent uses lives in `finsight/prompts/*.yaml` — never inline in code.
A prompt file holds one named prompt with an append-only version history:

    name: grade
    role: fast                      # which router role runs it
    versions:
      - version: 1
        changelog: why this version exists
        system: optional system message
        template: |
          Question: $question ...   # string.Template ($var) — safe with JSON braces

Code asks for `get("grade")` (latest) or `get("grade", version=1)` (pinned). Rendering
uses `string.Template.substitute`, which raises on a missing variable instead of
silently emitting a broken prompt. Versions are append-only: edit = add a new version
with a changelog line, so every trace can name the exact prompt version that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    role: str
    template: str
    system: str | None = None
    changelog: str = ""

    @property
    def id(self) -> str:
        """Stable identifier for traces/logs, e.g. 'grade@1'."""
        return f"{self.name}@{self.version}"

    def render(self, **variables) -> str:
        try:
            return Template(self.template).substitute(**variables)
        except KeyError as e:
            raise KeyError(f"prompt {self.id}: missing variable {e.args[0]!r}") from None


@lru_cache
def _registry() -> dict[str, list[Prompt]]:
    reg: dict[str, list[Prompt]] = {}
    for path in sorted(PROMPTS_DIR.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        name, role = doc["name"], doc.get("role", "answer")
        versions = [
            Prompt(name=name, version=int(v["version"]), role=role,
                   template=v["template"], system=v.get("system"),
                   changelog=v.get("changelog", ""))
            for v in doc["versions"]
        ]
        reg[name] = sorted(versions, key=lambda p: p.version)
    return reg


def get(name: str, version: int | None = None) -> Prompt:
    """Latest version of a prompt, or a pinned one."""
    versions = _registry().get(name)
    if not versions:
        raise KeyError(f"unknown prompt {name!r}; have {sorted(_registry())}")
    if version is None:
        return versions[-1]
    for p in versions:
        if p.version == version:
            return p
    raise KeyError(f"prompt {name!r} has no version {version}; "
                   f"have {[p.version for p in versions]}")


def names() -> list[str]:
    return sorted(_registry())
