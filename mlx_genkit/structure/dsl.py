from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .grammar import Grammar
from .adherence import JsonAdherence
from ..config import GenerationConfig


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        inner = text[3:-3]
        if inner.startswith("\n"):
            inner = inner[1:]
        first_newline = inner.find("\n")
        if first_newline != -1:
            maybe_lang = inner[:first_newline].strip()
            if maybe_lang and all(ch.isalnum() or ch in "._+-" for ch in maybe_lang):
                inner = inner[first_newline + 1 :]
        return inner.strip()
    return text


@dataclass
class StructuredExample:
    input: str
    output: Dict[str, Any]

    def render(self) -> str:
        return "Input:\n" + _strip_fences(self.input) + "\nOutput:\n" + json.dumps(self.output, indent=2)


@dataclass
class StructuredSpec:
    schema: Dict[str, Any]
    fields: Sequence[str]
    examples: Sequence[Dict[str, Any]] = field(default_factory=list)
    name: Optional[str] = None
    grammar: Optional[Grammar] = None
    system_preamble: Optional[str] = None
    semantic_checks: Optional[Sequence[Any]] = None

    def render_task(self, task: str) -> str:
        lines: List[str] = []
        if self.name:
            lines.append(f"Specification: {self.name}")
        lines.append("Task:")
        lines.append(_strip_fences(task))
        lines.append("")
        lines.append("Return a JSON object with the following fields:")
        for f in self.fields:
            lines.append(f"- {f}")
        lines.append("")
        lines.append("The JSON must adhere to this schema:")
        lines.append(json.dumps(self.schema, indent=2))
        if self.examples:
            lines.append("")
            lines.append("Examples:")
            for idx, ex in enumerate(self.examples, 1):
                lines.append(f"Example {idx}:")
                lines.append(json.dumps(ex, indent=2))
        lines.append("")
        lines.append("Respond with JSON only. No code fences.")
        return "\n".join(lines)


def generate_structured(
    model: Any,
    tokenizer: Any,
    *,
    task: str,
    spec: StructuredSpec,
    config: Optional[GenerationConfig] = None,
    adherence: Optional[JsonAdherence] = None,
    **kwargs: Any,
):
    from ..api import generate

    cfg = config or GenerationConfig()
    adherence_obj = adherence or JsonAdherence(retries=0, strict_only_json=True)
    grammar = spec.grammar or Grammar.json_schema(spec.schema)
    prompt = spec.render_task(task)
    if spec.system_preamble:
        prompt = spec.system_preamble.strip() + "\n\n" + prompt
    result = generate(
        model,
        tokenizer,
        prompt,
        cfg,
        json_schema=spec.schema,
        grammar=grammar,
        adherence=adherence_obj,
        semantic_checks=spec.semantic_checks,
        **kwargs,
    )
    result.meta.setdefault("spec", spec.name or "anonymous_spec")
    return result
