from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dep
    yaml = None

from .api import GenerationConfig, generate
from .loader import auto_load
from .structure.adherence import JsonAdherence
from .structure.grammar import Grammar
from .structure.result import GenerateResult
from .structure.semantic import build_semantic_checks
from .structure.validators import ValidatorLike


def _resolve_path(path: str, *, base_dir: Optional[Path] = None) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and base_dir is not None:
        resolved = base_dir / resolved
    return resolved.resolve()


def _load_suite(path: str, *, require_object: bool = True) -> Any:
    content = _resolve_path(path).read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)
    if require_object and not isinstance(data, dict):
        raise ValueError("Suite file must decode to an object")
    return data


def _parse_validator_spec(spec: str) -> ValidatorLike:
    if spec in {"jsonschema", "json_schema"}:
        return "jsonschema"
    if spec.startswith("pydantic:"):
        module_path = spec.split(":", 1)[1]
        module_name, _, attr = module_path.rpartition(".")
        if not module_name:
            raise ValueError("pydantic validator requires module.Class path")
        module = __import__(module_name, fromlist=[attr])
        model_cls = getattr(module, attr)
        return ("pydantic", model_cls)
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
        module = __import__(module_name, fromlist=[attr])
        return getattr(module, attr)
    raise ValueError(f"Unsupported validator spec: {spec}")


@dataclass
class EvalCase:
    name: str
    prompt: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    json_schema: Optional[Dict[str, Any]] = None
    grammar: Optional[Grammar] = None
    adherence: JsonAdherence = field(default_factory=JsonAdherence)
    validators: Optional[List[ValidatorLike]] = None
    semantic_checks: Optional[Sequence[Any]] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalOutcome:
    case: EvalCase
    result: GenerateResult


class EvalSuite:
    def __init__(self, suite_path: str) -> None:
        self.suite_path = _resolve_path(suite_path)
        self.suite_dir = self.suite_path.parent
        self.raw = _load_suite(str(self.suite_path), require_object=True)
        self.model_id = self.raw.get("model")
        if not self.model_id:
            raise ValueError("Suite must specify a 'model'")
        base_cfg = self.raw.get("config", {})
        self.base_config = GenerationConfig(**base_cfg)
        self.cases = self._build_cases(self.raw.get("cases") or self.raw.get("items") or [])
        self.name = self.raw.get("name") or self.suite_path.stem

    def _build_cases(self, items: Sequence[Dict[str, Any]]) -> List[EvalCase]:
        cases: List[EvalCase] = []
        if not items:
            raise ValueError("Suite must define at least one case")
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError("Each case must be an object")
            name = item.get("name") or f"case_{idx+1}"
            prompt = item.get("prompt")
            if prompt is None and "messages" in item:
                prompt = item["messages"]
            if prompt is None:
                raise ValueError(f"Case '{name}' missing prompt or messages")
            json_schema = item.get("json_schema")
            grammar = None
            if item.get("grammar_gbnf"):
                grammar_path = _resolve_path(item["grammar_gbnf"], base_dir=self.suite_dir)
                grammar_text = grammar_path.read_text(encoding="utf-8")
                grammar = Grammar.gbnf(grammar_text)
            elif json_schema is not None:
                grammar = Grammar.json_schema(json_schema)
            adherence = JsonAdherence(
                retries=int(item.get("retries", self.raw.get("retries", 0))),
                strict_only_json=bool(item.get("strict_only_json", self.raw.get("strict_only_json", False))),
            )
            validators = None
            validator_specs = item.get("validators") or self.raw.get("validators")
            if validator_specs:
                validators = [_parse_validator_spec(v) for v in validator_specs]
            semantic_checks = None
            checks_def = item.get("semantic_checks") or self.raw.get("semantic_checks")
            if checks_def:
                spec_list = _load_semantic_defs(checks_def, base_dir=self.suite_dir)
                if spec_list:
                    semantic_checks = build_semantic_checks(spec_list)
            config_overrides = item.get("config") or {}
            cases.append(
                EvalCase(
                    name=name,
                    prompt=prompt,
                    metadata=item.get("meta", {}),
                    json_schema=json_schema,
                    grammar=grammar,
                    adherence=adherence,
                    validators=validators,
                    semantic_checks=semantic_checks,
                    config_overrides=config_overrides,
                )
            )
        return cases

    def run(self) -> List[EvalOutcome]:
        model, tokenizer, _local = auto_load(self.model_id)
        outcomes: List[EvalOutcome] = []
        for case in self.cases:
            cfg = copy.deepcopy(self.base_config)
            for key, value in case.config_overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
            result = generate(
                model,
                tokenizer,
                case.prompt,
                cfg,
                hooks=None,
                json_schema=case.json_schema,
                grammar=case.grammar,
                adherence=case.adherence,
                validators=case.validators,
                semantic_checks=case.semantic_checks,
            )
            result.meta.setdefault("case", case.name)
            outcomes.append(EvalOutcome(case=case, result=result))
        return outcomes

    @staticmethod
    def render_markdown(outcomes: List[EvalOutcome], suite_name: str) -> str:
        lines = [f"# Evaluation Report: {suite_name}", "", "| Case | Schema OK | Semantic OK | Attempts |", "| --- | --- | --- | --- |"]
        for outcome in outcomes:
            schema = "✅" if outcome.result.schema_ok else "❌"
            semantic = "✅" if outcome.result.semantic_ok in (None, True) else "❌"
            lines.append(
                f"| {outcome.case.name} | {schema} | {semantic} | {outcome.result.attempts} |"
            )
        passed = sum(1 for o in outcomes if o.result.ok)
        lines.append("")
        lines.append(f"**Passed:** {passed}/{len(outcomes)}")
        return "\n".join(lines)

    @staticmethod
    def to_dict(outcomes: List[EvalOutcome], suite_name: str) -> Dict[str, Any]:
        return {
            "suite": suite_name,
            "total": len(outcomes),
            "passed": sum(1 for o in outcomes if o.result.ok),
            "cases": [
                {
                    "name": o.case.name,
                    "schema_ok": o.result.schema_ok,
                    "semantic_ok": o.result.semantic_ok,
                    "attempts": o.result.attempts,
                    "violations": o.result.violations,
                    "meta": o.result.meta,
                }
                for o in outcomes
            ],
        }
def _load_semantic_defs(defs: Any, *, base_dir: Optional[Path] = None) -> Optional[List[Dict[str, Any]]]:
    if defs is None:
        return None
    if isinstance(defs, str):
        data = _load_suite(str(_resolve_path(defs, base_dir=base_dir)), require_object=False)
        if isinstance(data, dict):
            defs = data.get("checks") or data.get("semantic_checks")
        else:
            defs = data
    if isinstance(defs, list):
        return defs
    raise ValueError("Semantic checks must be a list or path to a list")
