from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .grammar import Grammar
from .result import GenerateResult
from .semantic import SemanticCheck, run_semantic_checks
from .validators import ValidatorResult, resolve_validators


@dataclass
class JsonAdherence:
    retries: int = 0
    strict_only_json: bool = False
    retry_messages: Optional[Sequence[str]] = None
    prepend_system: bool = True
    auto_trim_fences: bool = True

    def instructions_for(self, attempt: int, violations: Sequence[Dict[str, Any]]) -> str:
        templates = self.retry_messages or DEFAULT_RETRY_MESSAGES
        idx = min(attempt, len(templates) - 1)
        base = templates[idx]
        if violations:
            details = "; ".join(v.get("message", "") for v in violations if v.get("message"))
            if details:
                base = f"{base} Fix issues: {details}."
        if self.strict_only_json and "JSON" not in base.upper():
            base = base + " Respond with JSON only."
        return base


DEFAULT_RETRY_MESSAGES = (
    "Please respond with valid JSON that matches the requested schema.",
    "Reminder: output must be strictly valid JSON. Do not include explanations.",
    "Critical: Emit JSON only. Repair prior validation errors and return valid JSON.",
)


_FENCE_PATTERN = re.compile(r"```(?P<lang>[^`\n]*)\n?(?P<body>[\s\S]*?)```", re.MULTILINE)


def _clone_prompt(prompt: Any) -> Any:
    if isinstance(prompt, str):
        return prompt
    return copy.deepcopy(prompt)


def _apply_instruction(prompt: Any, instruction: str, *, prepend_system: bool = True) -> Any:
    if isinstance(prompt, str):
        sep = "\n\n" if prompt.strip() else ""
        return f"{prompt}{sep}{instruction}"
    if isinstance(prompt, list):
        updated = copy.deepcopy(prompt)
        msg = {"role": "system" if prepend_system else "user", "content": instruction}
        updated.append(msg)
        return updated
    return prompt


@dataclass
class AttemptRecord:
    attempt: int
    violations: List[Dict[str, Any]] = field(default_factory=list)
    validator: Optional[str] = None


class StructuredGenerationEngine:
    def __init__(
        self,
        *,
        backend,
        prompt: Any,
        config,
        hooks: Optional[Any],
        json_schema: Optional[Dict[str, Any]],
        grammar: Optional[Grammar],
        validators: Optional[Sequence[Any]],
        semantic_checks: Optional[Sequence[SemanticCheck]],
        adherence: JsonAdherence,
        on_parse_fail: Optional[Callable[[str, Exception], Optional[str]]],
        on_semantic_fail: Optional[Callable[[Dict[str, Any], Sequence[Dict[str, Any]]], Optional[Dict[str, Any]]]],
        log_writer: Optional[Callable[[GenerateResult, Dict[str, Any]], None]],
        stream_observer: Optional[Any] = None,
    ) -> None:
        self.backend = backend
        self.base_prompt = _clone_prompt(prompt)
        self.config = config
        self.hooks = hooks
        self.json_schema = json_schema
        self.grammar = grammar
        self.validator_fns = resolve_validators(validators, json_schema=json_schema)
        self.semantic_checks = semantic_checks
        self.adherence = adherence
        self.on_parse_fail = on_parse_fail
        self.on_semantic_fail = on_semantic_fail
        self.log_writer = log_writer
        self.stream_observer = stream_observer
        self.schema_required = json_schema is not None
        self.backend_name = getattr(backend, "name", "unknown")
        self.grammar_supported = True
        if grammar and not grammar.supports_backend(self.backend_name):
            if grammar.kind != "json_schema":
                raise NotImplementedError(
                    f"Grammar kind '{grammar.kind}' is not supported by backend '{self.backend_name}'."
                )
            self.grammar_supported = False

    # ------------------------------------------------------------------
    def run(self) -> GenerateResult:
        prompt = _clone_prompt(self.base_prompt)
        max_attempts = (self.adherence.retries or 0) + 1
        last_result: Optional[GenerateResult] = None

        for attempt in range(max_attempts):
            record = AttemptRecord(attempt=attempt)
            if self.stream_observer is not None:
                self.stream_observer.reset()
            result = self._generate_once(prompt, attempt)
            if self._process_attempt(result, record):
                self._log(result)
                return result

            last_result = result
            if attempt + 1 >= max_attempts:
                break
            prompt = self._augment_prompt(prompt, record, attempt)

        if last_result is None:
            raise RuntimeError("No generation attempts executed")
        self._log(last_result)
        return last_result

    def _generate_once(self, prompt: Any, attempt: int) -> GenerateResult:
        result = self.backend.generate(
            prompt,
            self.config,
            hooks=self.hooks,
            stream_observer=self.stream_observer,
        )
        result.attempts = attempt + 1
        result.meta.setdefault("schema_required", self.schema_required)
        if self.grammar is not None:
            result.meta.setdefault("grammar_kind", self.grammar.kind)
            result.meta.setdefault("grammar_supported", self.grammar_supported)
        if self.stream_observer is not None:
            result.meta.setdefault("stream_tokens_emitted", self.stream_observer.emitted)
            if self.stream_observer.invalid_triggered:
                result.meta.setdefault("stream_invalid_path", True)
        return result

    def _process_attempt(self, result: GenerateResult, record: AttemptRecord) -> bool:
        if not self._parse_result(result, record):
            return False
        if not self._validate_schema(result, record):
            return False
        if not self._apply_semantic_checks(result, record):
            return False
        result.violations = []
        return True

    def _parse_result(self, result: GenerateResult, record: AttemptRecord) -> bool:
        ok, obj, only_json, normalized_text, error = self._attempt_parse_json(result.text)
        result.only_json = only_json
        if ok:
            result.json = obj
            if normalized_text and normalized_text != result.text:
                result.text = normalized_text
            return True

        record.violations.append({"type": "parse_error", "message": str(error) if error else "parse failure"})
        result.schema_ok = False
        result.violations = record.violations
        if self.schema_required:
            result.meta["schema_required"] = True

        if self.on_parse_fail and error is not None:
            repaired = self.on_parse_fail(result.text, error)  # type: ignore[arg-type]
            if repaired:
                ok, obj, only_json, normalized_text, secondary_error = self._attempt_parse_json(repaired)
                if ok:
                    record.violations.clear()
                    result.text = normalized_text or repaired
                    result.only_json = only_json
                    result.json = obj
                    return True
                # Update error context for subsequent retries
                if secondary_error is not None:
                    record.violations[-1] = {
                        "type": "parse_error",
                        "message": str(secondary_error),
                    }
        return False

    def _validate_schema(self, result: GenerateResult, record: AttemptRecord) -> bool:
        if result.json is None:
            return False

        schema_ok, validator_name, validator_errors = self._run_validators(result.json)
        if schema_ok:
            result.schema_ok = True
            record.validator = validator_name
            if validator_name:
                result.meta.setdefault("validator", validator_name)
            result.meta.setdefault("schema_required", self.schema_required)
            return True

        record.violations.append(
            {
                "type": "schema_validation",
                "validator": validator_name,
                "errors": validator_errors,
                "message": "; ".join(validator_errors) if validator_errors else "Schema validation failed",
            }
        )
        result.schema_ok = False
        result.violations = record.violations
        return False

    def _apply_semantic_checks(self, result: GenerateResult, record: AttemptRecord) -> bool:
        if not self.semantic_checks:
            result.semantic_ok = True
            return True

        if result.json is None:
            return False

        violations = run_semantic_checks(result.json, self.semantic_checks)
        if not violations:
            result.semantic_ok = True
            return True

        result.semantic_ok = False
        entries = [
            {
                "type": "semantic",
                "check": v.check,
                "message": v.message,
            }
            for v in violations
        ]
        record.violations.extend(entries)
        result.violations = record.violations

        if self.on_semantic_fail:
            repaired = self.on_semantic_fail(result.json, entries)  # type: ignore[arg-type]
            if repaired is not None:
                follow_up = run_semantic_checks(repaired, self.semantic_checks)
                if not follow_up:
                    record.violations.clear()
                    result.json = repaired
                    result.semantic_ok = True
                    result.violations = []
                    return True
                record.violations.extend(
                    {
                        "type": "semantic",
                        "check": v.check,
                        "message": v.message,
                    }
                    for v in follow_up
                )
        return False

    def _augment_prompt(self, prompt: Any, record: AttemptRecord, attempt: int) -> Any:
        instruction = self.adherence.instructions_for(attempt, record.violations)
        return _apply_instruction(prompt, instruction, prepend_system=self.adherence.prepend_system)

    def _attempt_parse_json(
        self, text: str
    ) -> Tuple[bool, Optional[Any], bool, Optional[str], Optional[Exception]]:
        stripped = text.strip()
        decoder = json.JSONDecoder()
        variants = self._collect_parse_variants(text, decoder)
        if stripped and stripped not in variants:
            variants.insert(0, stripped)
        last_error: Optional[JSONDecodeError] = None

        for variant in variants:
            start_candidates = [variant.find("{"), variant.find("[")]
            start_candidates = [i for i in start_candidates if i != -1]
            if not start_candidates:
                last_error = JSONDecodeError("No JSON object found", variant, 0)
                continue
            for start in sorted(start_candidates):
                try:
                    obj, end = decoder.raw_decode(variant[start:])
                    prefix = variant[:start].strip()
                    suffix = variant[start + end :].strip()
                    only_json = not prefix and not suffix
                    if self.adherence.strict_only_json and not only_json:
                        raise JSONDecodeError("Non-JSON content present", variant, start)
                    normalized_candidate = variant[start : start + end].strip()
                    if only_json:
                        normalized = normalized_candidate
                    elif variant == stripped:
                        normalized = variant
                    else:
                        normalized = text
                    return True, obj, only_json, normalized, None
                except JSONDecodeError as err:
                    last_error = err

        if last_error is None:
            last_error = JSONDecodeError("Unable to decode JSON", stripped, 0)
        return False, None, False, None, last_error

    def _collect_parse_variants(self, text: str, decoder: json.JSONDecoder) -> List[str]:
        variants: List[str] = []
        seen: set[str] = set()

        def add_candidate(candidate: str) -> None:
            trimmed = candidate.strip()
            if not trimmed or trimmed in seen:
                return
            seen.add(trimmed)
            variants.append(trimmed)

        stripped = text.strip()
        if stripped:
            add_candidate(stripped)

        if not self.adherence.auto_trim_fences:
            return variants

        for match in _FENCE_PATTERN.finditer(text):
            body = match.group("body")
            add_candidate(body)

        if self.adherence.strict_only_json:
            return variants

        for idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                _, end = decoder.raw_decode(text[idx:])
            except JSONDecodeError:
                continue
            snippet = text[idx : idx + end]
            add_candidate(snippet)

        return variants

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return text
        closing = stripped.rfind("```")
        if closing <= 0:
            return text
        inner = stripped[3:closing].strip()
        lines = inner.splitlines()
        if len(lines) >= 2:
            first = lines[0].strip()
            if first:
                lines = lines[1:]
                inner = "\n".join(lines)
        inner = inner.strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].lstrip()
        return inner or text

    def _run_validators(self, obj: Any) -> Tuple[bool, Optional[str], List[str]]:
        if not self.validator_fns:
            return True, None, []
        errors: List[str] = []
        passed_names: List[str] = []
        failed_names: List[str] = []
        for validator in self.validator_fns:
            name = getattr(validator, "__validator_name__", getattr(validator, "__name__", "validator"))
            outcome: ValidatorResult = validator(obj)
            resolved_name = outcome.validator_name or name
            if outcome.ok:
                passed_names.append(resolved_name)
                continue
            failed_names.append(resolved_name)
            if outcome.errors:
                errors.extend(outcome.errors)
            else:
                errors.append(f"{resolved_name} failed validation")
        if failed_names:
            return False, ", ".join(failed_names), errors
        combined_name = ", ".join(passed_names) if passed_names else None
        return True, combined_name, []

    def _parse_json(self, text: str) -> Tuple[bool, Optional[Any], Optional[Exception], bool]:
        stripped = text.strip()
        decoder = json.JSONDecoder()
        start_candidates = [stripped.find("{"), stripped.find("[")]
        start_candidates = [i for i in start_candidates if i != -1]
        if not start_candidates:
            err = JSONDecodeError("No JSON object found", stripped, 0)
            return False, None, err, False
        last_error: Optional[JSONDecodeError] = None
        for start in sorted(start_candidates):
            try:
                obj, end = decoder.raw_decode(stripped[start:])
                prefix = stripped[:start].strip()
                suffix = stripped[start + end :].strip()
                only_json = not prefix and not suffix
                if self.adherence.strict_only_json and not only_json:
                    raise JSONDecodeError("Non-JSON content present", stripped, start)
                return True, obj, None, only_json
            except JSONDecodeError as err:  # pragma: no cover - dependent on text
                last_error = err
                continue
        if last_error is None:
            last_error = JSONDecodeError("Unable to decode JSON", stripped, 0)
        return False, None, last_error, False

    def _log(self, result: GenerateResult) -> None:
        if not self.log_writer:
            return
        context = {
            "schema": bool(self.json_schema),
            "backend": getattr(self.backend, "name", "unknown"),
        }
        self.log_writer(result, context)
