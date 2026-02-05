from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union


ValidatorLike = Union[str, Callable[[Any], Any], Tuple[str, Any]]


@dataclass
class ValidatorResult:
    ok: bool
    errors: List[str]
    validator_name: str


def _coerce_error_list(err: Any) -> List[str]:
    if err is None:
        return []
    if isinstance(err, str):
        return [err] if err else []
    if isinstance(err, dict):
        return [str(err)]
    if isinstance(err, Iterable):
        return [str(e) for e in err]
    return [str(err)]


def _jsonschema_validator(schema: Any) -> Callable[[Any], ValidatorResult]:
    if schema is None:
        raise ValueError("json_schema must be provided for jsonschema validation")
    try:
        from jsonschema import Draft7Validator  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "jsonschema package is required for json schema validation. Install with `pip install jsonschema`."
        ) from exc

    validator = Draft7Validator(schema)

    def _validate(data: Any) -> ValidatorResult:
        errors = []
        for err in validator.iter_errors(data):
            path = "/".join(str(p) for p in err.path) or "<root>"
            errors.append(f"{path}: {err.message}")
        return ValidatorResult(ok=not errors, errors=errors, validator_name="jsonschema")

    _validate.__validator_name__ = "jsonschema"  # type: ignore[attr-defined]
    return _validate


def _pydantic_validator(model: Any) -> Callable[[Any], ValidatorResult]:
    try:
        from pydantic import BaseModel, ValidationError
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "pydantic is required for pydantic validation. Install with `pip install pydantic`."
        ) from exc

    if isinstance(model, type) and issubclass(model, BaseModel):
        model_cls = model
    else:
        raise ValueError("pydantic validator expects a BaseModel subclass as payload")

    def _validate(data: Any) -> ValidatorResult:
        try:
            if hasattr(model_cls, "model_validate"):
                model_cls.model_validate(data)  # type: ignore[attr-defined]
            else:  # pragma: no cover - pydantic v1 fallback
                model_cls.parse_obj(data)  # type: ignore[attr-defined]
            return ValidatorResult(ok=True, errors=[], validator_name=model_cls.__name__)
        except ValidationError as err:  # pragma: no cover - depends on user model
            return ValidatorResult(
                ok=False,
                errors=[err.json()],
                validator_name=model_cls.__name__,
            )

    _validate.__validator_name__ = model_cls.__name__  # type: ignore[attr-defined]
    return _validate


def _callable_validator(fn: Callable[[Any], Any], name: Optional[str] = None) -> Callable[[Any], ValidatorResult]:
    validator_name = name or getattr(fn, "__name__", "callable")

    def _validate(data: Any) -> ValidatorResult:
        outcome = fn(data)
        if outcome is None:
            return ValidatorResult(ok=True, errors=[], validator_name=validator_name)
        if isinstance(outcome, bool):
            return ValidatorResult(ok=outcome, errors=[] if outcome else [f"{validator_name} returned False"], validator_name=validator_name)
        if isinstance(outcome, tuple) and len(outcome) == 2:
            ok, err = outcome
            err_list = _coerce_error_list(err)
            return ValidatorResult(ok=bool(ok), errors=err_list, validator_name=validator_name)
        if isinstance(outcome, str):
            err_list = _coerce_error_list(outcome)
            return ValidatorResult(
                ok=False,
                errors=err_list or [f"{validator_name} returned an error"],
                validator_name=validator_name,
            )
        if isinstance(outcome, Iterable) and not isinstance(outcome, dict):
            err_list = _coerce_error_list(outcome)
            if err_list:
                return ValidatorResult(ok=False, errors=err_list, validator_name=validator_name)
            return ValidatorResult(ok=True, errors=[], validator_name=validator_name)
        return ValidatorResult(ok=True, errors=[], validator_name=validator_name)

    _validate.__validator_name__ = validator_name  # type: ignore[attr-defined]
    return _validate


def resolve_validators(
    validators: Optional[Sequence[ValidatorLike]],
    *,
    json_schema: Any = None,
) -> List[Callable[[Any], ValidatorResult]]:
    resolved: List[Callable[[Any], ValidatorResult]] = []
    specs: Sequence[ValidatorLike]
    if validators is None:
        specs = ["jsonschema"] if json_schema is not None else []
    else:
        specs = validators

    for spec in specs:
        if spec in {"jsonschema", "json_schema"}:
            resolved.append(_jsonschema_validator(json_schema))
        elif isinstance(spec, tuple) and spec and spec[0] == "pydantic":
            resolved.append(_pydantic_validator(spec[1]))
        elif spec == "pydantic":
            raise ValueError(
                "Validator 'pydantic' requires payload: provide as ('pydantic', BaseModelSubclass)"
            )
        elif callable(spec):
            resolved.append(_callable_validator(spec))
        else:
            raise ValueError(f"Unsupported validator specification: {spec!r}")
    return resolved
