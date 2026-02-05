from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .api import GenerationConfig, generate
from .structure.adherence import JsonAdherence
from .structure.grammar import Grammar
from .structure.logs import JsonlLogWriter
from .structure.semantic import build_semantic_checks
from .structure.stream import generate_stream
from .structure.validators import ValidatorLike
from .eval import EvalSuite
from .loader import auto_load, _sanitize_repo_id


def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_optional_int_list(value: Optional[str], *, flag_name: str) -> Optional[List[int]]:
    if value is None:
        return None
    try:
        parsed = _parse_int_list(value)
    except ValueError as exc:
        raise SystemExit(f"{flag_name} must be a comma-separated list of integers") from exc
    return parsed or None


def _resolve_safe_cache_subpath(cache_dir: str, model_id_or_path: str) -> tuple[str, str]:
    model_input = (model_id_or_path or "").strip()
    if not model_input:
        raise SystemExit("Model id/path cannot be empty")

    local_name = _sanitize_repo_id(model_input)
    if not local_name or local_name in {".", ".."} or not any(ch.isalnum() for ch in local_name):
        raise SystemExit(f"Model id/path '{model_id_or_path}' cannot be converted to a safe cache directory")

    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    local_path = (cache_root / local_name).resolve()
    if local_path == cache_root:
        raise SystemExit("Refusing to operate on cache root")
    try:
        local_path.relative_to(cache_root)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise SystemExit("Refusing to operate outside cache directory") from exc
    return model_input, str(local_path)


def _looks_like_chat_model(model_id_or_path: str) -> bool:
    """Heuristic to decide if a model is a chat/instruct variant.

    Used only for CLI defaulting; users can override with --no-auto-chat.
    """
    t = (model_id_or_path or "").lower()
    # Strong signals
    if "instruct" in t or "-chat" in t or "/chat" in t or "chat-" in t or "-it" in t:
        return True
    # Family-specific loose matches (still safe due to tokenizer-side guard)
    if "llama-3" in t or "llama3" in t:
        return ("instruct" in t) or ("chat" in t) or ("-it" in t)
    if "mistral" in t or "mixtral" in t:
        return ("instruct" in t) or ("-it" in t)
    if "gemma" in t:
        return ("instruct" in t) or ("-it" in t)
    if "phi-3" in t or "phi-4" in t:
        return ("instruct" in t) or ("-it" in t)
    if "qwen" in t:
        return True  # Many Qwen weights ship with chat templates; API guards if absent
    return False


def _load_json(path: str) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_json_schema(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    return _load_json(path)


def _build_skeleton(schema: Optional[Dict[str, Any]]) -> Any:
    if not isinstance(schema, dict):
        return {}
    schema_type = schema.get("type")
    if schema_type == "object" or (schema_type is None and "properties" in schema):
        props = schema.get("properties", {})
        required = schema.get("required", list(props.keys()))
        out: Dict[str, Any] = {}
        for key in required:
            out[key] = _build_skeleton(props.get(key, {}))
        return out
    if schema_type == "array":
        return []
    if schema_type in {"number", "integer"}:
        return 0
    if schema_type == "boolean":
        return False
    return ""


def _parse_validator_spec(spec: str) -> ValidatorLike:
    if spec in {"jsonschema", "json_schema"}:
        return "jsonschema"
    if spec.startswith("pydantic:"):
        module_path = spec.split(":", 1)[1]
        module_name, _, attr = module_path.rpartition(".")
        if not module_name:
            raise ValueError("pydantic validator requires module.Class path")
        module = importlib.import_module(module_name)
        model_cls = getattr(module, attr)
        return ("pydantic", model_cls)
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
        module = importlib.import_module(module_name)
        fn = getattr(module, attr)
        return fn
    raise ValueError(f"Unsupported validator spec: {spec}")


def _load_semantic_checks(path: Optional[str]) -> Optional[List[Any]]:
    if not path:
        return None
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError("semantic checks file must describe a list")
    return build_semantic_checks(payload)


def _resolve_grammar(args, json_schema: Optional[Dict[str, Any]]) -> Optional[Grammar]:
    if args.grammar_gbnf:
        grammar_text = Path(args.grammar_gbnf).expanduser().read_text(encoding="utf-8")
        return Grammar.gbnf(grammar_text)
    if json_schema is not None:
        return Grammar.json_schema(json_schema)
    return None


def _make_parse_fail_hook(kind: Optional[str], schema: Optional[Dict[str, Any]]):
    if not kind:
        return None
    if kind == "skeleton":
        skeleton_obj = _build_skeleton(schema)
        skeleton_text = json.dumps(skeleton_obj, indent=2)

        def hook(_: str, __: Exception) -> Optional[str]:
            return skeleton_text

        return hook
    if kind == "trim":
        def hook(text: str, _: Exception) -> Optional[str]:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return text[start : end + 1]
            return None

        return hook
    raise ValueError(f"Unknown on-parse-fail handler: {kind}")


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(base.get(key), value)
        return merged
    return override if override is not None else base


def _make_semantic_fail_hook(kind: Optional[str], schema: Optional[Dict[str, Any]]):
    if not kind:
        return None
    if kind == "fill-required":
        skeleton_obj = _build_skeleton(schema)

        def hook(obj: Dict[str, Any], _violations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
            return _deep_merge(skeleton_obj, copy.deepcopy(obj))

        return hook
    raise ValueError(f"Unknown on-semantic-fail handler: {kind}")


def _build_structured_kwargs(
    args: argparse.Namespace,
    json_schema: Optional[Dict[str, Any]],
    grammar: Optional[Grammar],
) -> Dict[str, Any]:
    validators: Optional[List[ValidatorLike]] = None
    if args.validator:
        validators = [_parse_validator_spec(v) for v in args.validator]

    semantic_checks = _load_semantic_checks(args.semantic_checks)

    needs_adherence = any(
        [
            json_schema is not None,
            validators,
            semantic_checks,
            bool(args.strict_only_json),
            int(args.retries) > 0,
            args.on_parse_fail,
            args.on_semantic_fail,
            args.log_jsonl,
        ]
    )

    adherence = (
        JsonAdherence(
            retries=max(0, int(args.retries)),
            strict_only_json=bool(args.strict_only_json),
        )
        if needs_adherence
        else None
    )

    parse_hook = _make_parse_fail_hook(args.on_parse_fail, json_schema)
    semantic_hook = _make_semantic_fail_hook(args.on_semantic_fail, json_schema)
    log_writer = (
        JsonlLogWriter(args.log_jsonl, include_raw_on_fail=args.log_raw_on_fail)
        if args.log_jsonl
        else None
    )

    kwargs: Dict[str, Any] = {
        "json_schema": json_schema,
        "grammar": grammar,
        "validators": validators,
        "semantic_checks": semantic_checks,
        "adherence": adherence,
        "on_parse_fail": parse_hook,
        "on_semantic_fail": semantic_hook,
        "log_writer": log_writer,
    }

    return {key: value for key, value in kwargs.items() if value is not None}


def generate_cmd():
    ap = argparse.ArgumentParser(prog="mlxgk.generate", description="Generate text with MLX + generation-parity features")
    ap.add_argument("--model", required=True, help="HF repo id or local MLX path")
    ap.add_argument("--prompt", required=False, help="Prompt text")
    ap.add_argument(
        "--messages-json",
        default=None,
        help="JSON list of chat messages [{role, content}, ...] to apply the model's chat template",
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--typical-p", type=float, default=0.0)
    ap.add_argument("--epsilon-cutoff", type=float, default=0.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--min-new-tokens", type=int, default=0)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--length-penalty", type=float, default=0.0)
    ap.add_argument("--early-stopping", action="store_true")
    ap.add_argument("--backend", default=None, help="Force backend implementation (default: auto)")
    # Chat template controls
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--auto-chat", action="store_true", help="Auto-apply chat template for plain prompts if the tokenizer has one")
    grp.add_argument("--no-auto-chat", action="store_true", help="Disable auto chat templating")
    grp.add_argument("--assume-user-chat", action="store_true", help="Treat a plain prompt as a user message and apply chat template if available")
    ap.add_argument("--system", default=None, help="Optional system prompt when using chat templating")
    ap.add_argument("--stop", default=None, help="Comma-separated stop sequences")
    ap.add_argument(
        "--stop-strings",
        default=None,
        help="Alias for --stop; comma-separated stop strings (will be merged)",
    )
    ap.add_argument("--suppress-tokens", default=None, help="Comma-separated token ids to suppress (ints)")
    ap.add_argument("--begin-suppress-tokens", default=None, help="Comma-separated token ids suppressed at first step")
    ap.add_argument("--force-words", default=None, help="Comma-separated phrases to force (joined by tokenizer)")
    ap.add_argument("--speculative", action="store_true")
    ap.add_argument("--draft-model", default=None, help="Draft model id/path for speculative decoding")
    # Structured generation options
    ap.add_argument("--json-schema", default=None, help="Path to JSON Schema (object) file for structured adherence")
    ap.add_argument("--grammar-gbnf", default=None, help="Optional GBNF grammar file")
    ap.add_argument("--retries", type=int, default=0, help="Number of structured adherence retries")
    ap.add_argument("--strict-only-json", action="store_true", help="Require responses to contain JSON only")
    ap.add_argument(
        "--validator",
        action="append",
        default=[],
        help="Validator specification (jsonschema, pydantic:module.Model, or module:function)",
    )
    ap.add_argument("--semantic-checks", default=None, help="Path to JSON file describing semantic checks")
    ap.add_argument(
        "--on-parse-fail",
        choices=["skeleton", "trim"],
        default=None,
        help="Post-processor when JSON parsing fails",
    )
    ap.add_argument(
        "--on-semantic-fail",
        choices=["fill-required"],
        default=None,
        help="Post-processor when semantic checks fail",
    )
    ap.add_argument("--log-jsonl", default=None, help="Path to append structured adherence logs (JSONL)")
    ap.add_argument("--log-raw-on-fail", action="store_true", help="Include raw text when schema validation fails")
    ap.add_argument("--pretty", action="store_true", help="Pretty print JSON output if available")
    ap.add_argument("--stream", action="store_true", help="Stream tokens (best-effort) during generation")
    args = ap.parse_args()

    # Auto-load (convert if needed)
    model, tokenizer, local_path = auto_load(args.model)
    print(f"[mlx-genkit] using model from {local_path}", file=sys.stderr)

    # Determine prompt input: messages (JSON) or plain string
    prompt_input: Any
    if args.messages_json:
        import json

        prompt_input = json.loads(args.messages_json)
    else:
        if args.prompt is None:
            raise SystemExit("Either --prompt or --messages-json must be provided")
        prompt_input = args.prompt

    force_words_ids = None
    if args.force_words:
        phrases = [p for p in args.force_words.split(",") if p.strip()]
        force_words_ids = [tokenizer.encode(p, add_special_tokens=False) for p in phrases]
    suppress_tokens = _parse_optional_int_list(args.suppress_tokens, flag_name="--suppress-tokens")
    begin_suppress_tokens = _parse_optional_int_list(
        args.begin_suppress_tokens,
        flag_name="--begin-suppress-tokens",
    )

    # Merge legacy --stop and alias --stop-strings
    stop_sequences = None
    stop_alias = []
    if args.stop:
        stop_alias.extend([s for s in args.stop.split(",") if s])
    if args.stop_strings:
        stop_alias.extend([s for s in args.stop_strings.split(",") if s])
    if stop_alias:
        # de-duplicate, preserve order
        seen = set()
        merged = []
        for s in stop_alias:
            if s not in seen:
                seen.add(s)
                merged.append(s)
        stop_sequences = merged

    # Decide default auto-chat if user didn't set a preference
    default_auto_chat = None
    # Prefer the tokenizer's actual capability (most reliable)
    has_template = bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")
    if not (args.auto_chat or args.no_auto_chat or args.assume_user_chat):
        default_auto_chat = True if has_template else None

    cfg = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_p=args.top_p,
        top_k=args.top_k,
        typical_p=args.typical_p,
        epsilon_cutoff=args.epsilon_cutoff,
        seed=args.seed,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_new_tokens=args.min_new_tokens if args.min_new_tokens > 0 else None,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping,
        stop_sequences=stop_sequences,
        auto_chat_template=(
            True
            if (args.assume_user_chat or args.auto_chat)
            else (False if args.no_auto_chat else default_auto_chat)
        ),
        system_prompt=args.system,
        assume_user_chat=args.assume_user_chat,
        stop_strings=stop_sequences,
        force_words_ids=force_words_ids,
        suppress_tokens=suppress_tokens,
        begin_suppress_tokens=begin_suppress_tokens,
        use_speculative=args.speculative,
        draft_model_id=args.draft_model,
        backend=args.backend,
    )

    # Telemetry: report whether chat templating will be applied
    chat_input = "messages" if args.messages_json else "plain"
    # Mirror library logic for plain prompts: assume_user_chat OR (auto_chat_template if not None else has_template)
    final_auto = cfg.assume_user_chat or (cfg.auto_chat_template if cfg.auto_chat_template is not None else has_template)
    applied = has_template and (chat_input == "messages" or final_auto)
    print(
        f"[mlx-genkit] chat templating: applied={'yes' if applied else 'no'} "
        f"(input={chat_input}, tokenizer_template={'yes' if has_template else 'no'}, "
        f"auto={cfg.auto_chat_template}, assume_user={cfg.assume_user_chat})",
        file=sys.stderr,
    )
    json_schema = _load_json_schema(args.json_schema)
    grammar = _resolve_grammar(args, json_schema)
    generation_kwargs = _build_structured_kwargs(args, json_schema, grammar)

    backend_name = (cfg.backend or "mlx").lower()
    if grammar and not grammar.supports_backend(backend_name):
        if grammar.kind != "json_schema":
            raise SystemExit(
                f"Grammar kind '{grammar.kind}' is not supported by backend '{backend_name}'."
            )
        print(
            f"[mlx-genkit] backend '{backend_name}' does not provide native grammar support; falling back to validator retries.",
            file=sys.stderr,
        )

    if args.stream:
        stream_printed = False

        def _on_token(tok: Any, _idx: int) -> None:
            nonlocal stream_printed
            stream_printed = True
            if isinstance(tok, int):
                fragment = tokenizer.decode([tok])
            else:
                fragment = str(tok)
            if fragment:
                sys.stdout.write(fragment)
                sys.stdout.flush()

        result = generate_stream(
            model,
            tokenizer,
            prompt_input,
            cfg,
            hooks=None,
            on_token=_on_token,
            **generation_kwargs,
        )
        if stream_printed:
            print()
    else:
        result = generate(
            model,
            tokenizer,
            prompt_input,
            cfg,
            hooks=None,
            **generation_kwargs,
        )

    if args.pretty and result.json is not None:
        print(json.dumps(result.json, indent=2))  # noqa: T201
    elif not args.stream:
        print(result.text)  # noqa: T201

    print(
        f"[mlx-genkit] schema_ok={result.schema_ok} only_json={result.only_json} "
        f"semantic_ok={result.semantic_ok} attempts={result.attempts}",
        file=sys.stderr,
    )


def download_cmd():
    ap = argparse.ArgumentParser(prog="mlxgk.download", description="Download and convert an HF model to MLX format, print local path")
    ap.add_argument("--model", required=True, help="HF repo id or local MLX path")
    ap.add_argument("--cache-dir", default=None, help="Cache directory for converted models (default: ./mlx_cache)")
    ap.add_argument("--quantize", action="store_true", help="Quantize during conversion")
    ap.add_argument("--trust-remote-code", action="store_true", help="Allow custom model code from the repo")
    ap.add_argument("--force", action="store_true", help="Reconvert even if cached model exists (deletes existing cache dir)")
    args = ap.parse_args()

    cache_dir = args.cache_dir or os.path.join(os.getcwd(), "mlx_cache")
    model_input, local_path = _resolve_safe_cache_subpath(cache_dir, args.model)

    if args.force and os.path.exists(local_path):
        import shutil

        if os.path.islink(local_path) or not os.path.isdir(local_path):
            raise SystemExit(f"Refusing to delete non-directory cache path: {local_path}")
        shutil.rmtree(local_path)

    # Convert without loading into memory
    _m, _t, out_path = auto_load(
        model_input,
        cache_dir=str(Path(cache_dir).expanduser().resolve()),
        quantize=args.quantize,
        trust_remote_code=args.trust_remote_code,
        load_model=False,
    )
    print(out_path)  # noqa: T201


def eval_cmd():
    ap = argparse.ArgumentParser(prog="mlxgk.eval", description="Run adherence evaluation suites and emit a markdown report")
    ap.add_argument("--suite", required=True, help="Path to suite YAML/JSON definition")
    ap.add_argument("--markdown", default=None, help="Optional path to write markdown report")
    ap.add_argument("--json", default=None, help="Optional path to write JSON summary")
    args = ap.parse_args()

    suite = EvalSuite(args.suite)
    outcomes = suite.run()
    markdown = EvalSuite.render_markdown(outcomes, suite.name)
    summary = EvalSuite.to_dict(outcomes, suite.name)

    if args.markdown:
        Path(args.markdown).expanduser().write_text(markdown, encoding="utf-8")
    else:
        print(markdown)  # noqa: T201

    if args.json:
        Path(args.json).expanduser().write_text(json.dumps(summary, indent=2), encoding="utf-8")
