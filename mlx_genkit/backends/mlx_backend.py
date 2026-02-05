from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Sequence, Tuple, Union

from ..adapters import make_tokenizer_bridge, detect_components, project_logits, ModelComponents
from ..injection import ResidualInjectionHook, LogitBiasHook, ResidualInjector
from ..sampling import (
    make_sampler,
    make_logits_processor_chain,
    make_force_words_processor,
)
from ..structure.result import GenerateResult
from ..config import GenerationConfig
from ..utils import (
    try_import_mlx,
    try_import_mlx_lm_cache,
    set_seed,
    stable_log_softmax,
    as_mx_array,
    filter_kwargs_for_callable,
)


@dataclass
class GenerationSetup:
    """Container for precomputed generation state used by MLX backend."""

    model: Any
    tokenizer: Any
    config: GenerationConfig
    hooks: Optional[List[Union[ResidualInjectionHook, LogitBiasHook]]]
    tk: Any
    prompt_tokens: List[int]
    components: ModelComponents
    eos_ids: List[int]
    processors: List[Callable[[List[int], Any], Any]]
    stop_token_sequences: List[List[int]]
    raw_stop_strings: List[str]
    prompt_cache: Any
    residual_hooks: List[ResidualInjectionHook]
    injector: Optional[ResidualInjector]
    forced_map: Dict[int, int]
    forced_bos: Optional[int]


@dataclass
class _IncrementalRawStopState:
    generated_text: str
    max_stop_length: int
    enabled: bool = True


def _decode_generated_suffix(tk: Any, tokens: Sequence[int], prompt_len: int) -> str:
    if len(tokens) <= prompt_len:
        return ""
    return tk.decode(list(tokens[prompt_len:]))


def _find_earliest_stop_index(text: str, raw_stops: Sequence[str]) -> Optional[int]:
    earliest: Optional[int] = None
    for stop in raw_stops:
        if not stop:
            continue
        idx = text.find(stop)
        if idx < 0:
            continue
        if earliest is None or idx < earliest:
            earliest = idx
    return earliest


def _trim_tokens_by_generated_text(
    tk: Any,
    tokens: List[int],
    prompt_len: int,
    generated_text: str,
) -> List[int]:
    trimmed_gen_ids = tk.encode(generated_text, add_special_tokens=False)
    return tokens[:prompt_len] + trimmed_gen_ids


def _match_raw_stops_full_decode(
    tk: Any,
    tokens: List[int],
    prompt_len: int,
    raw_stops: Sequence[str],
) -> Tuple[List[int], bool]:
    gen_tokens = tokens[prompt_len:]
    if not gen_tokens:
        return tokens, False
    decoded_gen = tk.decode(gen_tokens)
    idx = _find_earliest_stop_index(decoded_gen, raw_stops)
    if idx is None:
        return tokens, False
    return _trim_tokens_by_generated_text(tk, tokens, prompt_len, decoded_gen[:idx]), True


def _make_incremental_raw_stop_state(
    raw_stops: Sequence[str],
    *,
    enabled: bool,
) -> Optional[_IncrementalRawStopState]:
    if not enabled:
        return None
    max_stop_len = max((len(stop) for stop in raw_stops if stop), default=0)
    if max_stop_len <= 0:
        return None
    return _IncrementalRawStopState(generated_text="", max_stop_length=max_stop_len, enabled=True)


def _match_raw_stops_incremental(
    tk: Any,
    tokens: List[int],
    prompt_len: int,
    raw_stops: Sequence[str],
    state: _IncrementalRawStopState,
    new_token: int,
) -> Tuple[List[int], bool]:
    if state.enabled:
        try:
            piece = tk.decode([new_token])
        except Exception:
            state.enabled = False
        else:
            if piece:
                state.generated_text += piece
                search_from = max(0, len(state.generated_text) - len(piece) - state.max_stop_length)
                idx = _find_earliest_stop_index(state.generated_text[search_from:], raw_stops)
                if idx is not None:
                    stop_idx = search_from + idx
                    trimmed = state.generated_text[:stop_idx]
                    state.generated_text = trimmed
                    return _trim_tokens_by_generated_text(tk, tokens, prompt_len, trimmed), True
    if not state.enabled:
        tokens, hit = _match_raw_stops_full_decode(tk, tokens, prompt_len, raw_stops)
        if hit:
            state.generated_text = _decode_generated_suffix(tk, tokens, prompt_len)
        return tokens, hit
    return tokens, False


def _detect_tokenizer_vocab_size(tokenizer: Any) -> Optional[int]:
    vocab = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab, int) and vocab > 0:
        return vocab
    try:
        size = len(tokenizer)
    except Exception:
        size = None
    if isinstance(size, int) and size > 0:
        return size
    return None


def _check_speculative_tokenizer_compatibility(
    base_tokenizer: Any,
    draft_tokenizer: Any,
    *,
    prompt_hint: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    if draft_tokenizer is None:
        return False, "speculative_draft_tokenizer_unavailable"
    base_tk = make_tokenizer_bridge(base_tokenizer)
    draft_tk = make_tokenizer_bridge(draft_tokenizer)

    base_vocab = _detect_tokenizer_vocab_size(base_tokenizer)
    draft_vocab = _detect_tokenizer_vocab_size(draft_tokenizer)
    if base_vocab is not None and draft_vocab is not None and base_vocab != draft_vocab:
        return False, "speculative_vocab_size_mismatch"

    if (
        base_tk.eos_token_id is not None
        and draft_tk.eos_token_id is not None
        and base_tk.eos_token_id != draft_tk.eos_token_id
    ):
        return False, "speculative_eos_token_mismatch"

    probes = ["", "a", "hello", " hello", "\n", "A quick compatibility probe."]
    if prompt_hint:
        probes.append(prompt_hint[:64])
    checked_any = False
    for idx, text in enumerate(probes):
        try:
            base_ids = base_tk.encode(text, add_special_tokens=False)
            draft_ids = draft_tk.encode(text, add_special_tokens=False)
        except Exception:
            continue
        checked_any = True
        if base_ids != draft_ids:
            return False, f"speculative_encoding_mismatch_probe_{idx}"
    if not checked_any:
        return True, None
    return True, None


def _prepare_prompt(tokenizer, prompt: Union[str, Sequence[int]]):
    tk = make_tokenizer_bridge(tokenizer)
    if isinstance(prompt, str):
        ids = tk.encode(prompt)
    else:
        ids = list(prompt)
    return ids, tk


from typing import Optional as _Optional  # local alias to avoid collision


def _maybe_render_chat_prompt(tokenizer: Any, prompt: Any, config: _Optional[GenerationConfig] = None) -> Any:
    """If prompt looks like HF-style chat `messages`, render with chat template.

    Accepts a list of {role, content} dicts (or tuples) and returns a string
    produced via `tokenizer.apply_chat_template` when available, otherwise a
    simple fallback formatting. If `prompt` is already a string or a list of
    ints, it is returned unchanged.
    """
    # Fast-path: strings and explicit token id sequences are left unchanged
    if isinstance(prompt, str):
        # Optional auto-application for plain strings
        auto = None
        if config is not None:
            # assume_user_chat explicitly forces chat templating for plain prompts
            if getattr(config, "assume_user_chat", False):
                auto = True
            else:
                auto = config.auto_chat_template
        # Heuristic default: if tokenizer exposes a non-empty chat_template, assume chat model
        if auto is None:
            auto = bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")
        if auto:
            try:
                messages = []
                if config and getattr(config, "system_prompt", None):
                    messages.append({"role": "system", "content": config.system_prompt})
                messages.append({"role": "user", "content": prompt})
                from ..interop import apply_chat_template  # local import

                return apply_chat_template(tokenizer, messages, add_generation_prompt=True)
            except Exception:
                # Fallback: return unchanged
                return prompt
        return prompt
    if isinstance(prompt, (list, tuple)) and all(isinstance(x, int) for x in prompt):
        return prompt
    # Detect list of chat message dicts
    is_messages = False
    if isinstance(prompt, (list, tuple)) and prompt:
        first = prompt[0]
        if isinstance(first, dict) and ("role" in first and "content" in first):
            is_messages = True
    if not is_messages:
        return prompt
    # Render using interop helper
    try:
        from ..interop import apply_chat_template  # local import to avoid cycles

        return apply_chat_template(tokenizer, prompt, add_generation_prompt=True)
    except Exception:
        # If anything goes wrong, just pass prompt through
        return prompt


def _maybe_make_prompt_cache(model, max_kv_size: Optional[int] = None):
    lm_cache = try_import_mlx_lm_cache()
    if lm_cache is None:
        return None
    try:
        return lm_cache.make_prompt_cache(model, max_kv_size=max_kv_size)
    except Exception:
        return None


def _resolve_eos_ids(config: GenerationConfig, tk) -> List[int]:
    if config.eos_token_ids is not None:
        return [i for i in config.eos_token_ids if i is not None]
    if config.eos_token_id is not None:
        return [config.eos_token_id]
    if tk.eos_token_id is not None:
        return [tk.eos_token_id]
    return []


def _collect_stop_sequences(
    tokenizer: Any,
    tk,
    config: GenerationConfig,
) -> Tuple[List[str], List[List[int]]]:
    raw: List[str] = []
    if config.stop_sequences:
        raw.extend([s for s in config.stop_sequences if s])
    if config.stop_strings:
        for s in config.stop_strings:
            if s and s not in raw:
                raw.append(s)
    token_seqs: List[List[int]] = []
    if raw:
        bridge = tk if tk is not None else make_tokenizer_bridge(tokenizer)
        for item in raw:
            encoded = bridge.encode(item, add_special_tokens=False) if item else []
            if encoded:
                token_seqs.append(encoded)
    return raw, token_seqs


def _build_generation_setup(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    config: GenerationConfig,
    hooks: Optional[List[Union[ResidualInjectionHook, LogitBiasHook]]],
) -> GenerationSetup:
    rendered_prompt = _maybe_render_chat_prompt(tokenizer, prompt, config)
    prompt_tokens, tk = _prepare_prompt(tokenizer, rendered_prompt)
    components = detect_components(model)
    eos_ids = _resolve_eos_ids(config, tk)

    bias_vec = None
    residual_hooks: List[ResidualInjectionHook] = []
    if hooks:
        for hook in hooks:
            if isinstance(hook, LogitBiasHook):
                vec = hook.resolve_vector(model)
                bias = _compute_bias_from_vector(components, vec, hook.alpha)
                bias_vec = bias if bias_vec is None else (bias_vec + bias)
            if isinstance(hook, ResidualInjectionHook):
                residual_hooks.append(hook)

    processors = make_logits_processor_chain(
        repetition_penalty=config.repetition_penalty,
        repetition_context_size=config.repetition_context_size,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
        frequency_penalty=config.frequency_penalty,
        presence_penalty=config.presence_penalty,
        bias_vector=bias_vec,
        bad_words_ids=config.bad_words_ids,
        min_new_tokens=config.min_new_tokens,
        eos_token_ids=eos_ids,
        prompt_len=len(prompt_tokens),
        suppress_tokens=config.suppress_tokens,
        begin_suppress_tokens=config.begin_suppress_tokens,
        forced_decoder_map={pos: tid for (pos, tid) in (config.forced_decoder_ids or [])},
        forced_bos_token_id=config.forced_bos_token_id,
    )
    if config.force_words_ids:
        processors.append(
            make_force_words_processor(
                config.force_words_ids, prompt_len=len(prompt_tokens), strict_start=True
            )
        )

    raw_stops, stop_token_seqs = _collect_stop_sequences(tokenizer, tk, config)
    prompt_cache = _maybe_make_prompt_cache(model, max_kv_size=config.max_kv_size)

    injector = (
        ResidualInjector(components.layers, components.hidden_size)
        if residual_hooks
        else None
    )

    return GenerationSetup(
        model=model,
        tokenizer=tokenizer,
        config=config,
        hooks=hooks,
        tk=tk,
        prompt_tokens=prompt_tokens,
        components=components,
        eos_ids=eos_ids,
        processors=processors,
        stop_token_sequences=stop_token_seqs,
        raw_stop_strings=raw_stops,
        prompt_cache=prompt_cache,
        residual_hooks=residual_hooks,
        injector=injector,
        forced_map={pos: tid for (pos, tid) in (config.forced_decoder_ids or [])},
        forced_bos=config.forced_bos_token_id,
    )


def _step_logits(model, components: ModelComponents, input_tokens, cache=None, input_embeddings=None):
    """Compute last-position logits, handling common MLX(-LM) return conventions."""

    def _split_output(output: Any, prev_cache: Any):
        logits_local = output
        next_cache = prev_cache
        if isinstance(output, dict):
            if "logits" in output:
                logits_local = output["logits"]
            if "cache" in output:
                next_cache = output["cache"]
        elif isinstance(output, (tuple, list)) and output:
            logits_local = output[0]
            if len(output) >= 2:
                next_cache = output[1]
        else:
            if hasattr(output, "logits"):
                logits_local = getattr(output, "logits")
            if hasattr(output, "cache"):
                next_cache = getattr(output, "cache")
        return logits_local, next_cache

    # Some models do not support input_embeddings kwarg.
    if input_embeddings is not None:
        try:
            out = model(input_tokens, cache=cache, input_embeddings=input_embeddings)  # type: ignore
        except TypeError:
            out = model(input_tokens, cache=cache)  # type: ignore
    else:
        out = model(input_tokens, cache=cache)  # type: ignore

    logits, next_cache = _split_output(out, cache)

    # Some models may return hidden states prior to head; detect via shape heuristic.
    hidden_size = int(getattr(components, "hidden_size", 0) or 0)
    vocab_size = int(getattr(components, "vocab_size", 0) or 0)
    if hidden_size and hasattr(logits, "ndim") and logits.ndim in (2, 3) and logits.shape[-1] == hidden_size:
        if (not vocab_size) or vocab_size != hidden_size:
            logits = project_logits(components, logits)

    if not hasattr(logits, "ndim"):
        raise ValueError("Model output does not look like an array")  # pragma: no cover - defensive

    if logits.ndim == 3:
        return logits[:, -1, :], next_cache
    if logits.ndim == 2:
        return logits, next_cache
    if logits.ndim == 1:
        return logits.reshape((1, -1)), next_cache
    raise ValueError(f"Unsupported logits rank: {logits.ndim}")  # pragma: no cover - defensive


def _batched_last_logits(
    model,
    components: ModelComponents,
    batch_token_lists: List[List[int]],
):
    """Compute last-position logits for a batch of token sequences.

    Assumes all sequences are the same length (true within beam steps).
    """
    mx, _ = try_import_mlx()
    arr = as_mx_array(batch_token_lists, dtype=mx.int32)
    logits, _ = _step_logits(model, components, arr, cache=None)
    return logits


def forward_with_hidden(
    model, tokenizer, tokens: Sequence[int], capture_layers: Optional[List[int]] = None, strict: bool = False
):
    """Forward tokens and optionally capture hidden states at specific layers.

    Note: With MLX compilation, Python-level patches may not trigger in some
    optimized paths. Captures can therefore be empty depending on model/compile
    settings. Logits are always returned for the final position.
    """
    mx, _ = try_import_mlx()
    components = detect_components(model)
    ids = as_mx_array(tokens, dtype=mx.int32)
    cache = None if strict else _maybe_make_prompt_cache(model)

    captured: Dict[int, Any] = {}
    if capture_layers and not strict:
        inj = ResidualInjector(components.layers, components.hidden_size)
        # Instrumentation: patch a no-op that captures outputs at specified layers
        to_capture = []
        for l in capture_layers:
            idx = l if l >= 0 else len(components.layers) + l
            if 0 <= idx < len(components.layers):
                to_capture.append(idx)

        class _Capture:
            def __init__(self, layer, idx):
                self.layer = layer
                self.idx = idx
                self._orig = layer.__call__

            def __call__(self, x, *args, **kwargs):
                out = self._orig(x, *args, **kwargs)
                captured[self.idx] = out
                return out

            def apply(self):
                self.layer.__call__ = self.__call__  # type: ignore

            def restore(self):
                self.layer.__call__ = self._orig  # type: ignore

        patches = []
        for idx in to_capture:
            p = _Capture(components.layers[idx], idx)
            p.apply()
            patches.append(p)
        logits, _ = _step_logits(model, components, ids[None], cache)
        for p in patches:
            p.restore()
    elif strict:
        # Manual forward with proper causal mask and no cache
        try:
            from mlx_lm.models.base import create_attention_mask  # type: ignore
        except Exception:
            create_attention_mask = None
        h = components.embed(ids[None])
        mask = create_attention_mask(h, None) if create_attention_mask is not None else "causal"
        to_capture = set()
        if capture_layers:
            for l in capture_layers:
                idx = l if l >= 0 else len(components.layers) + l
                if 0 <= idx < len(components.layers):
                    to_capture.add(idx)
        for i, layer in enumerate(components.layers):
            h = layer(h, mask, cache=None)
            if i in to_capture:
                captured[i] = h
        if components.norm is not None:
            h = components.norm(h)
        logits_full = project_logits(components, h)
        logits = logits_full[:, -1, :]
    else:
        logits, _ = _step_logits(model, components, ids[None], cache)

    return logits, captured


def _compute_bias_from_vector(components: ModelComponents, vec, alpha: float):
    # bias = (W @ v) scaled by alpha
    proj = components.vocab_projection
    bias = proj.project(vec.reshape((1, -1))).reshape((-1,))
    mx, _ = try_import_mlx()
    return (bias * alpha).astype(mx.float32)


def generate_dict(
    model: Any,
    tokenizer: Any,
    prompt: Union[str, Sequence[int]],
    config: GenerationConfig,
    hooks: Optional[List[Union[ResidualInjectionHook, LogitBiasHook]]] = None,
    stream_observer: Optional[Any] = None,
) -> Dict[str, Any]:
    """HF-compatible sampling interface over MLX models.

    - Applies processors (repetition penalty, no-repeat-ngrams, bad-words, etc.).
    - Supports forced BOS/EOS and forced decoder ids.
    - Works with explicit lm_head or tied/quantized embeddings.
    """
    mx, _ = try_import_mlx()
    set_seed(config.seed)

    setup = _build_generation_setup(model, tokenizer, prompt, config, hooks)

    if config.num_beams and config.num_beams > 1:
        return _beam_search_generate(setup)

    processors = setup.processors
    tk = setup.tk
    prompt_ids = setup.prompt_tokens
    prompt_len = len(prompt_ids)
    tokens: List[int] = list(prompt_ids)
    y = as_mx_array(tokens, dtype=mx.int32)[None]
    n_generated = 0
    eos_reached = False
    finish_reason: Optional[str] = None
    speculative_fallback_reason: Optional[str] = None

    # Handle forced BOS token at first generation step
    forced_bos = setup.forced_bos
    forced_map = setup.forced_map
    eos_ids = setup.eos_ids
    prompt_cache = setup.prompt_cache
    residual_hooks = setup.residual_hooks
    injector = setup.injector
    stop_token_seqs = setup.stop_token_sequences
    raw_stops = setup.raw_stop_strings
    components = setup.components

    def _build_response(current_tokens: List[int], current_eos: bool, current_finish: Optional[str]) -> Dict[str, Any]:
        payload = {
            "text": _decode_generated_suffix(tk, current_tokens, prompt_len),
            "tokens": current_tokens,
            "eos_reached": current_eos,
            "finish_reason": current_finish,
            "stream_tokens_emitted": stream_observer.emitted if stream_observer else 0,
            "stream_invalid_path": bool(stream_observer and stream_observer.invalid_triggered),
        }
        if speculative_fallback_reason:
            payload["speculative_fallback_reason"] = speculative_fallback_reason
        return payload

    sampler = make_sampler(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        min_p=config.min_p,
        min_tokens_to_keep=config.min_tokens_to_keep,
        typical_p=config.typical_p,
        epsilon_cutoff=config.epsilon_cutoff,
    )

    active_stream = None
    invalid_stop = False
    if stream_observer is not None and not config.use_speculative and not (config.num_beams and config.num_beams > 1):
        active_stream = stream_observer
    main_raw_stop_state = _make_incremental_raw_stop_state(raw_stops, enabled=active_stream is None)

    # Fast path using mlx-lm generate_step when possible (no residual injection)
    if injector is None and not (config.num_beams and config.num_beams > 1) and not config.use_speculative:
        try:
            from mlx_lm.generate import generate_step as mlx_generate_step  # type: ignore
        except Exception:
            mlx_generate_step = None
        if mlx_generate_step is not None:
            sampler = make_sampler(
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                min_p=config.min_p,
                min_tokens_to_keep=config.min_tokens_to_keep,
                typical_p=config.typical_p,
                epsilon_cutoff=config.epsilon_cutoff,
            )

            tokens: List[int] = list(prompt_ids)
            n_generated = 0
            eos_reached = False
            finish_reason = None
            fast_raw_stop_state = _make_incremental_raw_stop_state(raw_stops, enabled=active_stream is None)

            # Use mlx-lm generate_step and our processors
            gen = mlx_generate_step(
                as_mx_array(tokens, dtype=mx.int32),
                model,
                **filter_kwargs_for_callable(
                    mlx_generate_step,
                    {
                        "max_tokens": config.max_tokens,
                        "sampler": sampler,
                        "logits_processors": processors,
                        "max_kv_size": config.max_kv_size,
                        "prompt_cache": prompt_cache,
                    },
                ),
            )
            for y, logprobs in gen:
                try:
                    t = int(y.item())
                except Exception:
                    t = int(y)
                tokens.append(t)
                n_generated += 1

                if active_stream is not None:
                    if not active_stream.emit_token(t):
                        invalid_stop = True
                        eos_reached = True
                        finish_reason = finish_reason or "invalid_path"
                        break

                # EOS checks
                if config.forced_eos_token_id is not None and t == config.forced_eos_token_id:
                    eos_reached = True
                    finish_reason = "eos"
                    break
                if eos_ids and t in eos_ids:
                    eos_reached = True
                    finish_reason = "eos"
                    break
                # Token-level stop sequences
                if stop_token_seqs:
                    hit = False
                    for seq in stop_token_seqs:
                        n = len(seq)
                        if n > 0 and len(tokens) >= n and tokens[-n:] == seq:
                            tokens = tokens[:-n]
                            eos_reached = True
                            finish_reason = "stop_sequence"
                            hit = True
                            break
                    if hit:
                        break
                # String fallback stops on generated suffix
                if raw_stops:
                    if fast_raw_stop_state is not None:
                        tokens, hit = _match_raw_stops_incremental(
                            tk,
                            tokens,
                            prompt_len,
                            raw_stops,
                            fast_raw_stop_state,
                            t,
                        )
                    else:
                        tokens, hit = _match_raw_stops_full_decode(tk, tokens, prompt_len, raw_stops)
                    if hit:
                        eos_reached = True
                        finish_reason = "stop_sequence"
                        break
                if n_generated >= config.max_tokens:
                    finish_reason = "length"
                    break

            if invalid_stop and not finish_reason:
                finish_reason = "invalid_path"
            if finish_reason is None and n_generated >= config.max_tokens:
                finish_reason = "length"
            return _build_response(tokens, eos_reached, finish_reason)

    # Speculative decoding path
    if config.use_speculative:
        try:
            from mlx_lm.generate import speculative_generate_step  # type: ignore
        except Exception:
            speculative_generate_step = None
        if speculative_generate_step is None:
            # Fallback to normal path if not available
            speculative_fallback_reason = "speculative_generate_step_unavailable"
        else:
            # Load draft model if provided via config
            draft_model = None
            draft_tokenizer = None
            if config.draft_model_id:
                try:
                    # Use auto_load to support HF repo ids with on-demand conversion
                    from ..loader import auto_load  # type: ignore

                    draft_model, draft_tokenizer, _local = auto_load(config.draft_model_id)
                except Exception:
                    draft_model = None
                    draft_tokenizer = None
                    speculative_fallback_reason = "speculative_draft_model_load_failed"
            else:
                speculative_fallback_reason = "speculative_draft_model_missing"
            if draft_model is None:
                # If no draft model, fallback to normal path
                if speculative_fallback_reason is None:
                    speculative_fallback_reason = "speculative_draft_model_unavailable"
            else:
                prompt_hint = prompt if isinstance(prompt, str) else None
                tokenizer_ok, reason = _check_speculative_tokenizer_compatibility(
                    tokenizer,
                    draft_tokenizer,
                    prompt_hint=prompt_hint,
                )
                if not tokenizer_ok:
                    speculative_fallback_reason = reason or "speculative_draft_tokenizer_incompatible"
                else:
                    # Build sampler on normalized logprobs
                    sampler = make_sampler(
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        min_p=config.min_p,
                        min_tokens_to_keep=config.min_tokens_to_keep,
                        typical_p=config.typical_p,
                        epsilon_cutoff=config.epsilon_cutoff,
                    )
                    tokens = list(prompt_ids)
                    n_generated = 0
                    eos_reached = False
                    finish_reason = None
                    speculative_raw_stop_state = _make_incremental_raw_stop_state(raw_stops, enabled=True)
                    # no prompt_cache rotation in this path beyond default
                    gen = speculative_generate_step(
                        as_mx_array(tokens, dtype=mx.int32),
                        model,
                        draft_model,
                        **filter_kwargs_for_callable(
                            speculative_generate_step,
                            {
                                "num_draft_tokens": config.num_draft_tokens,
                                "max_tokens": config.max_tokens,
                                "sampler": sampler,
                                "logits_processors": processors,
                                "prompt_cache": None,
                            },
                        ),
                    )
                    for y, logprobs, _from_draft in gen:
                        try:
                            t = int(y.item())
                        except Exception:
                            t = int(y)
                        tokens.append(t)
                        n_generated += 1
                        # EOS checks
                        if config.forced_eos_token_id is not None and t == config.forced_eos_token_id:
                            eos_reached = True
                            finish_reason = "eos"
                            break
                        if eos_ids and t in eos_ids:
                            eos_reached = True
                            finish_reason = "eos"
                            break
                        # Token-level stop sequences
                        if stop_token_seqs:
                            hit = False
                            for seq in stop_token_seqs:
                                n = len(seq)
                                if n > 0 and len(tokens) >= n and tokens[-n:] == seq:
                                    tokens = tokens[:-n]
                                    eos_reached = True
                                    finish_reason = "stop_sequence"
                                    hit = True
                                    break
                            if hit:
                                break
                        # String fallback stops on generated suffix
                        if raw_stops:
                            if speculative_raw_stop_state is not None:
                                tokens, hit = _match_raw_stops_incremental(
                                    tk,
                                    tokens,
                                    prompt_len,
                                    raw_stops,
                                    speculative_raw_stop_state,
                                    t,
                                )
                            else:
                                tokens, hit = _match_raw_stops_full_decode(tk, tokens, prompt_len, raw_stops)
                            if hit:
                                eos_reached = True
                                finish_reason = "stop_sequence"
                                break
                    if finish_reason is None and n_generated >= config.max_tokens:
                        finish_reason = "length"
                    return _build_response(tokens, eos_reached, finish_reason)

    # generation loop
    while n_generated < config.max_tokens:
        # Patch residual injection for this step if present
        if injector and residual_hooks:
            try:
                injector.patch(residual_hooks, n_generated, batch=1, seq_len=y.shape[1], model=model)
            except Exception:
                injector.restore()
                injector = None  # fallback to no injection

        logits, prompt_cache = _step_logits(model, components, y, cache=prompt_cache)

        # Restore after step
        if injector and residual_hooks:
            injector.restore()

        logprobs = stable_log_softmax(logits)

        # Apply processors (HF-compatible order)
        for proc in processors:
            logits = proc(tokens, logits)
        logprobs = stable_log_softmax(logits)

        if forced_bos is not None and n_generated == 0:
            next_tok = as_mx_array([forced_bos], dtype=mx.int32)
        elif n_generated in forced_map:
            next_tok = as_mx_array([forced_map[n_generated]], dtype=mx.int32)
        else:
            next_tok = sampler(logprobs)
        mx.eval(next_tok)
        t = int(next_tok.item())
        tokens.append(t)
        n_generated += 1
        y = as_mx_array(tokens, dtype=mx.int32)[None]

        if active_stream is not None:
            if not active_stream.emit_token(t):
                invalid_stop = True
                eos_reached = True
                finish_reason = finish_reason or "invalid_path"
                break

        # Forced EOS token id
        if config.forced_eos_token_id is not None and t == config.forced_eos_token_id:
            eos_reached = True
            finish_reason = "eos"
            break

        # EOS by any eos id
        if eos_ids and t in eos_ids:
            eos_reached = True
            finish_reason = "eos"
            break

        # Token-level stop sequences
        if stop_token_seqs:
            for seq in stop_token_seqs:
                n = len(seq)
                if n > 0 and len(tokens) >= n and tokens[-n:] == seq:
                    # Trim the stop sequence from tokens
                    tokens = tokens[:-n]
                    eos_reached = True
                    finish_reason = "stop_sequence"
                    break
            if eos_reached:
                break
        # String-level fallback stop sequences
        if raw_stops:
            if main_raw_stop_state is not None:
                tokens, hit = _match_raw_stops_incremental(
                    tk,
                    tokens,
                    prompt_len,
                    raw_stops,
                    main_raw_stop_state,
                    t,
                )
            else:
                tokens, hit = _match_raw_stops_full_decode(tk, tokens, prompt_len, raw_stops)
            if hit:
                eos_reached = True
                finish_reason = "stop_sequence"
                break

    if not finish_reason and n_generated >= config.max_tokens:
        finish_reason = "length"
    if invalid_stop and finish_reason is None:
        finish_reason = "invalid_path"
    return _build_response(tokens, eos_reached, finish_reason)


def _apply_forced_token_mask(logits, forced_token_id: Optional[int]):
    if forced_token_id is None:
        return logits
    mx, _ = try_import_mlx()
    V = logits.shape[-1]
    mask = -mx.inf * mx.ones_like(logits)
    mask[..., forced_token_id] = 0.0
    return logits + mask


def _beam_search_generate(setup: GenerationSetup) -> Dict[str, Any]:
    model = setup.model
    tokenizer = setup.tokenizer
    config = setup.config
    components = setup.components
    prompt_ids = setup.prompt_tokens
    processors = setup.processors
    eos_ids = setup.eos_ids
    stop_token_seqs = setup.stop_token_sequences
    residual_hooks = setup.residual_hooks
    forced_map = setup.forced_map
    forced_bos = setup.forced_bos
    raw_stops = setup.raw_stop_strings
    tk = setup.tk

    mx, _ = try_import_mlx()
    num_beams = int(config.num_beams)
    max_new = int(config.max_tokens)
    length_penalty = float(config.length_penalty or 0.0)
    # beams: list of (tokens, score, finished, cache, primed)
    # `primed` means cache already includes the full `tokens` prefix and
    # subsequent steps can consume only the most recent token.
    beams: List[Tuple[List[int], float, bool, Any, bool]] = [(list(prompt_ids), 0.0, False, None, False)]
    finished: List[Tuple[List[int], float]] = []
    n_generated = 0
    use_cached_steps = True

    # Optional residual injection for beam path
    beam_injector = (
        ResidualInjector(components.layers, components.hidden_size)
        if residual_hooks
        else None
    )

    while n_generated < max_new:
        # Prepare alive beams
        alive = [(t, s, c, p) for (t, s, f, c, p) in beams if not f]
        if not alive:
            break

        # Per-beam eval state: (tokens, score, cache_for_step, primed, logits)
        eval_state: List[Tuple[List[int], float, Any, bool, Any]] = []
        if use_cached_steps:
            try:
                for tokens, score, beam_cache, primed in alive:
                    if primed and tokens:
                        step_input = as_mx_array([tokens[-1]], dtype=mx.int32)[None]
                        step_cache = beam_cache
                    else:
                        step_input = as_mx_array(tokens, dtype=mx.int32)[None]
                        step_cache = None
                    if beam_injector and residual_hooks:
                        try:
                            beam_injector.patch(
                                residual_hooks,
                                n_generated,
                                batch=1,
                                seq_len=step_input.shape[1],
                                model=model,
                            )
                            logits, next_cache = _step_logits(model, components, step_input, cache=step_cache)
                        finally:
                            beam_injector.restore()
                    else:
                        logits, next_cache = _step_logits(model, components, step_input, cache=step_cache)
                    eval_state.append((tokens, score, next_cache, True, logits))
            except Exception:
                # Defensive fallback for models without usable cache semantics.
                use_cached_steps = False
                eval_state = []

        if not eval_state:
            seqs = [t for (t, _s, _c, _p) in alive]
            # Fallback batch path (full-sequence logits each step).
            if beam_injector and residual_hooks:
                try:
                    beam_injector.patch(residual_hooks, n_generated, batch=len(seqs), seq_len=len(seqs[0]), model=model)
                    logits_batch = _batched_last_logits(model, components, seqs)
                finally:
                    beam_injector.restore()
            else:
                logits_batch = _batched_last_logits(model, components, seqs)
            for i, (tokens, score, _beam_cache, _primed) in enumerate(alive):
                eval_state.append((tokens, score, None, False, logits_batch[i : i + 1, :]))

        # For each beam, apply processors and constraints
        candidates: List[Tuple[float, int, int]] = []  # (new_score, beam_index, token)
        for i, (tokens, score, next_cache, next_primed, logits) in enumerate(eval_state):
            for proc in processors:
                logits = proc(tokens, logits)
            # Forced tokens
            pos = n_generated
            forced_tok = None
            if forced_bos is not None and n_generated == 0:
                forced_tok = forced_bos
            elif pos in forced_map:
                forced_tok = forced_map[pos]
            if forced_tok is not None:
                logits = _apply_forced_token_mask(logits, forced_tok)
            logprobs = stable_log_softmax(logits)
            # Select top-k for each beam to limit combinatorics
            k = num_beams
            # argsort ascending, take last k for top values
            idxs = mx.argsort(logprobs, axis=-1)[:, -k:]
            vals = mx.take_along_axis(logprobs, idxs, axis=-1)
            vals_list = [float(v.item()) for v in vals.reshape((-1,))]
            idxs_list = idxs.reshape((-1,)).tolist()
            for v, tok in zip(vals_list, idxs_list):
                candidates.append((score + v, i, int(tok)))

        # Select overall top beams
        candidates.sort(key=lambda x: x[0], reverse=True)
        new_beams: List[Tuple[List[int], float, bool, Any, bool]] = []
        for new_score, i_beam, tok in candidates:
            if len(new_beams) >= num_beams:
                break
            base_tokens, _base_score, base_cache, base_primed, _base_logits = eval_state[i_beam]
            new_tokens = base_tokens + [tok]
            # Finish checks: eos ids or stop sequences
            is_finish = False
            trimmed_tokens = new_tokens
            if (eos_ids and tok in eos_ids) or (
                config.forced_eos_token_id is not None and tok == config.forced_eos_token_id
            ):
                is_finish = True
            if not is_finish and stop_token_seqs:
                for seq in stop_token_seqs:
                    n = len(seq)
                    if n > 0 and len(new_tokens) >= n and new_tokens[-n:] == seq:
                        trimmed_tokens = new_tokens[:-n]
                        is_finish = True
                        break
            if not is_finish and raw_stops:
                generated = new_tokens[len(prompt_ids) :]
                if generated:
                    decoded = tk.decode(generated)
                    for s in raw_stops:
                        if s and s in decoded:
                            idx = decoded.find(s)
                            text_trimmed = decoded[:idx]
                            trimmed_ids = tk.encode(text_trimmed, add_special_tokens=False)
                            trimmed_tokens = new_tokens[: len(prompt_ids)] + trimmed_ids
                            is_finish = True
                            break
            new_beams.append(
                (
                    trimmed_tokens if is_finish else new_tokens,
                    new_score,
                    is_finish,
                    base_cache,
                    base_primed,
                )
            )

        beams = new_beams
        # Move finished beams out, keep up to num_beams alive
        alive_next: List[Tuple[List[int], float, bool, Any, bool]] = []
        for t, s, f, c, p in beams:
            if f:
                length = max(1, len(t) - len(prompt_ids))
                norm = (length ** (length_penalty)) if length_penalty != 0.0 else 1.0
                finished.append((t, s / norm))
            else:
                alive_next.append((t, s, f, c, p))
        # Keep best alive beams
        alive_next.sort(key=lambda x: x[1], reverse=True)
        beams = alive_next[: num_beams]

        n_generated += 1
        # Early stop if enough finished
        if config.early_stopping and len(finished) >= num_beams:
            break

    # If no finished, take best alive
    if not finished:
        finished = [(t, s) for (t, s, _f, _c, _p) in beams]
        # Normalize scores
        finished = [
            (t, (s / (max(1, len(t) - len(prompt_ids)) ** length_penalty)) if length_penalty != 0.0 else s)
            for (t, s) in finished
        ]
    finished.sort(key=lambda x: x[1], reverse=True)
    best_tokens = finished[0][0]
    text_out = _decode_generated_suffix(tk, best_tokens, len(prompt_ids))
    if best_tokens and eos_ids and (best_tokens[-1] in eos_ids):
        finish_reason = "eos"
    elif n_generated >= max_new:
        finish_reason = "length"
    else:
        finish_reason = "stop_sequence"
    return {
        "text": text_out,
        "tokens": best_tokens,
        "eos_reached": True,
        "finish_reason": finish_reason,
    }


class MlxGenerateBackend:
    """Backend that runs generation using MLX models."""

    name = "mlx"

    def __init__(self, *, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: Any,
        config: GenerationConfig,
        *,
        hooks: Optional[Any] = None,
        stream_observer: Optional[Any] = None,
    ) -> GenerateResult:
        raw = generate_dict(
            self.model,
            self.tokenizer,
            prompt,
            config,
            hooks=hooks,
            stream_observer=stream_observer,
        )
        meta = {
            "backend": self.name,
            "finish_reason": raw.get("finish_reason"),
        }
        if "stream_tokens_emitted" in raw:
            meta.setdefault("stream_tokens_emitted", raw["stream_tokens_emitted"])
        if raw.get("stream_invalid_path"):
            meta.setdefault("stream_invalid_path", True)
        if raw.get("speculative_fallback_reason"):
            meta.setdefault("speculative_fallback_reason", raw["speculative_fallback_reason"])
        return GenerateResult(
            text=raw.get("text", ""),
            tokens=raw.get("tokens"),
            eos_reached=raw.get("eos_reached"),
            finish_reason=raw.get("finish_reason"),
            meta=meta,
        )
