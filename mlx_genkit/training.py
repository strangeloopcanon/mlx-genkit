from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .adapters import make_tokenizer_bridge, detect_components, project_logits, ModelComponents
from .injection import ResidualInjectionHook, LogitBiasHook, SoftPromptHook
from .utils import try_import_mlx, stable_log_softmax, as_mx_array


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    grad_accum: int = 1
    microbatch: int = 0  # 0 disables
    dtype: Optional[str] = None  # 'bf16' for mixed precision compute
    label_smoothing: float = 0.0
    loss_scale: float = 1.0  # for mixed precision


def _ensure_soft_prompt(model, components: ModelComponents, hook: SoftPromptHook):
    mx, _ = try_import_mlx()
    key = hook.param_key
    if key in model:
        return model[key]
    # Initialize new soft prompt
    H = components.hidden_size
    S = hook.n_virtual
    if hook.init == "embed" and components.embed is not None:
        # Sample rows from embedding or average
        # Use mean of embedding rows as simple init
        W = components.embed.weight
        mean = W.mean(axis=0)
        sp = mx.broadcast_to(mean.reshape((1, -1)), (S, H))
        sp = sp + mx.random.normal(sp.shape) * (1.0 / (H**0.5))
    else:
        sp = mx.random.normal((S, H)) * (1.0 / (H**0.5))
    model[key] = sp
    return model[key]


def loss_forward(
    model: Any,
    tok_batch: Any,  # mx.array [B, L] int32
    mask: Optional[Any] = None,
    hooks: Optional[List[Any]] = None,
    soft_prompt: Optional[Any] = None,  # mx.array [S, H] trainable (overrides SoftPromptHook)
) -> Any:
    """Forward pass that returns logits [B, L, V] for loss computation.

    - Applies differentiable hooks (logit bias; residual patch kept simple).
    - Supports prepending a soft prompt via input_embeddings; slices it off
      before loss to align with labels.
    - Keeps logits math in fp32 for stability.
    """
    mx, nn = try_import_mlx()
    components = detect_components(model)
    B, L = tok_batch.shape

    # Build optional input embeddings with soft prompt (explicit or via hook)
    input_embeddings = None
    sp_used = None
    if hooks:
        for h in hooks:
            if isinstance(h, SoftPromptHook):
                sp_used = _ensure_soft_prompt(model, components, h)
                break
    if soft_prompt is not None:
        sp_used = soft_prompt

    if sp_used is not None:
        # Compute token embeddings for the batch and prepend soft prompt
        # Note: we bypass embedding lookup by providing input_embeddings directly
        tok_embeds = components.embed(tok_batch)
        sp = sp_used.reshape((1, sp_used.shape[0], -1))
        sp = mx.broadcast_to(sp, (B, sp.shape[1], sp.shape[2]))
        input_embeddings = mx.concat([sp, tok_embeds], axis=1)
        dummy_tokens = mx.zeros((B, input_embeddings.shape[1]), dtype=mx.int32)
        try:
            logits = model(dummy_tokens, input_embeddings=input_embeddings)
            # Align to original token positions: drop the virtual soft prompt positions
            S = sp_used.shape[0]
            logits = logits[:, S:, :]
        except TypeError:
            # Fallback: model does not support input_embeddings kwarg
            logits = model(tok_batch)
    else:
        logits = model(tok_batch)

    # If model returns hidden states, project to logits
    if logits.shape[-1] == components.hidden_size and components.vocab_size != components.hidden_size:
        logits = project_logits(components, logits)

    # Residual injection during training: we can patch layers to add vec
    if hooks:
        for h in hooks:
            if isinstance(h, LogitBiasHook):
                vec = h.resolve_vector(model)
                proj = components.vocab_projection
                if proj.linear is not None:
                    # Use as_linear to support quantized/tied embeddings
                    v3 = vec.reshape((1, 1, -1))
                    b = proj.linear(v3)  # [1,1,V]
                    logits = logits + b * h.alpha
                else:
                    W = proj.weight
                    if h.freeze_projection:
                        W = mx.stop_gradient(W)
                    bias = (vec.reshape((1, -1)) @ W.T).reshape((-1,)) * h.alpha
                    logits = logits + bias.reshape((1, 1, -1))

    logits = logits.astype(mx.float32)
    return logits


def xent_loss(
    logits: Any,  # [B, T, V]
    labels: Any,  # [B, T]
    *,
    ignore_index: int,
    label_smoothing: float = 0.0,
) -> Any:
    mx, _ = try_import_mlx()
    V = logits.shape[-1]
    logprobs = stable_log_softmax(logits)
    B, T = labels.shape
    labels = labels.astype(mx.int32)
    valid = labels != ignore_index
    labels_clamped = mx.maximum(labels, 0)
    gathered = mx.take_along_axis(logprobs, labels_clamped.reshape(B, T, 1), axis=-1).reshape(B, T)
    if label_smoothing and label_smoothing > 0.0:
        eps = float(label_smoothing)
        avg_logp = logprobs.mean(axis=-1)  # [B, T]
        nll = -(1.0 - eps) * gathered - eps * avg_logp
    else:
        nll = -gathered
    nll = mx.where(valid, nll, mx.zeros_like(nll))
    denom = mx.maximum(valid.sum(), 1)
    return nll.sum() / denom


def apply_lora(
    model: Any,
    rank: int,
    alpha: float,
    targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"),
) -> List[Any]:
    """Attach LoRA adapters to named Linear layers in the model.

    Returns a list of patched modules for reference.
    """
    mx, nn = try_import_mlx()

    # Tree map helpers (unary/binary) across param/grad pytrees
    tm_core = getattr(mx, "tree_map", None) or getattr(nn, "tree_map", None)

    def _tree_map_unary(fn, x):
        if tm_core is not None:
            return tm_core(fn, x)
        # Fallback recursive map
        if isinstance(x, dict):
            return {k: _tree_map_unary(fn, v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            out = [_tree_map_unary(fn, v) for v in x]
            return type(x)(out)
        return fn(x)

    def _tree_map_binary(fn, x, y):
        if tm_core is not None:
            return tm_core(lambda a, b: fn(a, b), x, y)
        if isinstance(x, dict) and isinstance(y, dict):
            return {k: _tree_map_binary(fn, x[k], y[k]) for k in x.keys()}
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            out = [_tree_map_binary(fn, xi, yi) for xi, yi in zip(x, y)]
            return type(x)(out)
        return fn(x, y)
    patched = []

    def wants(name: str):
        return any(t in name for t in targets)

    class LoRALinear:
        def __init__(self, linear: nn.Linear, rank: int, alpha: float):
            self.linear = linear
            in_dim = linear.weight.shape[1]
            out_dim = linear.weight.shape[0]
            scale = alpha / float(rank)
            self.A = mx.random.normal((in_dim, rank)) * (1.0 / (in_dim**0.5))
            self.B = mx.zeros((rank, out_dim))
            self.scale = scale
            self._orig = linear.__call__

        def __call__(self, x):
            base = self._orig(x)
            # x: [*, in] ; A: [in, r] ; B: [r, out]
            update = (x @ self.A) @ self.B
            return base + update * self.scale

        def apply(self):
            # Register parameters in the module so they are trainable
            self.linear["_lora_A"] = self.A
            self.linear["_lora_B"] = self.B
            self.linear["_lora_scale"] = mx.array(self.scale)
            self.linear.__call__ = self.__call__  # type: ignore

        def restore(self):
            self.linear.__call__ = self._orig  # type: ignore
            if "_lora_A" in self.linear:
                del self.linear["_lora_A"]
                del self.linear["_lora_B"]
                del self.linear["_lora_scale"]

        def merge(self):
            # W := W + B @ A^T * scale
            delta = (self.B @ self.A.T) * self.scale
            self.linear.weight = self.linear.weight + delta

    for parent_name, module in model.named_modules():
        for name, child in module.items():
            full = f"{parent_name}.{name}" if parent_name else name
            if wants(full) and hasattr(child, "weight") and hasattr(child, "__call__"):
                try:
                    lr = LoRALinear(child, rank, alpha)
                    lr.apply()
                    patched.append(lr)
                except Exception:
                    pass
    return patched


def merge_lora(model: Any) -> None:
    """Merge any applied LoRA adapters into base weights and restore forward.

    `apply_lora` patches modules by replacing their `__call__` with a bound
    wrapper method; we detect that wrapper here and invoke its merge/restore.
    """

    for _, module in model.named_modules():
        for _, child in module.items():
            call = getattr(child, "__call__", None)
            wrapper = getattr(call, "__self__", None)
            if wrapper is None:
                continue

            linear = getattr(wrapper, "linear", None)
            merge = getattr(wrapper, "merge", None)
            restore = getattr(wrapper, "restore", None)
            if linear is child and callable(merge) and callable(restore):
                try:
                    merge()
                    restore()
                except Exception:
                    pass


def unmerge_lora(model: Any) -> None:
    # No-op placeholder; in this minimal shim we do not track deltas separately post-merge
    return None


def train_step(
    model: Any,
    batch: Dict[str, Any],
    optimizer: Any,
    cfg: TrainingConfig,
    hooks: Optional[List[Any]] = None,
    pad_id: Optional[int] = None,
):
    """Single training step with optional microbatching and grad accumulation.

    batch expects keys: 'tokens' (int32 [B, L])
    """
    mx, nn = try_import_mlx()

    # Tree map helpers (unary/binary) across param/grad pytrees
    tm_core = getattr(mx, "tree_map", None) or getattr(nn, "tree_map", None)

    def _tree_map_unary(fn, x):
        if tm_core is not None:
            return tm_core(fn, x)
        if isinstance(x, dict):
            return {k: _tree_map_unary(fn, v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            out = [_tree_map_unary(fn, v) for v in x]
            return type(x)(out)
        return fn(x)

    def _tree_map_binary(fn, x, y):
        if tm_core is not None:
            return tm_core(lambda a, b: fn(a, b), x, y)
        if isinstance(x, dict) and isinstance(y, dict):
            return {k: _tree_map_binary(fn, x[k], y[k]) for k in x.keys()}
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            out = [_tree_map_binary(fn, xi, yi) for xi, yi in zip(x, y)]
            return type(x)(out)
        return fn(x, y)

    tokens = batch["tokens"]
    B, L = tokens.shape
    ignore_index = pad_id if pad_id is not None else -100

    def loss_fn(tokens_slice):
        return xent_loss(
            loss_forward(model, tokens_slice, hooks=hooks)[:, :-1, :],
            tokens_slice[:, 1:],
            ignore_index=ignore_index,
            label_smoothing=cfg.label_smoothing,
        )

    # Microbatch split
    if cfg.microbatch and cfg.microbatch > 0 and B > cfg.microbatch:
        splits = list(range(0, B, cfg.microbatch))
    else:
        splits = [0]

    total_loss = 0.0
    total_steps = 0
    grads_accum = None

    def closure():
        nonlocal total_loss, total_steps, grads_accum
        for i in splits:
            sl = tokens[i : i + (cfg.microbatch or B)]
            # Compute value and grads wrt model trainable params
            # Mixed-precision compute path: cast params to bf16 for compute only
            if cfg.dtype == "bf16":
                fp32_params = model.trainable_parameters()
                bf16_params = _tree_map_unary(lambda p: p.astype(mx.bfloat16), fp32_params)
                model.update(bf16_params)
                val_and_grad = nn.value_and_grad(model, lambda: loss_fn(sl) * cfg.loss_scale)
                loss, grads = val_and_grad()
                # Back to fp32 params for optimizer update later
                model.update(fp32_params)
                grads = _tree_map_unary(lambda g: g.astype(mx.float32) / cfg.loss_scale, grads)
                loss = loss / cfg.loss_scale
            else:
                val_and_grad = nn.value_and_grad(model, lambda: loss_fn(sl))
                loss, grads = val_and_grad()
            total_loss = total_loss + float(loss.item())
            total_steps += 1
            if grads_accum is None:
                grads_accum = grads
            else:
                grads_accum = _tree_map_binary(lambda a, b: a + b, grads_accum, grads)
        # Average grads over microbatches
        if total_steps > 1:
            grads_accum = _tree_map_unary(lambda g: g / total_steps, grads_accum)

    closure()

    # Clip gradients
    from mlx.optimizers import clip_grad_norm

    clipped, _ = clip_grad_norm(grads_accum, cfg.grad_clip)
    optimizer.update(model, clipped)

    return total_loss / max(total_steps, 1)


# ------------------ Training utilities (public API) ------------------


def sequence_logprob(
    model: Any,
    tokens: Any,  # mx.array [B, T] int32
    labels: Any,  # mx.array [B, T] int32, -100 to ignore
    *,
    hooks: Optional[List[Any]] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    """Per-sample sequence log-probability over supervised positions.

    Returns [B] with mean (or sum) of token log-probs where labels != ignore_index.
    """
    mx, _ = try_import_mlx()
    logits = loss_forward(model, tokens, hooks=hooks)  # [B, T, V]
    logprobs = stable_log_softmax(logits)
    B, T = labels.shape
    labels_i = labels.astype(mx.int32)
    valid = labels_i != ignore_index
    labels_clamped = mx.maximum(labels_i, 0)
    tok_lp = mx.take_along_axis(logprobs, labels_clamped.reshape(B, T, 1), axis=-1).reshape(B, T)
    tok_lp = mx.where(valid, tok_lp, mx.zeros_like(tok_lp))
    counts = valid.sum(axis=-1).astype(tok_lp.dtype)
    counts = mx.maximum(counts, mx.array(1.0, dtype=counts.dtype))
    if reduction == "sum":
        return tok_lp.sum(axis=-1)
    return tok_lp.sum(axis=-1) / counts


def token_kl(
    model: Any,
    ref_model: Any,
    tokens: Any,  # mx.array [B, T]
    labels: Any,  # mx.array [B, T] (mask via ignore_index)
    *,
    hooks: Optional[List[Any]] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
):
    """Per-sample token-level KL(pi || p_ref) averaged over supervised positions.

    Computes KL on normalized log-probs over vocabulary at each position, masks
    to labels != ignore_index, and reduces per sample.
    """
    mx, _ = try_import_mlx()
    logp = stable_log_softmax(loss_forward(model, tokens, hooks=hooks))
    logq = stable_log_softmax(loss_forward(ref_model, tokens, hooks=None))
    p = mx.exp(logp)
    kl_tok = (p * (logp - logq)).sum(axis=-1)  # [B, T]
    valid = (labels.astype(mx.int32) != ignore_index)
    kl_tok = mx.where(valid, kl_tok, mx.zeros_like(kl_tok))
    counts = valid.sum(axis=-1).astype(kl_tok.dtype)
    counts = mx.maximum(counts, mx.array(1.0, dtype=counts.dtype))
    if reduction == "sum":
        return kl_tok.sum(axis=-1)
    return kl_tok.sum(axis=-1) / counts
