from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


def try_import_mlx():
    try:
        import mlx.core as mx  # type: ignore
        import mlx.nn as nn  # type: ignore

        return mx, nn
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "MLX is required. Install with `pip install mlx` (and mlx-lm if you want cache acceleration)."
        ) from e


def try_import_mlx_lm_cache():
    try:
        from mlx_lm.models import cache as lm_cache  # type: ignore

        return lm_cache
    except Exception:
        return None


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    mx, _ = try_import_mlx()
    mx.random.seed(seed)


def to_cpu(x):
    # Convenience to get a numpy array without pulling in numpy explicitly
    try:
        return x.squeeze().tolist()  # type: ignore[attr-defined]
    except Exception:
        return x


@dataclass
class VocabProjection:
    """Projection from hidden to logits.

    - If `linear` is set, calls it directly (works with quantized/tied embeddings via `as_linear`).
    - Otherwise uses `weight` with matmul, optionally transposed.
    """

    weight: Any = None  # mx.array or None
    transpose: bool = False
    linear: Any = None  # callable or nn.Module providing __call__(hidden)->logits

    def project(self, hidden) -> Any:
        if self.linear is not None:
            return self.linear(hidden)
        mx, _ = try_import_mlx()
        w = self.weight
        if w is None:
            raise ValueError("VocabProjection has no weight or linear callable")
        return hidden @ (w.T if self.transpose else w)


def stable_log_softmax(logits):
    mx, _ = try_import_mlx()
    logits32 = logits.astype(mx.float32)
    return logits32 - mx.logsumexp(logits32, axis=-1, keepdims=True)


def stable_softmax(logits, temperature: float = 1.0):
    """Numerically stable softmax with optional temperature.

    Applies scaling by 1/temperature before softmax; returns probabilities.
    """
    mx, _ = try_import_mlx()
    if temperature and temperature != 1.0:
        logits = logits * (1.0 / float(temperature))
    return mx.exp(stable_log_softmax(logits))


def as_mx_array(data, dtype=None):
    mx, _ = try_import_mlx()
    arr = mx.array(data)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def ensure_1d(arr):
    return arr.reshape((-1,))


def concat_tokens(a, b):
    mx, _ = try_import_mlx()
    if a is None:
        return b
    return mx.concat([a, b], axis=-1)


def slice_last(x, n: int):
    return x[..., -n:]


def detect_device_dtype(model) -> Tuple[str, Any]:
    mx, _ = try_import_mlx()
    # Walk the tree to find the first array to infer dtype and device
    device = mx.default_device()
    dtype = mx.float16
    try:
        # model.parameters() may be a generator of arrays
        params = []
        for k, v in model.state_dict().items():  # type: ignore[attr-defined]
            if hasattr(v, "dtype"):
                dtype = v.dtype
                break
        return str(device), dtype
    except Exception:
        return str(device), dtype


# ------------------ Model helpers ------------------


def ema_update(dst_model: Any, src_model: Any, decay: float = 0.999) -> None:
    """Exponential moving average update of dst_model params from src_model.

    Uses trainable parameters tree for both models; updates dst in-place.
    """
    mx, nn = try_import_mlx()
    d = dst_model.trainable_parameters()
    s = src_model.trainable_parameters()

    tm = getattr(mx, "tree_map", None) or getattr(nn, "tree_map", None)

    if tm is None:
        def _tree_map(fn, x, y):
            if isinstance(x, dict) and isinstance(y, dict):
                return {k: _tree_map(fn, x[k], y[k]) for k in x.keys()}
            if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
                out = [_tree_map(fn, xi, yi) for xi, yi in zip(x, y)]
                return type(x)(out)
            return fn(x, y)
        tm = _tree_map
    new = tm(lambda a, b: a * decay + b * (1.0 - decay), d, s)
    dst_model.update(new)


def build_action_mask(prompt_lens: Any, seq_len: int):
    """Build a boolean mask selecting positions >= prompt_len.

    - If prompt_lens is int: returns [T] bool mask
    - If prompt_lens is a list/array [B]: returns [B, T] mask
    """
    mx, _ = try_import_mlx()
    T = int(seq_len)
    if isinstance(prompt_lens, int):
        t = mx.arange(T)
        return t >= int(prompt_lens)
    # Assume 1D array-like of per-sample prompt lengths
    pl = mx.array(prompt_lens).reshape((-1, 1))
    t = mx.arange(T).reshape((1, -1))
    return t >= pl


def clone_reference(model: Any) -> Any:
    """Best-effort deep copy of an MLX model for use as a frozen reference.

    If deepcopy fails for a particular model class, prefer reloading via
    `mlx_lm.load()` from the same weights to obtain an independent reference.
    """
    try:
        import copy

        return copy.deepcopy(model)
    except Exception as e:  # pragma: no cover - environment/model specific
        raise RuntimeError(
            "clone_reference: deepcopy failed for this model. Consider reloading the model via `mlx_lm.load(...)` to create an independent reference."
        ) from e


def filter_kwargs_for_callable(fn: Any, kwargs: dict) -> dict:
    """Return only kwargs accepted by `fn`'s signature.

    Used to keep compatibility with evolving upstream signatures (e.g. mlx-lm).
    If signature introspection fails, returns `kwargs` unchanged.
    """

    try:
        from inspect import signature

        sig = signature(fn)
    except Exception:  # pragma: no cover - defensive
        return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}
