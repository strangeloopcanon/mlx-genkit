from __future__ import annotations

from typing import Any, Optional, Protocol

from ..config import GenerationConfig
from ..structure.result import GenerateResult
from .mlx_backend import MlxGenerateBackend
from .transformers_backend import TransformersGenerateBackend
from .vllm_backend import VLLMGenerateBackend

_SUPPORTED_BACKEND_NAMES = ("default", "hf", "mlx", "transformers", "vllm")


class GenerateBackend(Protocol):
    """Protocol implemented by concrete generation backends."""

    name: str

    def generate(
        self,
        prompt: Any,
        config: GenerationConfig,
        *,
        hooks: Optional[Any] = None,
        stream_observer: Optional[Any] = None,
    ) -> GenerateResult:  # pragma: no cover - interface definition
        ...


def resolve_backend(
    backend_name: Optional[str],
    *,
    model: Any,
    tokenizer: Any,
) -> GenerateBackend:
    """Return a backend implementation based on the requested name."""

    name = (backend_name or "mlx").lower()
    if name in {"mlx", "default"}:
        return MlxGenerateBackend(model=model, tokenizer=tokenizer)
    if name in {"transformers", "hf"}:
        return TransformersGenerateBackend(model=model, tokenizer=tokenizer)
    if name == "vllm":
        return VLLMGenerateBackend(model=model, tokenizer=tokenizer)
    supported = ", ".join(_SUPPORTED_BACKEND_NAMES)
    raise ValueError(f"Unsupported backend '{backend_name}'. Supported backends: {supported}")
