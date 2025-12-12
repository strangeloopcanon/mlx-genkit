__all__ = [
    "GenerationConfig",
    "generate",
    "ResidualInjectionHook",
    "LogitBiasHook",
    "SoftPromptHook",
    "forward_with_hidden",
    "detect_components",
    # Training
    "TrainingConfig",
    "loss_forward",
    "xent_loss",
    "apply_lora",
    "merge_lora",
    "unmerge_lora",
    "train_step",
    # Training utilities
    "sequence_logprob",
    "token_kl",
    # Model helpers
    "ema_update",
    "build_action_mask",
    "stable_softmax",
    "clone_reference",
    "GenerateResult",
    "JsonAdherence",
    "Grammar",
    "StructuredSpec",
    "generate_structured",
    "generate_stream",
    "StreamCallbacks",
    "JsonlLogWriter",
    "BatchResult",
    "generate_many",
    "MustContain",
    "EnumIn",
    "RegexOnField",
]

from .api import (
    GenerationConfig,
    generate,
    forward_with_hidden,
    detect_components,
)
from .structure.result import GenerateResult
from .structure.adherence import JsonAdherence
from .structure.grammar import Grammar
from .structure.dsl import StructuredSpec, generate_structured
from .structure.stream import generate_stream, StreamCallbacks
from .structure.logs import JsonlLogWriter
from .structure.batch import BatchResult, generate_many
from .structure.semantic import MustContain, EnumIn, RegexOnField
from .injection import ResidualInjectionHook, LogitBiasHook, SoftPromptHook

from .training import (
    TrainingConfig,
    loss_forward,
    xent_loss,
    apply_lora,
    merge_lora,
    unmerge_lora,
    train_step,
    sequence_logprob,
    token_kl,
)

from .utils import ema_update, build_action_mask, stable_softmax, clone_reference

__version__ = "0.4.5"
