from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..utils import try_import_mlx


@dataclass
class PPOConfig:
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    lr: float = 2e-4
    grad_clip: float = 1.0


@dataclass
class GRPOConfig:
    clip_ratio: float = 0.2
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    normalize_advantages: bool = True
    advantage_eps: float = 1e-6


@dataclass
class GSPOConfig:
    clip_ratio: float = 0.2
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    normalize_advantages: bool = True
    advantage_eps: float = 1e-6


def _masked_mean(x, mask=None, eps: float = 1e-8):
    mx, _ = try_import_mlx()
    if mask is None:
        return x.mean()
    m = mask.astype(x.dtype)
    denom = mx.maximum(m.sum(), eps)
    return (x * m).sum() / denom


def _normalize_advantages(adv, mask=None, eps: float = 1e-6):
    mx, _ = try_import_mlx()
    mean = _masked_mean(adv, mask, eps=eps)
    centered = adv - mean
    var = _masked_mean(centered * centered, mask, eps=eps)
    std = mx.sqrt(mx.maximum(var, eps))
    return centered / std


def grpo_advantages(group_rewards, eps: float = 1e-6):
    """Per-group reward normalization used by GRPO.

    `group_rewards` is expected to be shape [G, K] where G is number of prompts
    and K is the number of sampled completions per prompt.
    """
    mx, _ = try_import_mlx()
    if not hasattr(group_rewards, "ndim") or int(group_rewards.ndim) != 2:
        raise ValueError("grpo_advantages expects group_rewards with shape [G, K]")
    mean = group_rewards.mean(axis=-1, keepdims=True)
    centered = group_rewards - mean
    var = (centered * centered).mean(axis=-1, keepdims=True)
    std = mx.sqrt(mx.maximum(var, eps))
    return centered / std


def ppo_loss(
    *,
    logp_new,  # [N]
    logp_old,  # [N]
    advantage,  # [N]
    value_pred,  # [N]
    value_target,  # [N]
    cfg: PPOConfig,
):
    mx, _ = try_import_mlx()
    ratio = mx.exp(logp_new - logp_old)
    unclipped = ratio * advantage
    clipped = mx.clip(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantage
    policy_loss = -mx.minimum(unclipped, clipped).mean()
    value_loss = ((value_pred - value_target) ** 2).mean()
    # logp_new is for sampled actions; use Monte Carlo entropy estimator.
    entropy = -logp_new.mean()
    total = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
    return total, {"policy": policy_loss, "value": value_loss, "entropy": entropy}


def grpo_loss(
    *,
    logp_new,  # [G, K] or [N] completion-level log-prob
    logp_old,  # same shape as logp_new
    cfg: GRPOConfig,
    rewards=None,  # optional [G, K], used to derive advantages
    advantages=None,  # optional explicit advantages (same shape as logp_new)
    logp_ref=None,  # optional reference log-prob (same shape)
):
    """GRPO-style clipped policy loss over grouped completion samples."""
    mx, _ = try_import_mlx()
    if advantages is None:
        if rewards is None:
            raise ValueError("grpo_loss requires either `advantages` or `rewards`.")
        advantages = grpo_advantages(rewards, eps=cfg.advantage_eps)
    if cfg.normalize_advantages:
        advantages = _normalize_advantages(advantages, mask=None, eps=cfg.advantage_eps)

    ratio = mx.exp(logp_new - logp_old)
    unclipped = ratio * advantages
    clipped = mx.clip(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages
    policy_loss = -mx.minimum(unclipped, clipped).mean()
    entropy = -logp_new.mean()

    total = policy_loss - cfg.entropy_coef * entropy
    kl = None
    if logp_ref is not None:
        # Monte Carlo KL estimate on sampled actions.
        kl = (logp_new - logp_ref).mean()
        total = total + cfg.kl_coef * kl
    return total, {"policy": policy_loss, "entropy": entropy, "kl": kl}


def gspo_loss(
    *,
    logp_new,  # [B, T] token log-probs on sampled tokens
    logp_old,  # [B, T]
    advantages,  # [B, T] or [B]
    cfg: GSPOConfig,
    mask: Optional[Any] = None,  # [B, T] bool/int (1 keeps token)
    logp_ref=None,  # optional [B, T]
):
    """GSPO-style token-level clipped policy loss with optional masking."""
    if hasattr(advantages, "ndim") and hasattr(logp_new, "ndim"):
        if int(advantages.ndim) + 1 == int(logp_new.ndim):
            advantages = advantages.reshape((*advantages.shape, 1))
    if cfg.normalize_advantages:
        advantages = _normalize_advantages(advantages, mask=mask, eps=cfg.advantage_eps)

    mx, _ = try_import_mlx()
    ratio = mx.exp(logp_new - logp_old)
    unclipped = ratio * advantages
    clipped = mx.clip(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages
    policy_loss = -_masked_mean(mx.minimum(unclipped, clipped), mask)
    entropy = -_masked_mean(logp_new, mask)

    total = policy_loss - cfg.entropy_coef * entropy
    kl = None
    if logp_ref is not None:
        # Monte Carlo KL estimate on sampled tokens.
        kl = _masked_mean(logp_new - logp_ref, mask)
        total = total + cfg.kl_coef * kl
    return total, {"policy": policy_loss, "entropy": entropy, "kl": kl}
