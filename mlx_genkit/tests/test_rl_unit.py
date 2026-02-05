from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import mlx_genkit.rl.ppo as ppo_mod


class _FakeMx:
    @staticmethod
    def exp(x):  # noqa: ANN001
        return np.exp(x)

    @staticmethod
    def clip(x, lo, hi):  # noqa: ANN001
        return np.clip(x, lo, hi)

    @staticmethod
    def minimum(a, b):  # noqa: ANN001
        return np.minimum(a, b)

    @staticmethod
    def maximum(a, b):  # noqa: ANN001
        return np.maximum(a, b)

    @staticmethod
    def sqrt(x):  # noqa: ANN001
        return np.sqrt(x)


class RLObjectiveTests(unittest.TestCase):
    def test_grpo_advantages_normalizes_per_group(self):
        rewards = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=float,
        )
        with patch.object(ppo_mod, "try_import_mlx", return_value=(_FakeMx(), None)):
            adv = ppo_mod.grpo_advantages(rewards)
        np.testing.assert_allclose(adv.mean(axis=1), np.array([0.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(adv.std(axis=1), np.array([1.0, 1.0]), atol=1e-6)

    def test_grpo_loss_supports_reward_derived_advantages_and_kl(self):
        rewards = np.array([[1.0, 2.0]], dtype=float)
        logp_new = np.array([[-0.6, -0.2]], dtype=float)
        logp_old = np.array([[-0.7, -0.3]], dtype=float)
        logp_ref = np.array([[-0.8, -0.4]], dtype=float)
        cfg = ppo_mod.GRPOConfig(kl_coef=0.5, entropy_coef=0.1)
        with patch.object(ppo_mod, "try_import_mlx", return_value=(_FakeMx(), None)):
            total, stats = ppo_mod.grpo_loss(
                logp_new=logp_new,
                logp_old=logp_old,
                rewards=rewards,
                cfg=cfg,
                logp_ref=logp_ref,
            )
        self.assertIn("policy", stats)
        self.assertIn("entropy", stats)
        self.assertIn("kl", stats)
        self.assertIsNotNone(stats["kl"])
        # KL estimate is mean(logp_new - logp_ref)
        self.assertAlmostEqual(float(stats["kl"]), float((logp_new - logp_ref).mean()), places=6)
        self.assertTrue(np.isfinite(float(total)))

    def test_gspo_loss_uses_mask_and_supports_broadcast_advantages(self):
        logp_new = np.array([[-1.0, -0.5, -0.2]], dtype=float)
        logp_old = np.array([[-1.1, -0.4, -0.3]], dtype=float)
        logp_ref = np.array([[-1.3, -0.6, -0.5]], dtype=float)
        advantages = np.array([1.0], dtype=float)  # [B], broadcast to [B, 1]
        mask = np.array([[1, 1, 0]], dtype=float)
        cfg = ppo_mod.GSPOConfig(kl_coef=1.0, entropy_coef=0.0, normalize_advantages=False)
        with patch.object(ppo_mod, "try_import_mlx", return_value=(_FakeMx(), None)):
            total, stats = ppo_mod.gspo_loss(
                logp_new=logp_new,
                logp_old=logp_old,
                advantages=advantages,
                cfg=cfg,
                mask=mask,
                logp_ref=logp_ref,
            )
        self.assertIn("policy", stats)
        self.assertIn("entropy", stats)
        self.assertIn("kl", stats)
        expected_kl = ((logp_new - logp_ref) * mask).sum() / mask.sum()
        self.assertAlmostEqual(float(stats["kl"]), float(expected_kl), places=6)
        self.assertTrue(np.isfinite(float(total)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
