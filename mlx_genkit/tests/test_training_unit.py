from __future__ import annotations

import random
import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mlx_genkit.data.loader import PrefetchDataLoader
import mlx_genkit.data.loader as loader_mod
import mlx_genkit.rl.ppo as ppo_mod
import mlx_genkit.training as training_mod
from mlx_genkit.training import TrainingConfig, apply_lora, merge_lora, train_step


class _DummyLinear:
    def __init__(self) -> None:
        self.weight = np.zeros((3, 2), dtype=float)
        self._storage = {}

    def __call__(self, x):  # noqa: ANN001
        return np.asarray(x) @ self.weight.T

    def __setitem__(self, key, value):  # noqa: ANN001
        self._storage[key] = value

    def __getitem__(self, key):  # noqa: ANN001
        return self._storage[key]

    def __contains__(self, key):  # noqa: ANN001
        return key in self._storage

    def __delitem__(self, key):  # noqa: ANN001
        del self._storage[key]


class _DummyLoRAWrapper:
    def __init__(self, linear: _DummyLinear) -> None:
        self.linear = linear
        self._orig = linear.__call__
        self.merged = False
        self.restored = False

    def __call__(self, x):  # noqa: ANN001
        return self._orig(x)

    def merge(self) -> None:
        self.merged = True

    def restore(self) -> None:
        self.restored = True
        self.linear.__call__ = self._orig


class _DummyModel(dict):
    def named_modules(self):  # noqa: ANN001
        return [("", self)]


class _DummyFailingWrapper(_DummyLoRAWrapper):
    def merge(self) -> None:
        raise RuntimeError("boom")


class _FakeLoss:
    def __init__(self, value: float) -> None:
        self._value = float(value)

    def item(self):
        return self._value

    def __mul__(self, other):  # noqa: ANN001
        return _FakeLoss(self._value * float(other))

    __rmul__ = __mul__

    def __truediv__(self, other):  # noqa: ANN001
        return _FakeLoss(self._value / float(other))


def _tree_map(fn, *trees):  # noqa: ANN001
    if not trees:
        raise ValueError("missing trees")
    head = trees[0]
    if isinstance(head, dict):
        return {k: _tree_map(fn, *(t[k] for t in trees)) for k in head.keys()}
    if isinstance(head, (list, tuple)):
        vals = [_tree_map(fn, *(t[i] for t in trees)) for i in range(len(head))]
        return type(head)(vals)
    return fn(*trees)


class _FakeMx:
    int32 = "int32"
    float32 = "float32"
    bfloat16 = "bfloat16"
    random = types.SimpleNamespace(normal=lambda shape: np.ones(shape, dtype=float))

    @staticmethod
    def zeros(shape, dtype=None):  # noqa: ANN001
        return np.zeros(shape, dtype=float)

    @staticmethod
    def array(x):  # noqa: ANN001
        return np.asarray(x)

    @staticmethod
    def tree_map(fn, *trees):  # noqa: ANN001
        return _tree_map(fn, *trees)

    @staticmethod
    def exp(x):  # noqa: ANN001
        return np.exp(x)

    @staticmethod
    def clip(x, lo, hi):  # noqa: ANN001
        return np.clip(x, lo, hi)

    @staticmethod
    def minimum(a, b):  # noqa: ANN001
        return np.minimum(a, b)


class _FakeNn:
    def __init__(self, grads):
        self._grads = iter(grads)

    def value_and_grad(self, model, fn):  # noqa: ANN001
        def _inner():
            _ = fn()
            return _FakeLoss(1.0), next(self._grads)

        return _inner


class _TrainDummyModel:
    def trainable_parameters(self):
        return {"w": 0.0}

    def update(self, params):  # noqa: ANN001
        self._updated = params


class _TrainDummyOptimizer:
    def __init__(self) -> None:
        self.calls = []

    def update(self, model, grads):  # noqa: ANN001
        self.calls.append(grads)


class LoRAMergeTests(unittest.TestCase):
    def test_merge_lora_detects_wrapper_and_calls_merge_restore(self):
        linear = _DummyLinear()
        wrapper = _DummyLoRAWrapper(linear)
        linear.__dict__["__call__"] = wrapper.__call__

        model = _DummyModel(layer=linear)
        merge_lora(model)

        self.assertTrue(wrapper.merged)
        self.assertTrue(wrapper.restored)

    def test_merge_lora_warns_on_wrapper_failure(self):
        linear = _DummyLinear()
        wrapper = _DummyFailingWrapper(linear)
        linear.__dict__["__call__"] = wrapper.__call__

        model = _DummyModel(layer=linear)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            merge_lora(model)

        msgs = [str(w.message) for w in rec]
        self.assertTrue(any("merge_lora: failed to merge" in m for m in msgs))

    def test_merge_lora_uses_correct_delta_shape_math(self):
        linear = _DummyLinear()
        model = _DummyModel(layer=linear)
        mx = _FakeMx()
        nn = types.SimpleNamespace()

        with patch.object(training_mod, "try_import_mlx", return_value=(mx, nn)):
            patched = apply_lora(model, rank=2, alpha=2.0, targets=("layer",))
            self.assertEqual(len(patched), 1)
            wrapper = patched[0]
            wrapper.A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)  # [in=2, r=2]
            wrapper.B = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]], dtype=float)  # [r=2, out=3]
            before = linear.weight.copy()
            merge_lora(model)

        expected_delta = (wrapper.A @ wrapper.B).T * wrapper.scale
        np.testing.assert_allclose(linear.weight, before + expected_delta)
        self.assertNotIn("_lora_A", linear)
        self.assertNotIn("_lora_B", linear)
        self.assertNotIn("_lora_scale", linear)

    def test_apply_lora_warns_if_layer_patch_fails(self):
        class _BrokenLinear:
            def __init__(self) -> None:
                self.weight = np.zeros((2, 2), dtype=float)

            def __call__(self, x):  # noqa: ANN001
                return x

            def __setitem__(self, key, value):  # noqa: ANN001
                raise RuntimeError("blocked")

        model = _DummyModel(layer=_BrokenLinear())
        mx = _FakeMx()
        nn = types.SimpleNamespace()

        with patch.object(training_mod, "try_import_mlx", return_value=(mx, nn)):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                patched = apply_lora(model, rank=2, alpha=1.0, targets=("layer",))

        self.assertEqual(patched, [])
        msgs = [str(w.message) for w in rec]
        self.assertTrue(any("apply_lora: failed to patch" in m for m in msgs))

    def test_train_step_grad_accum_updates_only_on_target_step(self):
        mx = _FakeMx()
        nn = _FakeNn(grads=[{"w": 1.0}, {"w": 2.0}])
        cfg = TrainingConfig(grad_accum=2, grad_clip=1.0)
        model = _TrainDummyModel()
        optimizer = _TrainDummyOptimizer()
        batch = {"tokens": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)}
        fake_opt_mod = types.ModuleType("mlx.optimizers")
        fake_opt_mod.clip_grad_norm = lambda grads, clip: (grads, None)
        fake_mlx_pkg = types.ModuleType("mlx")

        with patch.dict(sys.modules, {"mlx": fake_mlx_pkg, "mlx.optimizers": fake_opt_mod}):
            with patch.object(training_mod, "try_import_mlx", return_value=(mx, nn)):
                with patch.object(training_mod, "loss_forward", return_value=np.zeros((2, 3, 5), dtype=float)):
                    with patch.object(training_mod, "xent_loss", return_value=_FakeLoss(1.0)):
                        train_step(model, batch, optimizer, cfg)
                        self.assertEqual(len(optimizer.calls), 0)
                        train_step(model, batch, optimizer, cfg)

        self.assertEqual(len(optimizer.calls), 1)
        self.assertAlmostEqual(float(optimizer.calls[0]["w"]), 1.5, places=6)

    def test_train_step_passes_shifted_mask_into_supervised_loss(self):
        mx = _FakeMx()
        nn = _FakeNn(grads=[{"w": 1.0}])
        cfg = TrainingConfig(grad_accum=1, grad_clip=1.0)
        model = _TrainDummyModel()
        optimizer = _TrainDummyOptimizer()
        batch = {
            "tokens": np.array([[11, 12, 13, 14]], dtype=np.int32),
            "mask": np.array([[1, 0, 1, 1]], dtype=np.int32),
        }
        fake_opt_mod = types.ModuleType("mlx.optimizers")
        fake_opt_mod.clip_grad_norm = lambda grads, clip: (grads, None)
        fake_mlx_pkg = types.ModuleType("mlx")
        seen = {"loss_forward_mask": None, "xent_mask": None}

        def _fake_loss_forward(model, tok_batch, mask=None, hooks=None):  # noqa: ANN001
            seen["loss_forward_mask"] = np.asarray(mask).copy()
            return np.zeros((tok_batch.shape[0], tok_batch.shape[1], 3), dtype=float)

        def _fake_xent(logits, labels, **kwargs):  # noqa: ANN001
            seen["xent_mask"] = np.asarray(kwargs.get("mask")).copy()
            return _FakeLoss(1.0)

        with patch.dict(sys.modules, {"mlx": fake_mlx_pkg, "mlx.optimizers": fake_opt_mod}):
            with patch.object(training_mod, "try_import_mlx", return_value=(mx, nn)):
                with patch.object(training_mod, "loss_forward", side_effect=_fake_loss_forward):
                    with patch.object(training_mod, "xent_loss", side_effect=_fake_xent):
                        train_step(model, batch, optimizer, cfg)

        np.testing.assert_array_equal(seen["loss_forward_mask"], np.array([[1, 0, 1, 1]], dtype=np.int32))
        np.testing.assert_array_equal(seen["xent_mask"], np.array([[0, 1, 1]], dtype=np.int32))

    def test_ppo_entropy_uses_sampled_action_estimator(self):
        with patch.object(ppo_mod, "try_import_mlx", return_value=(_FakeMx(), None)):
            cfg = ppo_mod.PPOConfig(entropy_coef=0.3)
            logp_new = np.array([-1.0, -0.5], dtype=float)
            logp_old = np.array([-1.2, -0.4], dtype=float)
            adv = np.array([0.6, -0.2], dtype=float)
            value_pred = np.array([0.5, 0.1], dtype=float)
            value_target = np.array([0.4, -0.1], dtype=float)
            _, stats = ppo_mod.ppo_loss(
                logp_new=logp_new,
                logp_old=logp_old,
                advantage=adv,
                value_pred=value_pred,
                value_target=value_target,
                cfg=cfg,
            )
        self.assertAlmostEqual(float(stats["entropy"]), float(-logp_new.mean()), places=6)

    def test_prefetch_loader_does_not_materialize_iterable_dataset(self):
        class _StreamDataset:
            def __init__(self) -> None:
                self.iter_calls = 0

            def __iter__(self):
                self.iter_calls += 1
                for i in range(4):
                    yield [i]

        ds = _StreamDataset()
        with patch.object(loader_mod, "try_import_mlx", return_value=(types.SimpleNamespace(int32="int32"), None)):
            with patch.object(loader_mod, "as_mx_array", side_effect=lambda data, dtype=None: data):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dl = PrefetchDataLoader(ds, batch_size=2, prefetch=1, seed=7)
                self.assertEqual(ds.iter_calls, 0)
                batches = list(dl)

        self.assertEqual(ds.iter_calls, 1)
        flat = [item[0] for batch in batches for item in batch]
        self.assertEqual(flat, [0, 1, 2, 3])

    def test_prefetch_loader_shuffles_sequence_dataset(self):
        class _SeqDataset:
            def __init__(self, values):
                self.values = values

            def __len__(self):
                return len(self.values)

            def __getitem__(self, idx):  # noqa: ANN001
                return [self.values[idx]]

        ds = _SeqDataset([0, 1, 2, 3, 4])
        expected_idx = list(range(len(ds)))
        random.Random(11).shuffle(expected_idx)
        expected = [ds[i][0] for i in expected_idx]

        with patch.object(loader_mod, "try_import_mlx", return_value=(types.SimpleNamespace(int32="int32"), None)):
            with patch.object(loader_mod, "as_mx_array", side_effect=lambda data, dtype=None: data):
                dl = PrefetchDataLoader(ds, batch_size=2, prefetch=1, seed=11)
                batches = list(dl)

        flat = [item[0] for batch in batches for item in batch]
        self.assertEqual(flat, expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
