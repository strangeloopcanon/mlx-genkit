from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Tuple

from mlx_genkit.backends.mlx_backend import _step_logits


class _FakeArray:
    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.shape = tuple(int(x) for x in shape)
        self.ndim = len(self.shape)

    def __getitem__(self, key: Any) -> "_FakeArray":
        if not (isinstance(key, tuple) and len(key) == 3 and self.ndim == 3):
            raise TypeError("Unsupported indexing")
        b, l, v = self.shape
        new_shape = []
        if isinstance(key[0], slice):
            new_shape.append(b)
        if isinstance(key[1], slice):
            new_shape.append(l)
        if isinstance(key[2], slice):
            new_shape.append(v)
        return _FakeArray(tuple(new_shape))

    def reshape(self, shape: Tuple[int, ...]) -> "_FakeArray":
        if shape == (1, -1) and self.ndim == 1:
            return _FakeArray((1, self.shape[0]))
        return _FakeArray(shape)


class _StubModel:
    def __init__(self, output: Any) -> None:
        self._output = output

    def __call__(self, _tokens, *, cache=None, input_embeddings=None):  # noqa: ARG002
        return self._output


class StepLogitsTests(unittest.TestCase):
    def test_step_logits_handles_tensor_output(self):
        model = _StubModel(_FakeArray((1, 5, 10)))
        components = SimpleNamespace(hidden_size=0, vocab_size=0)
        logits, cache = _step_logits(model, components, None, cache="cache")
        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(cache, "cache")

    def test_step_logits_handles_tuple_output_with_cache(self):
        model = _StubModel((_FakeArray((1, 5, 10)), "new_cache"))
        components = SimpleNamespace(hidden_size=0, vocab_size=0)
        logits, cache = _step_logits(model, components, None, cache="old_cache")
        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(cache, "new_cache")

    def test_step_logits_handles_dict_output_with_cache(self):
        model = _StubModel({"logits": _FakeArray((1, 10)), "cache": "new_cache"})
        components = SimpleNamespace(hidden_size=0, vocab_size=0)
        logits, cache = _step_logits(model, components, None, cache="old_cache")
        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(cache, "new_cache")

    def test_step_logits_handles_1d_logits(self):
        model = _StubModel(_FakeArray((10,)))
        components = SimpleNamespace(hidden_size=0, vocab_size=0)
        logits, cache = _step_logits(model, components, None, cache=None)
        self.assertEqual(logits.shape, (1, 10))
        self.assertIsNone(cache)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

