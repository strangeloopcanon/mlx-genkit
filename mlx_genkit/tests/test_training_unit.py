from __future__ import annotations

import unittest

from mlx_genkit.training import merge_lora


class _DummyLinear:
    def __call__(self, x):  # noqa: ANN001
        return x


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


class LoRAMergeTests(unittest.TestCase):
    def test_merge_lora_detects_wrapper_and_calls_merge_restore(self):
        linear = _DummyLinear()
        wrapper = _DummyLoRAWrapper(linear)
        linear.__call__ = wrapper.__call__

        model = _DummyModel(layer=linear)
        merge_lora(model)

        self.assertTrue(wrapper.merged)
        self.assertTrue(wrapper.restored)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

