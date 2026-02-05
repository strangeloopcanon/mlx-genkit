from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, Tuple
from unittest.mock import patch

import mlx_genkit.backends.mlx_backend as mlx_backend
from mlx_genkit.config import GenerationConfig


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


class _FakeTokenBatch:
    def __getitem__(self, _key: Any) -> "_FakeTokenBatch":
        return self


class _FakeTokenizer:
    def __init__(
        self,
        *,
        eos_token_id: int = 0,
        vocab_size: int = 512,
        encode_override: Any = None,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.pad_token_id = None
        self.vocab_size = vocab_size
        self._encode_override = encode_override

    def encode(self, text: str, add_special_tokens: bool = False):  # noqa: ARG002
        if self._encode_override is not None:
            return list(self._encode_override(text))
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(int(x)) for x in ids)

    def __len__(self) -> int:
        return int(self.vocab_size)


class StepLogitsTests(unittest.TestCase):
    def test_step_logits_handles_tensor_output(self):
        model = _StubModel(_FakeArray((1, 5, 10)))
        components = SimpleNamespace(hidden_size=0, vocab_size=0)
        logits, cache = mlx_backend._step_logits(model, components, None, cache="cache")
        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(cache, "cache")

    def test_step_logits_handles_tuple_output_with_cache(self):
        model = _StubModel((_FakeArray((1, 5, 10)), "new_cache"))
        components = SimpleNamespace(hidden_size=0, vocab_size=0)
        logits, cache = mlx_backend._step_logits(model, components, None, cache="old_cache")
        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(cache, "new_cache")

    def test_step_logits_handles_dict_output_with_cache(self):
        model = _StubModel({"logits": _FakeArray((1, 10)), "cache": "new_cache"})
        components = SimpleNamespace(hidden_size=0, vocab_size=0)
        logits, cache = mlx_backend._step_logits(model, components, None, cache="old_cache")
        self.assertEqual(logits.shape, (1, 10))
        self.assertEqual(cache, "new_cache")

    def test_step_logits_handles_1d_logits(self):
        model = _StubModel(_FakeArray((10,)))
        components = SimpleNamespace(hidden_size=0, vocab_size=0)
        logits, cache = mlx_backend._step_logits(model, components, None, cache=None)
        self.assertEqual(logits.shape, (1, 10))
        self.assertIsNone(cache)


class GenerationTextBehaviorTests(unittest.TestCase):
    def test_generate_dict_returns_generated_suffix_only(self):
        tokenizer = _FakeTokenizer()
        tk = mlx_backend.make_tokenizer_bridge(tokenizer)
        prompt_ids = [ord("H"), ord("i")]
        cfg = GenerationConfig(max_tokens=0)
        setup = mlx_backend.GenerationSetup(
            model=object(),
            tokenizer=tokenizer,
            config=cfg,
            hooks=None,
            tk=tk,
            prompt_tokens=list(prompt_ids),
            components=SimpleNamespace(layers=[], hidden_size=0, vocab_size=0),
            eos_ids=[],
            processors=[],
            stop_token_sequences=[],
            raw_stop_strings=[],
            prompt_cache=None,
            residual_hooks=[],
            injector=object(),  # disable mlx-lm fast path for this test
            forced_map={},
            forced_bos=None,
        )
        fake_mx = SimpleNamespace(int32="int32")
        with patch.object(mlx_backend, "try_import_mlx", return_value=(fake_mx, None)):
            with patch.object(mlx_backend, "as_mx_array", return_value=_FakeTokenBatch()):
                with patch.object(mlx_backend, "_build_generation_setup", return_value=setup):
                    out = mlx_backend.generate_dict(object(), tokenizer, "Hi", cfg)
        self.assertEqual(out["tokens"], prompt_ids)
        self.assertEqual(out["text"], "")

    def test_beam_search_returns_generated_suffix_only(self):
        tokenizer = _FakeTokenizer()
        tk = mlx_backend.make_tokenizer_bridge(tokenizer)
        prompt_ids = [ord("P"), ord("R")]
        cfg = GenerationConfig(num_beams=2, max_tokens=0)
        setup = mlx_backend.GenerationSetup(
            model=object(),
            tokenizer=tokenizer,
            config=cfg,
            hooks=None,
            tk=tk,
            prompt_tokens=list(prompt_ids),
            components=SimpleNamespace(layers=[], hidden_size=0, vocab_size=0),
            eos_ids=[],
            processors=[],
            stop_token_sequences=[],
            raw_stop_strings=[],
            prompt_cache=None,
            residual_hooks=[],
            injector=None,
            forced_map={},
            forced_bos=None,
        )
        with patch.object(mlx_backend, "try_import_mlx", return_value=(SimpleNamespace(), None)):
            out = mlx_backend._beam_search_generate(setup)
        self.assertEqual(out["tokens"], prompt_ids)
        self.assertEqual(out["text"], "")


class RawStopAndSpeculativeHelperTests(unittest.TestCase):
    def test_incremental_raw_stop_matcher_trims_generated_text(self):
        tokenizer = _FakeTokenizer()
        tk = mlx_backend.make_tokenizer_bridge(tokenizer)
        state = mlx_backend._make_incremental_raw_stop_state(["STOP"], enabled=True)
        self.assertIsNotNone(state)
        tokens = [ord("P")]
        hit = False
        for ch in "ABCSTOPZZ":
            tok = ord(ch)
            tokens.append(tok)
            tokens, hit = mlx_backend._match_raw_stops_incremental(
                tk=tk,
                tokens=tokens,
                prompt_len=1,
                raw_stops=["STOP"],
                state=state,  # type: ignore[arg-type]
                new_token=tok,
            )
            if hit:
                break
        self.assertTrue(hit)
        self.assertEqual(tokenizer.decode(tokens[1:]), "ABC")

    def test_speculative_tokenizer_compatibility_detects_vocab_mismatch(self):
        base = _FakeTokenizer(eos_token_id=2, vocab_size=128)
        draft = _FakeTokenizer(eos_token_id=2, vocab_size=129)
        ok, reason = mlx_backend._check_speculative_tokenizer_compatibility(base, draft, prompt_hint="hello")
        self.assertFalse(ok)
        self.assertEqual(reason, "speculative_vocab_size_mismatch")


class BackendMetaTests(unittest.TestCase):
    def test_generate_exposes_speculative_fallback_reason_in_meta(self):
        backend = mlx_backend.MlxGenerateBackend(model=object(), tokenizer=object())
        raw = {
            "text": "result",
            "tokens": [1],
            "eos_reached": False,
            "finish_reason": "length",
            "speculative_fallback_reason": "speculative_vocab_size_mismatch",
        }
        with patch.object(mlx_backend, "generate_dict", return_value=raw):
            out = backend.generate("prompt", GenerationConfig())
        self.assertEqual(out.meta.get("speculative_fallback_reason"), "speculative_vocab_size_mismatch")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
