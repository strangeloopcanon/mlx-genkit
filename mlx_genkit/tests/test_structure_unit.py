from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from mlx_genkit.api import GenerationConfig, generate
from mlx_genkit.structure.adherence import JsonAdherence, StructuredGenerationEngine
from mlx_genkit.structure.batch import generate_many
from mlx_genkit.structure.result import GenerateResult
from mlx_genkit.structure.grammar import Grammar
from mlx_genkit.structure.stream import StreamCallbacks, build_stream_observer


class _StubBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.name = "stub"

    def generate(self, prompt, config, hooks=None, stream_observer=None):
        text = next(self.outputs)
        return GenerateResult(text=text)


class _CharTokenizer:
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class StructureTests(unittest.TestCase):
    def test_generate_returns_result_object(self):
        fake_backend = _StubBackend(["ok"])
        cfg = GenerationConfig(max_tokens=8)
        with patch("mlx_genkit.api.resolve_backend", return_value=fake_backend):
            result = generate(None, None, "hi", cfg)
        self.assertIsInstance(result, GenerateResult)
        self.assertEqual(result["text"], "ok")

    def test_structured_generation_with_retry(self):
        backend = _StubBackend(["not json", json.dumps({"a": 1})])
        cfg = GenerationConfig(max_tokens=16)
        engine = StructuredGenerationEngine(
            backend=backend,
            prompt="Return JSON",
            config=cfg,
            hooks=None,
            json_schema={"type": "object", "properties": {"a": {"type": "integer"}}},
            grammar=None,
            validators=[lambda data: (data.get("a") == 1, None)],
            semantic_checks=None,
            adherence=JsonAdherence(retries=1, strict_only_json=True),
            on_parse_fail=None,
            on_semantic_fail=None,
            log_writer=None,
        )
        result = engine.run()
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.schema_ok)
        self.assertEqual(result.json["a"], 1)

    def test_generate_many_batch_summary(self):
        outputs = [GenerateResult(text=str(i), json={"i": i}, schema_ok=True, semantic_ok=True) for i in range(3)]

        def _fake_generate(*args, **kwargs):
            return outputs.pop(0)

        items = ["a", "b", "c"]
        with patch("mlx_genkit.structure.batch.generate", side_effect=_fake_generate):
            batch, summary = generate_many(None, None, items)
        self.assertEqual(batch.summary["total"], 3)
        self.assertEqual(len(batch.results), 3)
        self.assertIsNone(summary)
        self.assertTrue(batch.ok())

    def test_unsupported_grammar_raises(self):
        backend = _StubBackend(["{}"])
        backend.name = "mlx"
        with self.assertRaises(NotImplementedError):
            StructuredGenerationEngine(
                backend=backend,
                prompt="hi",
                config=GenerationConfig(),
                hooks=None,
                json_schema=None,
                grammar=Grammar.gbnf("start ::= 'a'"),
                validators=None,
                semantic_checks=None,
                adherence=JsonAdherence(),
                on_parse_fail=None,
                on_semantic_fail=None,
                log_writer=None,
            )

    def test_strict_json_auto_trims_fenced_preface(self):
        text = "Here you go:\n```json\n{\n  \"a\": 1\n}\n```"
        backend = _StubBackend([text])
        engine = StructuredGenerationEngine(
            backend=backend,
            prompt="respond",
            config=GenerationConfig(),
            hooks=None,
            json_schema={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
            grammar=None,
            validators=[lambda data: (isinstance(data.get("a"), int), None)],
            semantic_checks=None,
            adherence=JsonAdherence(retries=0, strict_only_json=True),
            on_parse_fail=None,
            on_semantic_fail=None,
            log_writer=None,
        )
        result = engine.run()
        self.assertTrue(result.schema_ok)
        self.assertTrue(result.only_json)
        self.assertEqual(result.json, {"a": 1})
        self.assertEqual(result.text, '{\n  "a": 1\n}')

    def test_strict_json_without_auto_trim_fails_on_preface(self):
        text = "Here you go:\n```json\n{\n  \"a\": 1\n}\n```"
        backend = _StubBackend([text])
        engine = StructuredGenerationEngine(
            backend=backend,
            prompt="respond",
            config=GenerationConfig(),
            hooks=None,
            json_schema={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
            grammar=None,
            validators=[lambda data: (isinstance(data.get("a"), int), None)],
            semantic_checks=None,
            adherence=JsonAdherence(retries=0, strict_only_json=True, auto_trim_fences=False),
            on_parse_fail=None,
            on_semantic_fail=None,
            log_writer=None,
        )
        result = engine.run()
        self.assertFalse(result.schema_ok)
        self.assertIsNone(result.json)
        self.assertTrue(any(v.get("type") == "parse_error" for v in result.violations))

    def test_json_schema_grammar_reports_not_supported(self):
        backend = _StubBackend([json.dumps({"a": 1})])
        engine = StructuredGenerationEngine(
            backend=backend,
            prompt="respond",
            config=GenerationConfig(),
            hooks=None,
            json_schema={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
            grammar=Grammar.json_schema({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}),
            validators=[lambda data: (isinstance(data.get("a"), int), None)],
            semantic_checks=None,
            adherence=JsonAdherence(retries=0, strict_only_json=True),
            on_parse_fail=None,
            on_semantic_fail=None,
            log_writer=None,
        )
        result = engine.run()
        self.assertEqual(result.meta.get("grammar_kind"), "json_schema")
        self.assertFalse(result.meta.get("grammar_supported"))

    def test_stream_observer_tracks_valid_json(self):
        adherence = JsonAdherence(strict_only_json=True)
        tokenizer = _CharTokenizer()
        seen = []
        callbacks = StreamCallbacks(on_token=lambda tok, idx: seen.append((tok, idx)))
        observer = build_stream_observer(
            tokenizer=tokenizer,
            callbacks=callbacks,
            adherence=adherence,
            expect_json=True,
        )
        observer.reset()
        tokens = [ord(ch) for ch in '{"a":1}']
        for tok in tokens:
            cont = observer.emit_token(tok)
            self.assertTrue(cont)
        self.assertEqual(observer.emitted, len(tokens))
        self.assertFalse(observer.invalid_triggered)
        self.assertEqual([tok for tok, _ in seen], tokens)

    def test_stream_observer_detects_invalid_preface(self):
        adherence = JsonAdherence(strict_only_json=True)
        tokenizer = _CharTokenizer()
        captured = []
        callbacks = StreamCallbacks(on_invalid_path=lambda info: captured.append(info))
        observer = build_stream_observer(
            tokenizer=tokenizer,
            callbacks=callbacks,
            adherence=adherence,
            expect_json=True,
        )
        observer.reset()
        stopped = False
        for ch in "Here you go:\n":
            cont = observer.emit_token(ord(ch))
            if not cont:
                stopped = True
                break
        self.assertTrue(stopped)
        self.assertTrue(observer.invalid_triggered)
        self.assertTrue(captured)

    def test_stream_observer_can_continue_on_invalid(self):
        adherence = JsonAdherence(strict_only_json=True)
        tokenizer = _CharTokenizer()
        captured = []
        callbacks = StreamCallbacks(
            on_invalid_path=lambda info: captured.append(info),
            stop_on_invalid=False,
        )
        observer = build_stream_observer(
            tokenizer=tokenizer,
            callbacks=callbacks,
            adherence=adherence,
            expect_json=True,
        )
        observer.reset()
        for ch in "Here you go:\n{\"foo\":":
            cont = observer.emit_token(ord(ch))
        self.assertTrue(cont)
        self.assertTrue(captured)
        self.assertTrue(observer.invalid_triggered)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
