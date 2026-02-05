from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlx_genkit.eval import EvalSuite
from mlx_genkit.structure.result import GenerateResult


class EvalSuiteTests(unittest.TestCase):
    def test_eval_suite_runs_with_stub_generate(self):
        suite_payload = {
            "name": "unit_suite",
            "model": "stub/model",
            "config": {"max_tokens": 8, "temperature": 0.0},
            "cases": [
                {
                    "name": "case_one",
                    "prompt": "Return a JSON object with field foo",
                    "json_schema": {
                        "type": "object",
                        "properties": {"foo": {"type": "string"}},
                        "required": ["foo"],
                    },
                    "retries": 1,
                    "strict_only_json": True,
                }
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(suite_payload, handle)
            suite_path = handle.name

        try:
            fake_model = object()
            fake_tokenizer = object()

            def _fake_generate(*args, **kwargs):
                return GenerateResult(
                    text='{"foo": "bar"}',
                    json={"foo": "bar"},
                    schema_ok=True,
                    attempts=1,
                    only_json=True,
                )

            with patch(
                "mlx_genkit.eval.auto_load",
                return_value=(fake_model, fake_tokenizer, "./local"),
            ) as load_mock, patch(
                "mlx_genkit.eval.generate",
                side_effect=_fake_generate,
            ) as gen_mock:
                suite = EvalSuite(suite_path)
                outcomes = suite.run()

            self.assertEqual(load_mock.call_count, 1)
            self.assertEqual(gen_mock.call_count, 1)
            self.assertEqual(len(outcomes), 1)
            outcome = outcomes[0]
            self.assertTrue(outcome.result.schema_ok)
            self.assertEqual(outcome.result.json, {"foo": "bar"})
            markdown = EvalSuite.render_markdown(outcomes, suite.name)
            self.assertIn("case_one", markdown)
            summary = EvalSuite.to_dict(outcomes, suite.name)
            self.assertEqual(summary["passed"], 1)
        finally:
            os.unlink(suite_path)

    def test_eval_suite_resolves_relative_paths_from_suite_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite_dir = Path(tmpdir)
            fixtures = suite_dir / "fixtures"
            fixtures.mkdir(parents=True, exist_ok=True)

            grammar_text = 'root ::= "ok"'
            (fixtures / "case.gbnf").write_text(grammar_text, encoding="utf-8")
            (fixtures / "checks.json").write_text(
                json.dumps(
                    [
                        {
                            "type": "must_contain",
                            "field": "foo",
                            "substrings": ["bar"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            suite_payload = {
                "name": "relative_paths_suite",
                "model": "stub/model",
                "semantic_checks": "fixtures/checks.json",
                "cases": [
                    {
                        "name": "relative_case",
                        "prompt": "Return JSON",
                        "grammar_gbnf": "fixtures/case.gbnf",
                    }
                ],
            }
            suite_path = suite_dir / "suite.json"
            suite_path.write_text(json.dumps(suite_payload), encoding="utf-8")

            suite = EvalSuite(str(suite_path))

            self.assertEqual(suite.cases[0].grammar.kind, "gbnf")
            self.assertEqual(suite.cases[0].grammar.payload, grammar_text)
            self.assertIsNotNone(suite.cases[0].semantic_checks)
            self.assertEqual(len(suite.cases[0].semantic_checks), 1)
            check = suite.cases[0].semantic_checks[0]
            self.assertEqual(getattr(check, "name", None), "must_contain")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
