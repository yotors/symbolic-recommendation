"""Compact translation boundaries with mocked prediction and the real compiler."""

import copy
import asyncio
import importlib.util
import json
import subprocess
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from recommendation.integrations import llm_compact_backend as compact
from recommendation.integrations.llm_article_facts import _configured_dspy, _library


def _row(index=0, **changes):
    result = {"source_index": index, "concepts": [], "format": None,
              "event_types": [], "intents": [], "audiences": []}
    result.update(changes)
    return result


class CompactImportTest(unittest.TestCase):
    def test_module_import_does_not_load_dspy_or_nl2pln(self):
        result = subprocess.run([
            sys.executable, "-c", "import sys; import recommendation.integrations.llm_compact_backend; "
            "assert 'dspy' not in sys.modules; assert 'nl2pettachainer' not in sys.modules",
        ], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(importlib.util.find_spec("dspy"), "Typed backend tests use existing NL2PLN/.venv")
class CompactBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use the same confined, disabled-cache bootstrap as real extraction;
        # importing the SDK or constructing a signature makes no provider call.
        cls.real_dspy = _configured_dspy()

    def setUp(self):
        self.library = _library()
        self.models = self.library.models
        from nl2pettachainer.backends.base import BackendError
        self.error = BackendError
        self.contexts = []
        self.adapter = object()

        @contextmanager
        def context(**settings):
            self.contexts.append(settings)
            yield

        self.dspy = SimpleNamespace(context=context, JSONAdapter=lambda: self.adapter)

    def translate(self, rows, *, count=1, feedback="", extra=None):
        lm = SimpleNamespace(cache=False, kwargs={"max_tokens": 1024, "num_retries": 0,
                                                 "request_timeout": 30.0})
        prediction = SimpleNamespace(descriptors=rows, **(extra or {}))
        backend = SimpleNamespace(_lm=lm, _predict=Mock(return_value=prediction),
                                  _prediction_usage=Mock(return_value=self.models.Usage(
                                      input_tokens=90, output_tokens=45)))
        request = self.models.TranslateRequest(
            sentences=[json.dumps({"article_subject": f"article_{index}", "title": "A launch",
                                   "abstract": "A new spacecraft launched."}) for index in range(count)],
            namespace="unit", context={"predicates": ["HasConcept"]},
        )
        with patch.dict(sys.modules, {"dspy": self.dspy}), \
                patch.object(compact, "make_strict_json_adapter", return_value=self.adapter):
            result = compact.translate_compact(backend, request, feedback)
        return result, backend, request

    def test_typed_descriptor_schema_is_compact_strict_and_bounded(self):
        model = compact._descriptor_model()
        schema = model.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]), set(_row()))
        self.assertEqual(schema["properties"]["concepts"]["maxItems"], 8)
        for field in ("event_types", "intents", "audiences"):
            self.assertEqual(schema["properties"][field]["maxItems"], 4)
        format_types = schema["properties"]["format"]["anyOf"]
        text_format = next(item for item in format_types if item["type"] == "string")
        self.assertEqual(text_format["minLength"], 1)
        self.assertEqual(text_format["maxLength"], 160)
        self.assertNotIn("enum", text_format)
        self.assertNotIn("truth", json.dumps(schema))

    def test_predictor_has_only_content_inputs_and_typed_descriptor_output(self):
        signature = Mock(return_value="typed-signature")
        predict = Mock(return_value="predictor")
        fake = SimpleNamespace(Signature=signature, Predict=predict,
                               InputField=lambda **kw: kw, OutputField=lambda **kw: kw)
        with patch.dict(sys.modules, {"dspy": fake}):
            result = compact.make_compact_predictor("Specific content policy.")
        self.assertEqual(result, "predictor")
        fields, instructions = signature.call_args.args
        self.assertEqual(set(fields), {"sentences", "validator_feedback", "descriptors"})
        self.assertEqual(fields["sentences"][0], list[str])
        self.assertEqual(fields["descriptors"][0], list[compact._descriptor_model()])
        self.assertIn("Specific content policy.", instructions)
        self.assertIn("EVERY input-list position", instructions)
        predict.assert_called_once_with("typed-signature")

    def test_real_dspy_accepts_the_compact_typed_signature_without_inference(self):
        predictor = compact.make_compact_predictor("Extract grounded article content.")
        self.assertIsInstance(predictor, self.real_dspy.Predict)
        self.assertEqual(set(predictor.signature.input_fields), {"sentences", "validator_feedback"})
        self.assertEqual(set(predictor.signature.output_fields), {"descriptors"})
        self.assertEqual(predictor.signature.output_fields["descriptors"].annotation,
                         list[compact._descriptor_model()])

    def test_real_compiler_receives_ground_facts_with_fixed_assertion_stv(self):
        (draft, usage), _, _ = self.translate([_row(
            concepts=[" Space   Exploration ", "space exploration"], format="report",
            event_types=["Spacecraft Launch"], intents=["Inform"], audiences=["Engineers"],
        )])
        self.assertEqual(len(draft.statements), 5)
        self.assertFalse(draft.queries)
        for fact in draft.statements:
            self.assertIsInstance(fact, self.models.FactDraft)
            self.assertIsInstance(fact.atom.arguments[0], self.models.SymbolTerm)
            self.assertIsInstance(fact.atom.arguments[1], self.models.StringTerm)
            self.assertEqual(fact.atom.arguments[0].value, "article_0")
            self.assertEqual(fact.source_sentence_indexes, [0])
            self.assertEqual(fact.truth.model_dump(), {"strength": 1.0, "confidence": 1.0})
        statements, queries = self.library.compiler.compile_translation(draft, "unit")
        self.assertEqual(len(statements), 5)
        self.assertFalse(queries)
        self.assertIn('(HasConcept article_0 "space exploration")', statements[0].source)
        self.assertEqual(usage.model_dump(), {"input_tokens": 90, "output_tokens": 45})

    def test_existing_lm_safety_settings_and_call_boundary_are_preserved(self):
        (_, _), backend, request = self.translate([_row()], feedback="bounded validator feedback")
        self.assertEqual(self.contexts, [{"lm": backend._lm, "adapter": self.adapter, "track_usage": True}])
        self.assertFalse(backend._lm.cache)
        self.assertEqual(backend._lm.kwargs, {"max_tokens": 1024, "num_retries": 0,
                                            "request_timeout": 30.0})
        backend._predict.assert_called_once_with(sentences=request.sentences,
                                                validator_feedback="bounded validator feedback")
        backend._prediction_usage.assert_called_once()

    def test_zero_fact_row_is_required_and_does_not_invent_a_dummy_fact(self):
        (draft, _), _, _ = self.translate([_row(1, concepts=["astronomy"]), _row(0)], count=2)
        self.assertEqual(len(draft.statements), 1)
        self.assertEqual(draft.statements[0].source_sentence_indexes, [1])
        self.assertEqual(draft.statements[0].atom.arguments[0].value, "article_1")
        (empty, _), _, _ = self.translate([_row(0), _row(1)], count=2)
        self.assertEqual(empty.statements, [])
        self.assertEqual(empty.queries, [])

    def test_out_of_order_rows_compile_deterministically_without_source_mixups(self):
        rows = [_row(0, concepts=["astronomy"]), _row(1, concepts=["football"])]
        (forward, _), _, _ = self.translate(rows, count=2)
        (reverse, _), _, _ = self.translate(rows[::-1], count=2)
        self.assertEqual(forward, reverse)

    def test_missing_duplicate_out_of_range_and_noninteger_indexes_reject(self):
        invalid = [[], [_row(0), _row(0)], [_row(0), _row(2)], [_row(-1), _row(1)],
                   [_row(True), _row(0)], [_row("0"), _row(1)], [_row(0.0), _row(1)]]
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(self.error):
                self.translate(rows, count=2)

    def test_extra_ids_labels_rules_queries_code_or_truth_fields_are_rejected(self):
        for field in ("id", "article_id", "label", "rules", "queries", "code", "truth", "score"):
            with self.subTest(field=field), self.assertRaises(self.error):
                self.translate([_row(**{field: "unsupported"})])
        with self.assertRaises(self.error):
            self.translate([_row()], extra={"rules": []})

    def test_descriptor_quotas_types_and_unknown_sentinels_are_enforced(self):
        invalid = [
            _row(concepts=["x"] * 9), _row(event_types=["x"] * 5),
            _row(intents=["x"] * 5), _row(audiences=["x"] * 5),
            _row(concepts="not a list"), _row(concepts=[42]), _row(concepts=[None]),
            _row(concepts=["x" * 161]), _row(concepts=[" "]),
            _row(concepts=["line\nbreak"]), _row(concepts=["unknown"]),
        ]
        for row in invalid:
            with self.subTest(row=row), self.assertRaises(self.error):
                self.translate([row])

    def test_format_is_open_grounded_text_without_mapping_to_old_enum(self):
        for value, expected in (("analysis", "analysis"), ("gallery", "gallery"),
                                ("photo essay", "photo essay"), (" REPORT ", "report"),
                                (" ＡＮＡＬＹＳＩＳ ", "analysis"), ("x" * 160, "x" * 160)):
            with self.subTest(value=value):
                (draft, _), _, _ = self.translate([_row(format=value)])
                self.assertEqual(len(draft.statements), 1)
                fact = draft.statements[0]
                self.assertEqual(fact.atom.predicate, "HasFormat")
                self.assertEqual(fact.atom.arguments[1].value, expected)
                statements, _ = self.library.compiler.compile_translation(draft, "unit")
                self.assertIn(json.dumps(expected), statements[0].source)

    def test_format_rejects_nontext_control_overlength_and_unknown_without_truncating(self):
        invalid = [0, True, [], {"genre": "analysis"}, "", " ", "x" * 161,
                   "photo\nessay", "gallery\tview", "analysis\x7f", "unknown", "none",
                   "null", "n/a", "not specified", " UNKNOWN "]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(self.error):
                self.translate([_row(format=value)])

    def test_shared_library_statement_limit_is_enforced_not_silently_truncated(self):
        rows = [_row(index, concepts=[f"concept {i}" for i in range(8)], format="report",
                     event_types=[f"event {i}" for i in range(4)],
                     intents=[f"intent {i}" for i in range(4)],
                     audiences=[f"audience {i}" for i in range(4)]) for index in range(48)]
        # 48 * 21 = 1,008 facts exceeds TranslationDraft's shared 1,000 cap.
        with self.assertRaises(self.error):
            self.translate(rows, count=48)

    def test_model_strings_are_escaped_by_compiler_never_executed_as_code(self):
        payload = '\") (eval evil) ("'
        (draft, _), _, _ = self.translate([_row(concepts=[payload])])
        statements, _ = self.library.compiler.compile_translation(draft, "unit")
        self.assertEqual(draft.statements[0].atom.arguments[1].value, payload)
        self.library.contract.validate_statement(statements[0].source)
        self.assertIn(json.dumps(payload), statements[0].source)

    def test_queries_fail_before_predictor_and_provider_errors_remain_sanitized(self):
        request = self.models.TranslateRequest(sentences=["article text"], queries=["Rank me?"])
        backend = SimpleNamespace(_lm=object(), _predict=Mock())
        with patch.dict(sys.modules, {"dspy": self.dspy}), \
                patch.object(compact, "make_strict_json_adapter", return_value=self.adapter):
            with self.assertRaises(self.error):
                compact.translate_compact(backend, request)
        backend._predict.assert_not_called()
        backend._predict.side_effect = RuntimeError("PRIVATE PROVIDER MESSAGE")
        request = self.models.TranslateRequest(sentences=["article text"])
        with patch.dict(sys.modules, {"dspy": self.dspy}), \
                patch.object(compact, "make_strict_json_adapter", return_value=self.adapter):
            with self.assertRaises(self.error) as error:
                compact.translate_compact(backend, request)
        self.assertNotIn("PRIVATE", str(error.exception))
        backend._predict.assert_called_once()

    def test_input_descriptor_rows_are_not_mutated(self):
        rows = [_row(concepts=[" SPACE ", "space"])]
        original = copy.deepcopy(rows)
        self.translate(rows)
        self.assertEqual(rows, original)


@unittest.skipUnless(importlib.util.find_spec("dspy"), "Adapter tests use existing NL2PLN/.venv")
class StrictTextJSONAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dspy = _configured_dspy()
        from dspy.utils.exceptions import AdapterParseError
        cls.error = AdapterParseError

    def setUp(self):
        self.signature = compact.make_compact_predictor("Grounded content only.").signature
        self.adapter = compact.make_strict_json_adapter()
        self.payload = json.dumps({"descriptors": [_row(concepts=["space exploration"])]})
        self.inputs = {"sentences": ["A spacecraft launch."], "validator_feedback": ""}

    def test_valid_json_and_whole_json_fences_use_typed_rows(self):
        for text in (self.payload, f"```json\n{self.payload}\n```", f"```\n{self.payload}\n```"):
            with self.subTest(text=text):
                parsed = self.adapter.parse(self.signature, text)
                self.assertEqual(set(parsed), {"descriptors"})
                self.assertIsInstance(parsed["descriptors"][0], compact._descriptor_model())

    def test_malformed_json_wrappers_duplicates_and_extra_fields_are_not_repaired(self):
        invalid = [
            '{"descriptors": [],}', "{'descriptors': []}", '{"descriptors": [}',
            "Here is the output: " + self.payload, self.payload + " trailing prose",
            self.payload + self.payload, "```json\n" + self.payload,
            "```json\n" + self.payload + "\n```\nextra",
            '{"descriptors": [], "rules": []}', '{"descriptors": [], "descriptors": []}',
            '{"descriptors": "[]"}', '{"descriptors": null}', '{"descriptors": NaN}',
            '{"descriptors": Infinity}', '[]', '{}',
            json.dumps({"descriptors": [_row(rules=[])]}),
            json.dumps({"descriptors": [_row(source_index=True)]}),
        ]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(self.error):
                self.adapter.parse(self.signature, text)

    def test_one_transport_call_omits_response_format_tools_and_fallbacks(self):
        lm = Mock(return_value=[self.payload])
        lm.kwargs = {"max_tokens": 1024, "num_retries": 0, "cache": False}
        kwargs = {"temperature": 0.0, "response_format": {"type": "json_object"},
                  "tools": ["forbidden"], "tool_choice": "auto", "functions": [],
                  "function_call": "auto", "parallel_tool_calls": False}
        original = copy.deepcopy(kwargs)
        with patch.object(self.dspy.JSONAdapter, "__call__", side_effect=AssertionError("JSON fallback")), \
                patch.object(self.dspy.ChatAdapter, "__call__", side_effect=AssertionError("chat fallback")):
            result = self.adapter(lm, kwargs, self.signature, [], self.inputs)
        self.assertEqual(result[0]["descriptors"][0].concepts, ["space exploration"])
        lm.assert_called_once()
        self.assertEqual(set(lm.call_args.kwargs), {"messages", "temperature"})
        self.assertEqual(kwargs, original)
        self.assertEqual(lm.kwargs, {"max_tokens": 1024, "num_retries": 0, "cache": False})
        messages = lm.call_args.kwargs["messages"]
        self.assertIn("JSON object", " ".join(message["content"] for message in messages))

    def test_parse_failure_never_makes_a_second_transport_call(self):
        lm = Mock(return_value=['{"descriptors": [], "rules": []}'])
        lm.kwargs = {}
        with self.assertRaises(self.error):
            self.adapter(lm, {}, self.signature, [], self.inputs)
        lm.assert_called_once()
        self.assertEqual(set(lm.call_args.kwargs), {"messages"})

    def test_native_lm_defaults_are_rejected_without_mutation_or_transport(self):
        for field in ("response_format", "tools", "tool_choice"):
            with self.subTest(field=field):
                lm = Mock(return_value=[self.payload])
                lm.kwargs = {field: "configured-native-output", "max_tokens": 1024}
                original = copy.deepcopy(lm.kwargs)
                with self.assertRaisesRegex(ValueError, "native-output LM defaults"):
                    self.adapter(lm, {}, self.signature, [], self.inputs)
                lm.assert_not_called()
                self.assertEqual(lm.kwargs, original)

    def test_async_path_also_makes_one_plain_text_transport_call(self):
        lm = SimpleNamespace(kwargs={}, acall=AsyncMock(return_value=[self.payload]))
        result = asyncio.run(self.adapter.acall(
            lm, {"response_format": "forbidden"}, self.signature, [], self.inputs,
        ))
        self.assertEqual(result[0]["descriptors"][0].source_index, 0)
        lm.acall.assert_awaited_once()
        self.assertEqual(set(lm.acall.call_args.kwargs), {"messages"})


if __name__ == "__main__":
    unittest.main()
