from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from recommendation.integrations import llm_article_facts as facts
from recommendation.features.text_embeddings import article_text


def draft_fact(index=0, predicate="HasConcept", value="space exploration", **changes):
    result = {
        "kind": "fact", "atom": {"predicate": predicate, "arguments": [
            {"kind": "symbol", "value": f"article_{index}"}, {"kind": "string", "value": value},
        ]}, "truth": {"strength": 0.8, "confidence": 0.6},
        "source_sentence_indexes": [index],
    }
    result.update(changes)
    return result


class FakeBackend:
    name = "test/content-extractor"

    def __init__(self, drafts):
        self.drafts = list(drafts)
        self.requests = []

    def ready(self):
        return True

    def translate(self, request, *, feedback=""):
        self.requests.append(request.model_dump(mode="json"))
        result = self.drafts.pop(0)
        if isinstance(result, Exception):
            print("SECRET_PROVIDER_MESSAGE")
            raise result
        return facts._library().models.TranslationDraft.model_validate(result), facts._library().models.Usage(input_tokens=10, output_tokens=5)


@unittest.skipUnless(importlib.util.find_spec("dspy"), "NL2PLN integration tests require existing NL2PLN/.venv")
class LlmArticleFactsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=facts.WORKSPACE_ROOT, prefix=".test-llm-facts-")
        self.root = Path(self.temporary.name)
        self.cache = self.root / "annotations.json"
        self.articles = [
            {"id": "item_A", "source_id": "A", "title": "A spacecraft launch", "abstract": "A new space mission.", "label": 1, "user": "PRIVATE_USER"},
            {"id": "item_B", "source_id": "B", "title": "An astronomy explainer", "abstract": "", "label": 0},
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def extract(self, drafts, articles=None, **kwargs):
        backend = FakeBackend(drafts)
        service = facts.make_translation_service(backend)
        result = facts.extract_article_records(
            self.articles if articles is None else articles,
            cache_path=self.cache, service=service, **kwargs,
        )
        return result, backend

    def test_real_typed_compiler_provenance_and_article_scoped_fields(self):
        source = copy.deepcopy(self.articles)
        result, backend = self.extract([{"statements": [
            draft_fact(), draft_fact(0, "HasFormat", "report"),
            draft_fact(1, "HasFormat", "explainer"), draft_fact(1, "HasAudience", "astronomy learners"),
        ]}])
        records = result["extraction_records"]
        self.assertEqual(records["item_A"]["concepts"], ["space exploration"])
        self.assertEqual(records["item_A"]["format"], "report")
        self.assertEqual(records["item_B"]["concepts"], [])
        self.assertEqual(records["item_B"]["audiences"], ["astronomy learners"])
        provenance = records["item_A"]["provenance"]
        self.assertEqual(provenance["article_content_sha256"], hashlib.sha256(article_text(source[0]["title"], source[0]["abstract"]).encode()).hexdigest())
        self.assertEqual(provenance["source_id"], "A")
        self.assertEqual(provenance["source_truth_values"][0], {"strength": 0.8, "confidence": 0.6})
        self.assertIn("not calibrated", provenance["truth_value_interpretation"])
        self.assertIn('(HasConcept "item_A" "space exploration")', " ".join(provenance["anchored_statements"]))
        transmitted = json.dumps(backend.requests)
        self.assertNotIn("PRIVATE_USER", transmitted)
        self.assertNotIn('"label"', transmitted)
        self.assertNotIn("item_A", transmitted)
        self.assertEqual(self.articles, source)
        self.assertEqual(result["progress"]["input_tokens"], 10)

    def test_resume_uses_cache_and_content_identity_not_user_or_label(self):
        first, _ = self.extract([{"statements": [draft_fact(), draft_fact(1)]}])
        changed = copy.deepcopy(self.articles)
        changed[0]["label"] = 0
        changed[0]["user"] = "ANOTHER_USER"
        repeated, backend = self.extract([], changed, max_calls=0, max_new_articles=0)
        self.assertEqual(backend.requests, [])
        self.assertEqual(first["extraction_records"], repeated["extraction_records"])
        changed[0]["title"] += " "  # Exact input text changes even if normalized text does not.
        missing, _ = self.extract([], changed, max_calls=0)
        self.assertNotIn("item_A", missing["extraction_records"])
        self.assertIn("item_B", missing["extraction_records"])

    def test_duplicate_content_reanchors_to_each_safe_caller_id(self):
        duplicate = copy.deepcopy(self.articles[0])
        duplicate.update(id='odd ")(eval evil)', source_id="DUP")
        result, backend = self.extract([{"statements": [draft_fact()]}], [self.articles[0], duplicate])
        self.assertEqual(len(backend.requests[0]["sentences"]), 1)
        self.assertEqual(result["progress"]["new_articles"], 1)
        record = result["extraction_records"][duplicate["id"]]
        self.assertEqual(record["provenance"]["source_id"], "DUP")
        for source in record["provenance"]["anchored_statements"]:
            facts._library().contract.validate_statement(source)
        self.assertEqual(record["concepts"], ["space exploration"])

    def test_missing_facts_and_empty_text_stay_absent(self):
        result, _ = self.extract([{"statements": [draft_fact()]}])
        other = result["extraction_records"]["item_B"]
        self.assertEqual(other["provenance"]["status"], "no_facts")
        self.assertIsNone(other["format"])
        self.assertEqual(other["concepts"], [])
        empty = [{"id": "empty", "title": "", "abstract": None}]
        result, backend = self.extract([], empty, max_calls=0)
        self.assertEqual(backend.requests, [])
        self.assertEqual(result["extraction_records"]["empty"]["provenance"]["status"], "empty_text")

    def test_all_empty_model_draft_is_no_facts_not_a_dummy_rule(self):
        result, _ = self.extract([{"statements": [], "warnings": ["No supported descriptors"]}])
        for record in result["extraction_records"].values():
            self.assertEqual(record["provenance"]["status"], "no_facts")
            self.assertEqual(record["provenance"]["source_statements"], [])
            self.assertEqual(record["provenance"]["warnings"], ["No supported descriptors"])

    def test_rules_variables_foreign_predicates_and_bad_provenance_rejected(self):
        bad_facts = [
            draft_fact(source_sentence_indexes=[]),
            draft_fact(source_sentence_indexes=[0, 1]),
            draft_fact(source_sentence_indexes=[1]),
            draft_fact(predicate="Engagement"),
            draft_fact(0, "HasFormat", "unknown"),
            draft_fact(0, "HasFormat", ""),
            draft_fact(value="line\nbreak"),
        ]
        variable = draft_fact()
        variable["atom"]["arguments"][0] = {"kind": "variable", "value": "$x"}
        bad_facts.append(variable)
        rule = {"kind": "rule", "premises": [draft_fact()["atom"]], "conclusions": [draft_fact()["atom"]], "source_sentence_indexes": [0]}
        bad_facts.append(rule)
        for bad in bad_facts:
            with self.subTest(bad=bad), self.assertRaises(facts.ArticleExtractionError):
                self.extract([{"statements": [bad]}])
        self.assertFalse(self.cache.exists())

    def test_nonfinite_truth_and_conflicting_formats_are_rejected(self):
        for bad in (draft_fact(truth={"strength": math.nan, "confidence": 1}),
                    draft_fact(truth={"strength": 1, "confidence": math.inf})):
            with self.assertRaises(facts.ArticleExtractionError):
                self.extract([{"statements": [bad]}])
        with self.assertRaisesRegex(facts.ArticleExtractionError, "conflicting"):
            self.extract([{"statements": [draft_fact(0, "HasFormat", "review"), draft_fact(0, "HasFormat", "opinion")]}])
        self.assertFalse(self.cache.exists())

    def test_call_and_article_caps_bound_work(self):
        articles = [{"id": f"x{index}", "title": f"Unique article {index}"} for index in range(7)]
        result, backend = self.extract([{"statements": [draft_fact()]}], articles,
                                       batch_size=2, max_new_articles=5, max_calls=1)
        self.assertEqual(len(backend.requests), 1)
        self.assertEqual(result["progress"]["new_articles"], 2)
        self.assertEqual(result["progress"]["unprocessed_articles"], 5)

    def test_failed_later_batch_preserves_prior_cache_and_suppresses_provider_details(self):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), self.assertRaises(facts.ArticleExtractionError) as caught:
            self.extract([{"statements": [draft_fact()]}, RuntimeError("SECRET_PROVIDER_MESSAGE")], batch_size=1, max_calls=2)
        self.assertNotIn("SECRET_PROVIDER_MESSAGE", str(caught.exception) + stdout.getvalue() + stderr.getvalue())
        resumed, backend = self.extract([], max_calls=0)
        self.assertIn("item_A", resumed["extraction_records"])
        self.assertNotIn("item_B", resumed["extraction_records"])
        self.assertEqual(backend.requests, [])

    def test_existing_historical_file_and_cache_tampering_are_not_overwritten(self):
        self.cache.write_text('{"historical":"experiment"}', encoding="utf-8")
        before = self.cache.read_bytes()
        with self.assertRaisesRegex(facts.ArticleExtractionError, "not be overwritten"):
            self.extract([])
        self.assertEqual(self.cache.read_bytes(), before)
        self.cache.unlink()
        self.extract([{"statements": [draft_fact()]}])
        cached = json.loads(self.cache.read_text())
        next(iter(cached["entries"].values()))["record"]["concepts"].append("tampered")
        self.cache.write_text(json.dumps(cached), encoding="utf-8")
        before = self.cache.read_bytes()
        with self.assertRaises(facts.ArticleExtractionError):
            self.extract([])
        self.assertEqual(self.cache.read_bytes(), before)

    def test_provider_settings_are_allowlisted_and_do_not_mutate_environment(self):
        for folder in ("PeTTaChainer", "NL2PLN"):
            (self.root / folder).mkdir()
        (self.root / "PeTTaChainer/.env").write_text("AWS_REGION=stack-region\nUNRELATED_SECRET=not-read\nAWS_ACCESS_KEY_ID=stack-key\n", encoding="utf-8")
        (self.root / "NL2PLN/.env").write_text("AWS_REGION=parser-region\nNL2PC_BEDROCK_MODEL_ID=parser-model\n", encoding="utf-8")
        with mock.patch.object(facts, "WORKSPACE_ROOT", self.root), mock.patch.dict(os.environ, {"AWS_REGION": "explicit-region"}, clear=True):
            before = dict(os.environ)
            result = facts._provider_settings()
            self.assertEqual(result["AWS_REGION"], "explicit-region")
            self.assertNotIn("UNRELATED_SECRET", result)
            self.assertEqual(os.environ, before)

    def test_cache_outside_workspace_and_invalid_caps_fail_before_provider(self):
        backend = FakeBackend([])
        with self.assertRaises(facts.ArticleExtractionError):
            facts.extract_article_records(self.articles, cache_path="/tmp/not-this-workspace.json", service=facts.make_translation_service(backend))
        for kwargs in ({"max_calls": -1}, {"batch_size": 9}, {"max_new_articles": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(facts.ArticleExtractionError):
                self.extract([], **kwargs)
        self.assertEqual(backend.requests, [])

    def test_real_factory_counts_actual_lm_calls_and_keeps_credentials_in_memory(self):
        import dspy
        settings = {"NL2PC_BEDROCK_MODEL_ID": "test-model", "AWS_ACCESS_KEY_ID": "test-key",
                    "AWS_SECRET_ACCESS_KEY": "test-secret", "AWS_REGION": "test-region"}
        response = SimpleNamespace(usage={"prompt_tokens": 31, "completion_tokens": 8192},
                                   choices=[{"finish_reason": "length", "message": "PRIVATE_RAW_TEXT"}])
        before = dict(os.environ)
        with mock.patch.object(facts, "_provider_settings", return_value=settings), \
                mock.patch.object(dspy.LM, "forward", return_value=response) as forward:
            service = facts.build_real_service(timeout_seconds=23, max_calls=1)
            backend = service.backend.inner
            self.assertIs(backend._lm.forward(messages=[]), response)
            with self.assertRaisesRegex(facts.ArticleExtractionError, "budget exhausted"):
                backend._lm.forward(messages=[])
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(backend.provider_calls, 1)
            self.assertEqual(backend.provider_usage, {"input_tokens": 31, "output_tokens": 8192,
                                                       "responses_with_usage": 1, "finish_reasons": ["length"]})
            self.assertNotIn("PRIVATE_RAW_TEXT", json.dumps(backend.provider_usage))
            self.assertEqual(backend._lm.num_retries, 0)
            self.assertFalse(backend._lm.cache)
            self.assertFalse(dspy.cache.enable_disk_cache)
            self.assertFalse(dspy.cache.enable_memory_cache)
            self.assertEqual(backend._lm.kwargs["timeout"], 23)
            self.assertEqual(backend._lm.kwargs["aws_access_key_id"], "test-key")
            self.assertIn("untrusted article content", backend._predict.signature.instructions)
            self.assertEqual(service.repair_attempts, 0)
        self.assertEqual(os.environ, before)

    def test_safe_failure_details_allowlist_codes_but_never_messages(self):
        ClientError = type("ClientError", (RuntimeError,), {})
        inner = ClientError("SECRET authentication payload")
        inner.response = {"Error": {"Code": "AccessDeniedException", "Message": "SECRET"},
                          "ResponseMetadata": {"HTTPStatusCode": 403, "RequestId": "SECRET"}}
        outer = RuntimeError("SECRET outer message")
        outer.__cause__ = inner
        with self.assertRaises(facts.ArticleExtractionError) as caught:
            self.extract([outer])
        details = caught.exception.details
        self.assertIn("ClientError", details["exception_types"])
        self.assertEqual(details["aws_codes"], ["AccessDeniedException"])
        self.assertEqual(details["http_statuses"], [403])
        self.assertNotIn("SECRET", json.dumps(details))
        inner.response["Error"]["Code"] = "SECRET-unrecognized-code"
        self.assertEqual(facts._safe_exception_details(inner)["aws_codes"], [])

    def test_cancellation_is_checked_before_a_provider_request(self):
        backend = FakeBackend([])
        with self.assertRaisesRegex(facts.ArticleExtractionError, "cancelled"):
            facts.extract_article_records(self.articles, cache_path=self.cache,
                service=facts.make_translation_service(backend), should_stop=lambda: True)
        self.assertEqual(backend.requests, [])

    @staticmethod
    def _invalid_compact_output():
        from recommendation.integrations.llm_compact_backend import _descriptor_model
        from nl2pettachainer.backends.base import BackendError
        from pydantic import ValidationError
        try:
            _descriptor_model().model_validate({
                "source_index": "PRIVATE_INVALID_INPUT", "concepts": [], "format": None,
                "event_types": [], "intents": [], "audiences": [],
                "PRIVATE_EXTRA_FIELD": "PRIVATE_EXTRA_VALUE",
            })
        except ValidationError as exc:
            error = BackendError("PRIVATE_PROVIDER_ERROR")
            error.__cause__ = exc
            return error
        raise AssertionError("fixture must fail strict descriptor validation")

    @contextmanager
    def _factory_sequence(self, outcomes, *, max_calls=2):
        import dspy
        from recommendation.integrations import llm_compact_backend as compact
        calls, remaining = [], list(outcomes)
        response = SimpleNamespace(usage={"prompt_tokens": 31, "completion_tokens": 7},
                                   choices=[{"finish_reason": "stop"}])

        def translate(backend, request, feedback):
            calls.append({"lm": backend._lm, "feedback": feedback})
            backend._lm.forward(messages=[])
            outcome = remaining.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return (facts._library().models.TranslationDraft.model_validate({
                "statements": [draft_fact()],
            }), facts._library().models.Usage(input_tokens=31, output_tokens=7))

        with mock.patch.object(facts, "_provider_settings", return_value={}), \
                mock.patch.object(dspy.LM, "forward", return_value=response) as forward, \
                mock.patch.object(compact, "translate_compact", side_effect=translate):
            service = facts.build_real_service(model="bedrock/test-model", max_calls=max_calls)
            yield service, calls, forward

    def test_real_factory_regenerates_one_invalid_output_with_same_budgeted_lm(self):
        facts._library()
        before = facts.prompt_sha256()
        with self._factory_sequence([self._invalid_compact_output(), None]) as (service, calls, forward):
            result = facts.extract_article_records(self.articles, cache_path=self.cache,
                                                  service=service, max_calls=2)
            self.assertEqual(forward.call_count, 2)
            self.assertIs(calls[0]["lm"], calls[1]["lm"])
            self.assertEqual(calls[0]["feedback"], "")
            self.assertEqual(calls[1]["feedback"], facts.REGENERATION_FEEDBACK)
            self.assertEqual(result["progress"]["provider_calls"], 2)
            self.assertEqual(result["progress"]["batch_calls"], 1)
            self.assertEqual(result["progress"]["input_tokens"], 62)
            self.assertEqual(result["progress"]["output_tokens"], 14)
            stats = result["extraction_records"]["item_A"]["provenance"]["extraction_attempts"]
            self.assertEqual((stats["attempts"], stats["regenerations"], stats["provider_calls"]), (2, 1, 2))
            self.assertEqual(stats["policy"]["max_regenerations"], 1)
            self.assertEqual(stats["policy"]["implementation_sha256"],
                             hashlib.sha256(Path(facts.__file__).read_bytes()).hexdigest())
            self.assertNotIn("PRIVATE", json.dumps(stats))
            self.assertEqual(service.backend.inner.call_limit, 2)
        self.assertEqual(before, facts.prompt_sha256())

    def test_two_invalid_outputs_stop_without_using_remaining_authorized_calls(self):
        facts._library()
        with self._factory_sequence([self._invalid_compact_output(), self._invalid_compact_output(), None],
                                   max_calls=5) as (service, calls, forward):
            with self.assertRaises(facts.ArticleExtractionError) as caught:
                facts.extract_article_records(self.articles, cache_path=self.cache,
                                              service=service, max_calls=5)
            self.assertEqual(forward.call_count, 2)
            self.assertEqual(len(calls), 2)
            self.assertEqual(service.backend.inner.call_limit, 5)
            self.assertEqual(caught.exception.details["provider_calls"], 2)
            self.assertEqual(caught.exception.details["provider_usage"]["input_tokens"], 62)
            stats = caught.exception.details["extraction_attempts"]
            self.assertEqual(stats["attempts"], 2)
            self.assertEqual(len(stats["validation_failures"]), 2)
            self.assertNotIn("PRIVATE", str(caught.exception) + json.dumps(caught.exception.details))
            self.assertFalse(self.cache.exists())

    def test_http_auth_rate_and_timeout_failures_never_regenerate(self):
        facts._library()
        for name, status in (("HTTPError", 503), ("AuthenticationError", 401),
                             ("RateLimitError", 429), ("APITimeoutError", None)):
            error = type(name, (RuntimeError,), {})("PRIVATE_PROVIDER_MESSAGE")
            if status is not None:
                error.status_code = status
            # A nested validation failure must not make a provider failure
            # eligible for regeneration.
            error.__cause__ = self._invalid_compact_output()
            with self.subTest(error=name), self._factory_sequence([error, None]) as (service, calls, forward):
                with self.assertRaises(facts.ArticleExtractionError) as caught:
                    facts.extract_article_records(self.articles, cache_path=self.cache,
                                                  service=service, max_calls=2)
                self.assertEqual(forward.call_count, 1)
                self.assertEqual(len(calls), 1)
                self.assertEqual(caught.exception.details["provider_calls"], 1)
                self.assertEqual(caught.exception.details["extraction_attempts"]["regenerations"], 0)

    def test_one_call_budget_returns_original_validation_error_without_recharging(self):
        facts._library()
        failure = self._invalid_compact_output()
        with self._factory_sequence([failure, None], max_calls=1) as (service, calls, forward):
            request = facts._library().models.TranslateRequest(sentences=["one article"])
            with self.assertRaises(type(failure)) as caught:
                service.backend.inner.translate(request)
            self.assertIs(caught.exception, failure)
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(service.backend.inner.call_limit, 1)
            self.assertEqual(service.backend.last_translation_stats["regenerations"], 0)

    def test_validation_diagnostics_exclude_inputs_messages_context_and_unknown_field_names(self):
        facts._library()
        details = facts._safe_exception_details(self._invalid_compact_output())
        self.assertIn({"type": "int_type", "location": ["source_index"]}, details["validation_errors"])
        self.assertIn({"type": "extra_forbidden", "location": ["<field>"]}, details["validation_errors"])
        self.assertNotIn("PRIVATE", json.dumps(details))
        for diagnostic in details["validation_errors"]:
            self.assertEqual(set(diagnostic), {"type", "location"})

    def test_retry_policy_fingerprint_does_not_invalidate_existing_content_cache(self):
        first, _ = self.extract([{"statements": [draft_fact(), draft_fact(1)]}])
        cache_bytes, prompt_hash = self.cache.read_bytes(), facts.prompt_sha256()
        with mock.patch.object(facts, "REGENERATION_FEEDBACK", "Different operational retry instruction"):
            self.assertEqual(facts.prompt_sha256(), prompt_hash)
            resumed, backend = self.extract([], max_calls=0, max_new_articles=0)
        self.assertEqual(backend.requests, [])
        self.assertEqual(self.cache.read_bytes(), cache_bytes)
        self.assertEqual(resumed["extraction_records"], first["extraction_records"])


if __name__ == "__main__":
    unittest.main()
