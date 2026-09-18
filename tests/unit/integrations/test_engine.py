import json
import unittest
from urllib.error import URLError
from unittest.mock import patch

from recommendation.integrations.engine import (
    EXPECTED_NL2PLN_CONTRACT,
    MAX_SEMANTIC_INPUT_CHARS,
    MAX_SEMANTIC_RESPONSE_BYTES,
    PeTTaChainerClient,
    PeTTaChainerConfigurationError,
    PeTTaChainerInputError,
    PeTTaChainerProtocolError,
    PeTTaChainerTimeoutError,
    PeTTaChainerUpstreamError,
    RECOMMENDATION_PREDICATE_SCHEMA,
)


class _Response:
    def __init__(self, payload=None, raw=None):
        self.payload = raw if raw is not None else json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, maximum=None):
        return self.payload if maximum is None else self.payload[:maximum]


def _converter(**overrides):
    result = {
        "contract_version": EXPECTED_NL2PLN_CONTRACT,
        "model": "semantic-test-model",
        "namespace": "kb_test",
        "warnings": [],
        "cached": False,
        "attempts": 1,
        "usage": {"input_tokens": 5, "output_tokens": 7},
    }
    result.update(overrides)
    return result


def _preview(**overrides):
    result = {
        "knowledge_base_id": "kb-1",
        "conversion": "nl2pln",
        "committed": False,
        "statements": [{
            "source": "(: s (HasTopic article ai) (STV 1 1))",
            "idempotency_key": "semantic-s-v1",
            "mine_patterns": False,
        }],
        "converter": _converter(),
    }
    result.update(overrides)
    return result


class SemanticClientTests(unittest.TestCase):
    def client(self, **overrides):
        values = {
            "base_url": "http://semantic.test",
            "api_key": "secret",
            "knowledge_base": "kb-1",
            "timeout": 12,
        }
        values.update(overrides)
        return PeTTaChainerClient(**values)

    def test_article_preview_uses_bounded_nl2pln_contract(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return _Response(_preview())

        with patch("recommendation.integrations.engine.urlopen", fake_urlopen):
            result = self.client().parse_article({
                "id": "n1", "title": "AI changes science", "abstract": "A study."
            })

        self.assertEqual(
            captured["url"],
            "http://semantic.test/v1/knowledge-bases/kb-1/knowledge",
        )
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(captured["body"]["conversion"], "nl2pln")
        self.assertTrue(captured["body"]["preview"])
        self.assertEqual(
            captured["body"]["predicate_schema"],
            list(RECOMMENDATION_PREDICATE_SCHEMA),
        )
        self.assertIn("Title: AI changes science", captured["body"]["text"])
        self.assertEqual(result["article_id"], "n1")
        self.assertEqual(result["statement_count"], 1)
        self.assertFalse(result["committed"])
        self.assertEqual(result["converter"], _converter())
        self.assertEqual(result["converter_provenance"], {
            "contract_version": EXPECTED_NL2PLN_CONTRACT,
            "model": "semantic-test-model",
            "namespace": "kb_test",
        })

    def test_valid_empty_statement_preview_is_preserved(self):
        converted = _preview(statements=[], statement_count=0)
        with patch("recommendation.integrations.engine.urlopen", return_value=_Response(converted)):
            result = self.client().parse_article({"id": "n1", "title": "No extracted facts"})
        self.assertEqual(result["statements"], [])
        self.assertEqual(result["statement_count"], 0)

    def test_article_requires_parseable_bounded_text(self):
        client = self.client()
        with self.assertRaisesRegex(PeTTaChainerInputError, "no title or abstract"):
            client.parse_article({"id": "n1"})
        with self.assertRaisesRegex(PeTTaChainerInputError, "character semantic input limit"):
            client.parse_article({"id": "n1", "title": "x" * MAX_SEMANTIC_INPUT_CHARS})

    def test_loopback_secret_fallback_uses_exact_parsed_hostname(self):
        with patch("recommendation.integrations.engine._local_stack_api_secret", return_value="local-secret") as secret:
            local = self.client(base_url="http://127.0.0.2:8000", api_key=None)
        self.assertEqual(local.api_key, "local-secret")
        secret.assert_called_once_with()

        with (
            patch("recommendation.integrations.engine._local_stack_api_secret", return_value="must-not-leak") as secret,
            patch("recommendation.integrations.engine._setting", return_value=""),
        ):
            confusing = self.client(base_url="http://localhost.evil.example", api_key=None)
        self.assertEqual(confusing.api_key, "")
        secret.assert_not_called()

    def test_base_url_rejects_userinfo_and_hostname_confusion(self):
        for value in (
            "http://localhost@evil.example:8000",
            "http://owner:password@localhost:8000",
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(PeTTaChainerConfigurationError, "userinfo"):
                    self.client(base_url=value)

    def test_timeout_must_be_finite_and_bounded(self):
        for value in (float("nan"), float("inf"), -1, 0, 301, True, "forever"):
            with self.subTest(value=value):
                with self.assertRaises(PeTTaChainerConfigurationError):
                    self.client(timeout=value)

    def test_unconfigured_credentials_have_typed_errors(self):
        with self.assertRaisesRegex(PeTTaChainerConfigurationError, "API_KEY"):
            self.client(api_key="").parse_text("A bounded article")
        with self.assertRaisesRegex(PeTTaChainerConfigurationError, "KB_ID"):
            self.client(knowledge_base="").parse_text("A bounded article")

    def test_timeout_and_unavailable_upstream_have_distinct_types(self):
        with patch("recommendation.integrations.engine.urlopen", side_effect=TimeoutError("private detail")):
            with self.assertRaisesRegex(PeTTaChainerTimeoutError, "timed out"):
                self.client().parse_text("A bounded article")
        with patch("recommendation.integrations.engine.urlopen", side_effect=URLError("private detail")):
            with self.assertRaisesRegex(PeTTaChainerUpstreamError, "unavailable"):
                self.client().parse_text("A bounded article")

    def test_response_size_and_json_shape_are_bounded(self):
        cases = (
            (_Response(raw=b"x" * (MAX_SEMANTIC_RESPONSE_BYTES + 1)), "byte limit"),
            (_Response(raw=b"not-json"), "invalid JSON"),
            (_Response(payload=["not", "an", "object"]), "JSON object"),
        )
        for response, message in cases:
            with self.subTest(message=message):
                with patch("recommendation.integrations.engine.urlopen", return_value=response):
                    with self.assertRaisesRegex(PeTTaChainerProtocolError, message):
                        self.client().parse_text("A bounded article")

    def test_preview_must_be_explicitly_non_committing_nl2pln(self):
        invalid = (
            (_preview(committed=True), "non-committing"),
            (_preview(committed=0), "non-committing"),
            (_preview(conversion="metadata"), "nl2pln"),
            (_preview(knowledge_base_id="another-kb"), "another knowledge base"),
            (_preview(knowledge_base_id=None), "knowledge base identity"),
            (_preview(converter=None), "converter provenance"),
        )
        for payload, message in invalid:
            with self.subTest(message=message):
                with patch("recommendation.integrations.engine.urlopen", return_value=_Response(payload)):
                    with self.assertRaisesRegex(PeTTaChainerProtocolError, message):
                        self.client().parse_text("A bounded article")

    def test_statement_shape_and_declared_count_are_validated(self):
        invalid = (
            (_preview(statements="not-a-list"), "must be a list"),
            (_preview(statements=[{"source": ""}]), "statement source"),
            (_preview(statement_count=2), "count does not match"),
            (_preview(statements=[{"source": "(: s (P a) (STV 1 1))", "mine_patterns": "false"}]),
             "mining flag"),
        )
        for payload, message in invalid:
            with self.subTest(message=message):
                with patch("recommendation.integrations.engine.urlopen", return_value=_Response(payload)):
                    with self.assertRaisesRegex(PeTTaChainerProtocolError, message):
                        self.client().parse_text("A bounded article")

    def test_converter_contract_and_provenance_are_validated(self):
        invalid = (
            (_preview(converter={}), "contract_version"),
            (_preview(converter=_converter(contract_version="unknown.v2")), "unsupported"),
            (_preview(converter=_converter(model="")), "model"),
            (_preview(converter=_converter(attempts=0)), "attempt"),
            (_preview(converter=_converter(usage=[])), "usage"),
        )
        for payload, message in invalid:
            with self.subTest(message=message):
                with patch("recommendation.integrations.engine.urlopen", return_value=_Response(payload)):
                    with self.assertRaisesRegex(PeTTaChainerProtocolError, message):
                        self.client().parse_text("A bounded article")

    def test_predicate_schema_is_bounded_before_request(self):
        duplicate = [dict(RECOMMENDATION_PREDICATE_SCHEMA[0])] * 2
        with self.assertRaisesRegex(PeTTaChainerInputError, "duplicate"):
            self.client().parse_text("A bounded article", duplicate)


if __name__ == "__main__":
    unittest.main()
