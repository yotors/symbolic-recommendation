import copy
import hashlib
import json
import unittest

from recommendation.features.concept_canonicalization import (
    CanonicalizationError, REGISTRY_SCHEMA, SCHEMA_VERSION,
    build_canonical_registry, canonicalize_annotations,
    lexical_form,
)
from recommendation.features.text_embeddings import article_text


def _record(article, concepts, *, format="news_report", event_types=None,
            intents=None, audiences=None):
    return {
        "concepts": list(concepts), "format": format,
        "event_types": list(event_types or []), "intents": list(intents or []),
        "audiences": list(audiences or []),
        "provenance": {
            "source_id": article.get("source_id", article["id"]),
            "article_content_sha256": hashlib.sha256(
                article_text(article.get("title"), article.get("abstract")).encode()
            ).hexdigest(),
            "model": "fixture",
        },
    }


class ConceptCanonicalizationTest(unittest.TestCase):
    def setUp(self):
        self.articles = [
            {
                "id": "a", "source_id": "source-a", "title": "AI and robots",
                "abstract": "Artificial intelligence research.",
                "title_entities": [{"WikidataId": "Q11660", "Label": "Artificial intelligence",
                                    "SurfaceForms": ["AI"]}],
                "abstract_entities": ["Q11012"],
                "label": 1, "clicks": 999,
                "article_entity_vector": [0.1, 0.2],
            },
            {
                "id": "b", "source_id": "source-b", "title": "Robot systems",
                "abstract": "Robots in factories.", "entities": ["robot"],
                "label": 0, "clicks": 0,
            },
        ]
        self.annotations = {
            "a": _record(self.articles[0], ["Artificial Intelligence", "AI", "Robots"],
                         event_types=["Product Launches"], intents=["Informative"]),
            "b": _record(self.articles[1], ["robot", "Artificial Intelligence", "AI"],
                         intents=["inform"]),
        }

    def test_normalization_aliases_and_collision_resistance(self):
        self.assertEqual(lexical_form("  Product_launches ", "event_types"), "product launches")
        self.assertEqual(lexical_form("NEWS REPORT", "format"), "report")
        self.assertEqual(lexical_form("Informative", "intents"), "inform")
    def test_entity_aliases_and_observed_acronyms_resolve_without_labels(self):
        registry = build_canonical_registry(self.annotations, self.articles)
        self.assertEqual(registry["schema"], REGISTRY_SCHEMA)
        self.assertEqual(registry["acronym_aliases"]["ai"], "artificial intelligence")
        self.assertEqual(registry["entity_aliases"]["ai"], "concept:entity/wikidata/q11660")
        result = canonicalize_annotations(self.annotations, self.articles, registry=registry)
        concepts = result["annotations"]["a"]["concepts"]
        self.assertIn("concept:entity/wikidata/q11660", concepts)
        self.assertIn("concept:entity/wikidata/q11012", concepts)
        ai_mappings = result["annotations"]["a"]["provenance"]["canonicalization"]["mappings"]["concepts"][:2]
        self.assertEqual({item["canonical_id"] for item in ai_mappings},
                         {"concept:entity/wikidata/q11660"})
        self.assertTrue(all(item["basis"].startswith(("explicit-entity", "observed-acronym"))
                            for item in ai_mappings))

    def test_ambiguous_global_entity_alias_is_not_merged_but_local_binding_is_safe(self):
        articles = copy.deepcopy(self.articles)
        articles.append({"id": "c", "title": "Another AI", "abstract": "",
                         "entities": [{"id": "Q999", "label": "AI"}]})
        annotations = copy.deepcopy(self.annotations)
        annotations["c"] = _record(articles[-1], ["AI"], format=None)
        registry = build_canonical_registry(annotations, articles)
        self.assertIn("ai", registry["ambiguous_entity_aliases"])
        self.assertNotIn("ai", registry["entity_aliases"])
        result = canonicalize_annotations(annotations, articles, registry=registry)
        self.assertIn("concept:entity/wikidata/q11660", result["annotations"]["a"]["concepts"])
        self.assertIn("concept:entity/wikidata/q999", result["annotations"]["c"]["concepts"])

    def test_deterministic_order_hashes_and_source_immutability(self):
        before = copy.deepcopy((self.annotations, self.articles))
        first = canonicalize_annotations(self.annotations, self.articles)
        second = canonicalize_annotations(
            dict(reversed(list(self.annotations.items()))), list(reversed(self.articles))
        )
        self.assertEqual(first, second)
        self.assertEqual((self.annotations, self.articles), before)
        self.assertEqual(first["schema"], SCHEMA_VERSION)
        self.assertEqual(list(first["annotations"]), ["a", "b"])
        output_hash = hashlib.sha256(json.dumps(
            first["annotations"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
        self.assertEqual(first["provenance"]["output_annotations_sha256"], output_hash)

    def test_behavior_fields_and_vectors_never_affect_registry_or_output(self):
        dataset_one = {"articles": copy.deepcopy(self.articles), "events": [{"action": "click"}],
                       "tests": [{"labels": {"a": 1}}]}
        dataset_two = copy.deepcopy(dataset_one)
        dataset_two["events"] = [{"action": "skip", "score": 123}]
        dataset_two["tests"] = [{"labels": {"a": 0}}]
        for article in dataset_two["articles"]:
            article["clicks"] = -100
            article["label"] = 999
            article["article_entity_vector"] = [99.0, -99.0]
        first = canonicalize_annotations(self.annotations, dataset_one)
        second = canonicalize_annotations(self.annotations, dataset_two)
        self.assertEqual(first, second)
        self.assertFalse(first["provenance"]["configuration"]["vectors_used"])
        self.assertEqual(first["provenance"]["configuration"]["behavioral_fields_used"], [])
        bad = copy.deepcopy(self.annotations)
        bad["a"]["click"] = True
        with self.assertRaisesRegex(CanonicalizationError, "non-content fields"):
            canonicalize_annotations(bad, self.articles)
        bad = copy.deepcopy(self.annotations)
        bad["a"]["provenance"]["label"] = 1
        with self.assertRaisesRegex(CanonicalizationError, "behavioral fields"):
            canonicalize_annotations(bad, self.articles)

    def test_frozen_registry_is_reusable_for_unseen_articles(self):
        registry = build_canonical_registry(self.annotations, self.articles)
        unseen_article = {"id": "new", "title": "AI systems", "abstract": ""}
        unseen = {"new": _record(unseen_article, ["AI"], format="analysis")}
        result = canonicalize_annotations(unseen, [unseen_article], registry=registry,
                                          include_entity_concepts=False)
        self.assertEqual(result["provenance"]["registry_source"], "supplied-frozen")
        self.assertIn("concept:entity/wikidata/q11660",
                      result["annotations"]["new"]["concepts"])

    def test_entity_limit_is_deterministic_and_explicit(self):
        article = {"id": "many", "title": "Many", "abstract": "",
                   "title_entities": ["Q3", "Q1", "Q2"]}
        annotations = {"many": _record(article, [], format=None)}
        result = canonicalize_annotations(annotations, [article], max_entity_concepts=2)
        self.assertEqual(result["annotations"]["many"]["concepts"],
                         ["concept:entity/wikidata/q1", "concept:entity/wikidata/q3"])
        self.assertEqual(result["provenance"]["statistics"]["entity_concepts_dropped_by_limit"], 1)

    def test_content_identity_unknown_ids_duplicate_ids_and_registry_are_strict(self):
        bad = copy.deepcopy(self.annotations)
        bad["a"]["provenance"]["article_content_sha256"] = "0" * 64
        with self.assertRaisesRegex(CanonicalizationError, "stale annotation"):
            canonicalize_annotations(bad, self.articles)
        bad = {"missing": copy.deepcopy(self.annotations["a"])}
        with self.assertRaisesRegex(CanonicalizationError, "absent from source"):
            canonicalize_annotations(bad, self.articles)
        with self.assertRaisesRegex(CanonicalizationError, "duplicate article ID"):
            canonicalize_annotations(self.annotations, [*self.articles, self.articles[0]])
        registry = build_canonical_registry(self.annotations, self.articles)
        registry["unexpected"] = True
        with self.assertRaisesRegex(CanonicalizationError, "registry fields"):
            canonicalize_annotations(self.annotations, self.articles, registry=registry)

    def test_extractor_envelope_supported_without_losing_source_provenance(self):
        envelope = {
            "schema": "mindplex-llm-article-facts-v1", "model": "fixture",
            "prompt_sha256": "a" * 64, "progress": {"available_articles": 2},
            "extraction_records": self.annotations,
        }
        result = canonicalize_annotations(envelope, self.articles)
        source = result["provenance"]["annotation_source"]
        self.assertNotIn("extraction_records", source)
        self.assertEqual(source["model"], "fixture")


if __name__ == "__main__":
    unittest.main()
