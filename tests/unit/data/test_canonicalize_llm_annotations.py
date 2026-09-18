import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from recommendation.cli.canonicalize_llm_annotations import (
    build_frozen_canonical_snapshot, main,
)
from recommendation.features.text_embeddings import article_text


def _annotation(article, concepts):
    return {
        "concepts": concepts, "format": "news report", "event_types": [],
        "intents": ["informative"], "audiences": [],
        "provenance": {
            "article_content_sha256": hashlib.sha256(article_text(
                article["title"], article["abstract"]
            ).encode()).hexdigest(),
        },
    }


class FrozenCanonicalSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.articles = [
            {"id": "history", "title": "Artificial intelligence", "abstract": "AI research",
             "title_entities": ["Q11660"]},
            {"id": "train", "title": "AI systems", "abstract": "Technology"},
            {"id": "eval", "title": "XR systems", "abstract": "Extended reality"},
        ]
        self.source = {"articles": self.articles, "events": []}
        self.histories = {"events": [{
            "article": "train", "history": ["history"], "action": "click",
            "label": 1, "user": "secret-user",
        }]}
        records = {
            "history": _annotation(self.articles[0], ["artificial intelligence", "ai"]),
            "train": _annotation(self.articles[1], ["artificial intelligence", "ai"]),
            "eval": _annotation(self.articles[2], ["extended reality", "xr"]),
        }
        self.annotations = {
            "schema": "mindplex-llm-article-facts-v1", "model": "fixture",
            "prompt_sha256": "a" * 64, "progress": {},
            "extraction_records": records,
        }

    def test_registry_is_training_only_outcome_blind_and_qids_are_opt_in(self):
        first = build_frozen_canonical_snapshot(
            self.source, self.histories, self.annotations,
        )
        changed = copy.deepcopy(self.histories)
        changed["events"][0].update(action="skip", label=0, user="another")
        second = build_frozen_canonical_snapshot(
            self.source, changed, self.annotations,
        )
        self.assertEqual(first, second)
        registry = first["provenance"]["registry"]
        self.assertEqual(registry["acronym_aliases"], {"ai": "artificial intelligence"})
        self.assertNotIn("xr", registry["acronym_aliases"])
        self.assertNotIn("concept:entity/wikidata/q11660",
                         first["annotations"]["history"]["concepts"])
        self.assertFalse(first["provenance"]["registry_training_scope"]["evaluation_inputs_used"])

    def test_cli_is_exclusive_and_emits_projection_compatible_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("source.json", "histories.json", "annotations.json")]
            for path, value in zip(paths, (self.source, self.histories, self.annotations)):
                path.write_text(json.dumps(value), encoding="utf-8")
            output = root / "canonical.json"
            self.assertEqual(main([
                "--data", str(paths[0]), "--histories", str(paths[1]),
                "--annotations", str(paths[2]), "--output", str(output),
            ]), 0)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["schema"], "mindplex-canonical-article-annotations-v1")
            self.assertEqual(set(payload["annotations"]), {"history", "train", "eval"})
            before = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "new path"):
                main([
                    "--data", str(paths[0]), "--histories", str(paths[1]),
                    "--annotations", str(paths[2]), "--output", str(output),
                ])
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
