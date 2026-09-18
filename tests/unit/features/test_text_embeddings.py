import builtins
import gzip
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from recommendation.features.text_embeddings import (
    SIDECAR_SCHEMA,
    TextEmbeddingError,
    article_text,
    build_text_embedding_sidecar,
    load_text_embedding_sidecar,
    read_article_corpus,
)
from recommendation.tests.fixtures import FakeEncoder


class TextEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_replay(self, articles=None):
        path = self.root / "replay.json.gz"
        payload = {
            "articles": articles or [
                {
                    "id": "article_N2",
                    "source_id": "N2",
                    "title": "  Second   title ",
                    "abstract": "Second abstract",
                },
                {
                    "id": "article_N1",
                    "source_id": "N1",
                    "title": "First title",
                    "abstract": "",
                },
            ],
            "metadata": {"dataset": "MIND-small", "projection": "test"},
        }
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_article_text_has_stable_normalization(self):
        self.assertEqual(
            article_text("  A\u212b   title ", " an\nabstract "),
            "A\u00c5 title\n\nan abstract",
        )

    def test_reads_replay_with_both_id_namespaces_and_canonical_order(self):
        corpus = read_article_corpus(self.write_replay())
        self.assertEqual(
            [(item.article_id, item.source_id) for item in corpus.articles],
            [("article_N1", "N1"), ("article_N2", "N2")],
        )
        self.assertEqual(corpus.articles[1].text, "Second title\n\nSecond abstract")
        self.assertEqual(corpus.source["kind"], "mind-replay-cache")
        self.assertEqual(len(corpus.content_sha256), 64)

    def test_reads_plain_and_zipped_reczoo_tsv(self):
        content = (
            "news_id\tcat\tsub_cat\ttitle_entities\tabstract_entities\ttitle\tabstract\n"
            "N2\tnews\tworld\t\t\tTwo\tAbstract two\n"
            "N1\ttech\tai\t\t\tOne\tAbstract one\n"
        )
        tsv = self.root / "news_corpus.tsv"
        tsv.write_text(content, encoding="utf-8")
        archive = self.root / "reczoo.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("nested/news_corpus.tsv", content)
        plain = read_article_corpus(tsv)
        zipped = read_article_corpus(archive)
        self.assertEqual(plain.articles, zipped.articles)
        self.assertEqual(plain.content_sha256, zipped.content_sha256)
        self.assertEqual(plain.articles[0].article_id, "N1")
        self.assertEqual(zipped.source["member"], "nested/news_corpus.tsv")

    def test_build_normalizes_vectors_and_records_provenance(self):
        encoder = FakeEncoder([[0.0, 3.0], [4.0, 0.0]])
        output = self.root / "vectors.npz"
        metadata = build_text_embedding_sidecar(
            self.write_replay(),
            output,
            encoder=encoder,
            model_name="fake/model",
            model_revision="abc123",
            batch_size=7,
        )
        self.assertTrue(output.is_file())
        self.assertFalse(any(output.parent.glob(f".{output.name}.*.tmp")))
        self.assertEqual(metadata["schema"], SIDECAR_SCHEMA)
        self.assertEqual(metadata["model"]["name"], "fake/model")
        self.assertEqual(metadata["model"]["revision"], "abc123")
        self.assertIsNone(metadata["model"]["resolved_revision"])
        self.assertEqual(metadata["dimensions"], 2)
        self.assertEqual(len(metadata["content_sha256"]), 64)
        self.assertEqual(len(metadata["vector_sha256"]), 64)
        self.assertEqual(
            encoder.calls[0][0], ["First title", "Second title\n\nSecond abstract"]
        )
        self.assertEqual(encoder.calls[0][1]["batch_size"], 7)
        self.assertFalse(encoder.calls[0][1]["normalize_embeddings"])

        loaded = load_text_embedding_sidecar(
            output,
            expected_model="fake/model",
            expected_revision="abc123",
            expected_content_sha256=metadata["content_sha256"],
        )
        np.testing.assert_allclose(np.linalg.norm(loaded.vectors, axis=1), 1.0)
        np.testing.assert_allclose(loaded.vectors, [[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(list(loaded.as_mapping()), ["article_N1", "article_N2"])
        self.assertEqual(list(loaded.as_mapping("source_id")), ["N1", "N2"])
        self.assertFalse(loaded.vectors.flags.writeable)

    def test_build_rejects_non_integer_batch_sizes(self):
        source=self.write_replay()
        for value in (True,1.5,0,-1):
            with self.subTest(value=value), self.assertRaises(TextEmbeddingError):
                build_text_embedding_sidecar(
                    source,self.root/f"invalid-{value}.npz",
                    encoder=FakeEncoder([[1,0],[0,1]]),batch_size=value,
                )

    def test_convenience_loader_does_not_import_ml_frameworks(self):
        output = self.root / "vectors.npz"
        build_text_embedding_sidecar(
            self.write_replay(), output,
            encoder=FakeEncoder([[1, 0], [0, 1]]), model_name="fake/model",
        )
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("sentence_transformers"):
                raise AssertionError(f"runtime loader imported {name}")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            mapping = load_text_embedding_sidecar(output).as_mapping("source_id")
        self.assertEqual(set(mapping), {"N1", "N2"})

    def test_rejects_invalid_encoder_output_without_overwriting_old_file(self):
        output = self.root / "vectors.npz"
        output.write_bytes(b"previous-good-artifact")
        before = output.read_bytes()
        with self.assertRaisesRegex(TextEmbeddingError, "zero or invalid"):
            build_text_embedding_sidecar(
                self.write_replay(), output,
                encoder=FakeEncoder([[0, 0], [1, 1]]), model_name="fake/model",
            )
        self.assertEqual(output.read_bytes(), before)

    def test_loader_detects_vector_tampering_and_model_mismatch(self):
        output = self.root / "vectors.npz"
        build_text_embedding_sidecar(
            self.write_replay(), output,
            encoder=FakeEncoder([[1, 0], [0, 1]]), model_name="fake/model",
        )
        with self.assertRaisesRegex(TextEmbeddingError, "model mismatch"):
            load_text_embedding_sidecar(output, expected_model="other/model")

        with np.load(output, allow_pickle=False) as payload:
            arrays = {name: np.array(payload[name]) for name in payload.files}
        arrays["vectors"][0] = [0.6, 0.8]  # Still unit norm; checksum must catch it.
        with output.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        with self.assertRaisesRegex(TextEmbeddingError, "checksum"):
            load_text_embedding_sidecar(output)

    def test_rejects_duplicate_ids_and_missing_text(self):
        duplicate = self.write_replay([
            {"id": "A", "source_id": "X", "title": "one"},
            {"id": "A", "source_id": "Y", "title": "two"},
        ])
        with self.assertRaisesRegex(TextEmbeddingError, "duplicate article ID"):
            read_article_corpus(duplicate)
        empty = self.write_replay([
            {"id": "A", "source_id": "X", "title": "", "abstract": ""},
        ])
        with self.assertRaisesRegex(TextEmbeddingError, "neither a title"):
            read_article_corpus(empty)


if __name__ == "__main__":
    unittest.main()
