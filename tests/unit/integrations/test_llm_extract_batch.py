from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from recommendation.integrations import llm_article_facts as facts
from recommendation.cli.llm_extract_batch import SHARD_COUNT, _plan_jobs, _shard_index, extract_dataset
from recommendation.features.text_embeddings import article_text


def fake_process_worker(job, _stop_event):
    cache_path = Path(job["cache_path"])
    cache = facts._load_cache(cache_path)
    new, records = 0, {}
    for article in job["articles"]:
        key = facts.article_cache_key(article, job["model"], prompt_hash=job["prompt_sha256"])
        if key not in cache["entries"] and new < job["max_new_articles"]:
            record = {"concepts": ["test observation"], "format": "report", "event_types": [], "intents": [], "audiences": [],
                      "provenance": {"schema": facts.SCHEMA_VERSION, "cache_key": key, "model": job["model"],
                                     "prompt_sha256": job["prompt_sha256"], "status": "parsed",
                                     "article_content_sha256": hashlib.sha256(article_text(article["title"], article["abstract"]).encode()).hexdigest()}}
            cache["entries"][key] = {"record": record, "record_sha256": facts._hash(record)}
            new += 1
        if key in cache["entries"]:
            records[article["id"]] = cache["entries"][key]["record"]
    if new:
        facts._write_cache(cache_path, cache)
    calls = math.ceil(new / job["batch_size"])
    return {"schema": facts.SCHEMA_VERSION, "model": job["model"], "prompt_sha256": job["prompt_sha256"],
            "extraction_records": records, "progress": {
                "input_articles": len(job["articles"]), "initial_cached_articles": len(records) - new,
                "new_articles": new, "batch_calls": calls, "provider_calls": calls,
                "input_tokens": new * 10, "output_tokens": new * 5,
            }}


def failing_process_worker(job, event):
    result = fake_process_worker(job, event)
    if job["shard"] == 0:
        raise facts.ArticleExtractionError("safe failure", details={"exception_types": ["AdapterParseError"]})
    return result


class BatchExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=facts.WORKSPACE_ROOT, prefix=".test-llm-batch-")
        self.root = Path(self.temporary.name)
        self.cache_dir = self.root / "cache"
        self.articles = [{"id": f"article_{index}", "title": f"Distinct scientific observation {index}", "abstract": ""}
                         for index in range(80)]

    def tearDown(self):
        self.temporary.cleanup()

    def test_plans_are_content_only_and_total_caps_are_not_per_worker(self):
        options = dict(cache_dir=self.cache_dir, model="test/model", max_new_articles=23,
                       max_calls=9, batch_size=4, timeout_seconds=20, max_output_tokens=8192)
        jobs = _plan_jobs(self.articles, **options)
        repeated = _plan_jobs(list(reversed(self.articles)), **options)
        self.assertEqual(jobs, repeated)
        self.assertEqual(len(jobs), SHARD_COUNT)
        self.assertEqual(sum(job["max_new_articles"] for job in jobs), 23)
        self.assertEqual(sum(job["max_calls"] for job in jobs), 9)
        example = self.articles[0]
        self.assertEqual(_shard_index(example), _shard_index({**example, "id": "other", "label": 1, "user": "u"}))

    def test_process_workers_merge_and_resume_with_zero_paid_calls(self):
        first = extract_dataset(self.articles, cache_dir=self.cache_dir, output=self.root / "first.json",
                                model="test/model", workers=2, max_new_articles=12, max_calls=4,
                                worker_function=fake_process_worker)
        self.assertEqual(first["progress"]["new_articles"], 12)
        self.assertLessEqual(first["progress"]["provider_calls"], 4)
        self.assertEqual(len(first["extraction_records"]), 12)
        second = extract_dataset(self.articles, cache_dir=self.cache_dir, output=self.root / "resumed.json",
                                 model="test/model", workers=1, max_new_articles=0, max_calls=0,
                                 worker_function=fake_process_worker)
        self.assertEqual(first["extraction_records"], second["extraction_records"])
        self.assertEqual(second["progress"]["provider_calls"], 0)
        self.assertEqual(second["progress"]["initial_cached_articles"], 12)
        self.assertEqual(set(first), {"schema", "model", "prompt_sha256", "extraction_records", "progress"})

    def test_failed_shard_retains_cache_and_does_not_publish_output(self):
        output = self.root / "failed.json"
        with self.assertRaisesRegex(facts.ArticleExtractionError, "caches are preserved"):
            extract_dataset(self.articles, cache_dir=self.cache_dir, output=output,
                            model="test/model", workers=2, max_new_articles=16, max_calls=8,
                            worker_function=failing_process_worker)
        self.assertFalse(output.exists())
        self.assertTrue((self.cache_dir / "shard-00.json").exists())

    def test_existing_output_and_invalid_bounds_fail_before_workers(self):
        output = self.root / "historical.json"
        output.write_text("historical", encoding="utf-8")
        with self.assertRaises(facts.ArticleExtractionError):
            extract_dataset(self.articles, cache_dir=self.cache_dir, output=output,
                            model="test/model", worker_function=fake_process_worker)
        self.assertEqual(output.read_text(), "historical")
        self.assertFalse(self.cache_dir.exists())
        with self.assertRaises(facts.ArticleExtractionError):
            extract_dataset(self.articles, cache_dir=self.cache_dir, output=self.root / "new.json",
                            model="test/model", workers=9, worker_function=fake_process_worker)

    def test_exclusive_publication_never_replaces_a_racing_existing_file(self):
        output = self.root / "raced.json"
        output.write_text("other writer", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            facts._write_cache(output, {"result": "ours"}, exclusive=True)
        self.assertEqual(output.read_text(), "other writer")
        self.assertFalse(list(self.root.glob(".raced.json.*")))


if __name__ == "__main__":
    unittest.main()
