"""Parallel bounded NL2PLN extraction with process-isolated providers.

Sixteen fixed content-hash shards keep cache identities stable when changing
the worker count. Only a new merged export is immutable; dedicated shard caches
are resumable. All article and provider-call limits are totals across workers.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import math
import multiprocessing
from pathlib import Path

from ..integrations import llm_article_facts as facts


SHARD_COUNT = 16
DRIVER_SCHEMA = "mindplex-llm-batch-extraction-v1"
_STOP_EVENT = None


def _initialize_worker(stop_event):
    global _STOP_EVENT
    _STOP_EVENT = stop_event


def _shard_index(article):
    # Only text affects partitioning: no source ID, outcome or user input.
    digest = facts._hash({key: article.get(key) or "" for key in ("title", "abstract")})
    return int(digest[:8], 16) % SHARD_COUNT


def _plan_jobs(articles, *, cache_dir, model, max_new_articles, max_calls,
               batch_size, timeout_seconds, max_output_tokens):
    groups = [[] for _ in range(SHARD_COUNT)]
    for article in facts._article_inputs(articles):
        groups[_shard_index(article)].append(article)
    prompt_hash = facts.prompt_sha256()
    jobs, capacities = [], []
    for index, group in enumerate(groups):
        group.sort(key=lambda item: (facts.article_cache_key(item, model, prompt_hash=prompt_hash), item["id"]))
        path = facts._cache_location(cache_dir / f"shard-{index:02d}.json")
        cached = facts._load_cache(path)["entries"]
        pending = {facts.article_cache_key(item, model, prompt_hash=prompt_hash)
                   for item in group}.difference(cached)
        capacities.append(len(pending))
        jobs.append({"shard": index, "articles": group, "cache_path": str(path),
                     "model": model, "prompt_sha256": prompt_hash,
                     "batch_size": batch_size, "timeout_seconds": timeout_seconds,
                     "max_output_tokens": max_output_tokens,
                     "max_new_articles": 0, "max_calls": 0})
    # Allocate whole useful batches so a small 32-article run does not pay for
    # sixteen 2-item calls when eight 4-item calls suffice.
    remaining_articles, remaining_calls = max_new_articles, max_calls
    while remaining_articles and remaining_calls:
        changed = False
        for index, job in enumerate(jobs):
            available = capacities[index] - job["max_new_articles"]
            count = min(batch_size, available, remaining_articles)
            if count and remaining_calls:
                job["max_new_articles"] += count
                job["max_calls"] += 1
                remaining_articles -= count
                remaining_calls -= 1
                changed = True
        if not changed:
            break
    # Extra explicitly authorized calls accommodate the SDK's structured-JSON
    # retry inside a batch. No worker can borrow beyond its assigned quota.
    active = [job for job in jobs if job["max_new_articles"]]
    if active:
        quotient, remainder = divmod(remaining_calls, len(active))
        for index, job in enumerate(active):
            job["max_calls"] += quotient + int(index < remainder)
    return jobs


def _provider_worker(job, stop_event):
    def progress_update(progress):
        print(json.dumps({"shard": job["shard"], "progress": progress}), flush=True)
        if stop_event.is_set():
            raise facts.ArticleExtractionError("parallel extraction stopped after another shard failed", details={
                "provider_calls": progress["provider_calls"],
                "provider_usage": {"input_tokens": progress["input_tokens"], "output_tokens": progress["output_tokens"]},
            })

    return facts.extract_article_records(
        job["articles"], cache_path=job["cache_path"], model=job["model"],
        batch_size=job["batch_size"], max_new_articles=job["max_new_articles"],
        max_calls=job["max_calls"], timeout_seconds=job["timeout_seconds"],
        max_output_tokens=job["max_output_tokens"], progress_callback=progress_update,
        should_stop=stop_event.is_set,
    )


def _guarded_worker(job, worker_function):
    stop_event = _STOP_EVENT
    if stop_event.is_set():
        return {"shard": job["shard"], "cancelled": True}
    try:
        return {"shard": job["shard"], "result": worker_function(job, stop_event)}
    except facts.ArticleExtractionError as exc:
        return {"shard": job["shard"], "error": "article extraction shard failed",
                "details": exc.details}
    except Exception as exc:
        return {"shard": job["shard"], "error": "article extraction shard failed",
                "details": facts._safe_exception_details(exc)}


def extract_dataset(articles, *, cache_dir, output, model=None, workers=4,
                    max_new_articles=128, max_calls=32, batch_size=4,
                    timeout_seconds=60, max_output_tokens=8192,
                    worker_function=None, progress_callback=None):
    """Run bounded process workers and atomically publish a new export.

``worker_function`` is an explicit injectable, picklable test worker. Ordinary
usage always executes the real NL2PLN provider; there is no automatic fallback.
    """
    facts._positive_integer(workers, "workers", maximum=8)
    facts._positive_integer(batch_size, "batch_size", maximum=8)
    if batch_size < 4:
        raise facts.ArticleExtractionError("parallel extraction batch_size must be between 4 and 8")
    facts._positive_integer(max_new_articles, "max_new_articles", zero=True)
    facts._positive_integer(max_calls, "max_calls", zero=True, maximum=10_000)
    facts._positive_integer(max_output_tokens, "max_output_tokens", maximum=16_384)
    if isinstance(timeout_seconds, bool) or not math.isfinite(float(timeout_seconds)) or not 1 <= float(timeout_seconds) <= 180:
        raise facts.ArticleExtractionError("provider timeout must be between 1 and 180 seconds")
    directory = Path(cache_dir).resolve()
    if not directory.is_relative_to(facts.WORKSPACE_ROOT.resolve()):
        raise facts.ArticleExtractionError("parallel annotation cache must be inside the workspace")
    destination = facts._cache_location(output)
    if destination.exists() or destination in {directory / f"shard-{index:02d}.json" for index in range(SHARD_COUNT)}:
        raise facts.ArticleExtractionError("merged output must be a new artifact, separate from shard caches")
    selected_model = model or facts.configured_model()
    jobs = _plan_jobs(articles, cache_dir=directory, model=selected_model,
                      max_new_articles=max_new_articles, max_calls=max_calls,
                      batch_size=batch_size, timeout_seconds=timeout_seconds,
                      max_output_tokens=max_output_tokens)
    directory.mkdir(parents=True, exist_ok=True)
    records, results = {}, []
    progress = {"input_articles": sum(len(job["articles"]) for job in jobs),
                "initial_cached_articles": 0, "new_articles": 0, "batch_calls": 0,
                "provider_calls": 0, "input_tokens": 0, "output_tokens": 0}
    context = multiprocessing.get_context("spawn")
    failure = None
    stop_event = context.Event()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                               initializer=_initialize_worker, initargs=(stop_event,)) as executor:
        future_jobs = {
            executor.submit(_guarded_worker, job, worker_function or _provider_worker): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(future_jobs):
            job = future_jobs[future]
            try:
                completed = future.result()
            except Exception as exc:
                completed = {"error": "extraction worker process failed", "details": facts._safe_exception_details(exc)}
            if completed.get("error") or completed.get("cancelled"):
                failure = completed
                stop_event.set()
                for pending in future_jobs:
                    pending.cancel()
                break
            result = completed["result"]
            current = result["progress"]
            if (result.get("schema") != facts.SCHEMA_VERSION
                    or result.get("model") != selected_model
                    or result.get("prompt_sha256") != job["prompt_sha256"]
                    or current["new_articles"] > job["max_new_articles"]
                    or current["provider_calls"] > job["max_calls"]):
                failure = {"error": "worker exceeded its assigned extraction contract"}
                stop_event.set()
                for pending in future_jobs:
                    pending.cancel()
                break
            allowed_ids = {item["id"] for item in job["articles"]}
            if not set(result["extraction_records"]).issubset(allowed_ids):
                failure = {"error": "worker returned an article outside its assigned shard"}
                stop_event.set()
                break
            records.update(result["extraction_records"])
            for key in progress:
                if key != "input_articles":
                    progress[key] += current.get(key, 0)
            results.append({"shard": job["shard"], "progress": current})
            if progress_callback is not None:
                progress_callback({"completed_shards": len(results), **progress})
    if failure is not None:
        # Other in-flight requests finish within their SDK deadline and save
        # their completed batch before observing the stop flag. Collect those
        # terminal counters too, not just the first failed worker's usage.
        observed = {"provider_calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "shards_without_usage": 0}
        for future in future_jobs:
            if future.cancelled():
                continue
            try:
                completed = future.result()
            except Exception:
                observed["shards_without_usage"] += 1
                continue
            if completed.get("cancelled"):
                continue
            if "result" in completed:
                counts = completed["result"]["progress"]
            else:
                details = completed.get("details", {})
                counts = {"provider_calls": details.get("provider_calls", 0),
                          **details.get("provider_usage", {})}
                if "provider_calls" not in details:
                    observed["shards_without_usage"] += 1
            for name in ("provider_calls", "input_tokens", "output_tokens"):
                observed[name] += counts.get(name, 0)
        raise facts.ArticleExtractionError(
            "parallel extraction stopped; successful shard caches are preserved and no merged output was published",
            details={**failure.get("details", {}), "all_shards_observed_totals": observed,
                     "assigned_provider_call_upper_bound": sum(job["max_calls"] for job in jobs)},
        )
    progress.update(available_articles=len(records), unprocessed_articles=progress["input_articles"] - len(records))
    progress["driver"] = {
        "schema": DRIVER_SCHEMA, "shards": SHARD_COUNT, "workers": workers,
        "max_new_articles": max_new_articles, "max_calls": max_calls,
        "batch_size": batch_size, "max_output_tokens": max_output_tokens,
        "timeout_seconds": timeout_seconds,
        "shard_quotas": [{key: job[key] for key in ("shard", "max_new_articles", "max_calls")} for job in jobs],
    }
    result = {"schema": facts.SCHEMA_VERSION, "model": selected_model,
              "prompt_sha256": facts.prompt_sha256(),
              "extraction_records": {key: records[key] for key in sorted(records)}, "progress": progress}
    facts._write_cache(destination, result, exclusive=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-articles", type=int, default=128)
    parser.add_argument("--max-calls", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    args = parser.parse_args(argv)
    try:
        if args.data.resolve() == args.output.resolve():
            raise facts.ArticleExtractionError("merged output cannot overwrite the source dataset")
        opener = gzip.open if args.data.suffix == ".gz" else open
        with opener(args.data, "rt", encoding="utf-8") as handle:
            dataset = json.load(handle)
        result = extract_dataset(
            dataset["articles"], cache_dir=args.cache_dir, output=args.output,
            model=args.model, workers=args.workers, max_new_articles=args.max_new_articles,
            max_calls=args.max_calls, batch_size=args.batch_size,
            timeout_seconds=args.timeout_seconds, max_output_tokens=args.max_output_tokens,
            progress_callback=lambda value: print(json.dumps({"total_progress": value}), flush=True),
        )
        print(json.dumps({"output": str(args.output.resolve()), "progress": result["progress"]}), flush=True)
        return 0
    except facts.ArticleExtractionError as exc:
        print(json.dumps({"error": str(exc), "details": exc.details}), flush=True)
        return 2
    except Exception as exc:
        print(json.dumps({"error": "parallel extraction failed safely", "details": facts._safe_exception_details(exc)}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
