# Symbolic Recommendation Lab

Live recommendation prototype using a real MeTTa fpMiner for behavioral-rule
discovery and PeTTaChainer for grounded point and pair proofs. Python owns data
loading, bounded workspace construction, orchestration, metrics, and HTTP; it
does not provide a fallback recommender.

## Active flow

1. Load MIND or a prepared dataset-agnostic causal replay.
2. Build pre-outcome candidate facts from article content and user history.
3. Mine supported target-aware structures from closed training cases.
4. Recount retained structures on the full training population and encode CTVs.
5. Ground candidate and candidate-pair facts in isolated PeTTaChainer channels.
6. Rank from proof-derived point evidence and a balanced pair tournament.
7. On feedback, update private user state, invalidate affected rankings, and
   rerank immediately. Mine accumulated complete cases asynchronously at the
   configured interval.

The current workspace includes bounded LLM content descriptors and frozen
text-vector observations. These create symbolic facts only; neither model
directly assigns recommendation scores.

## Run

Requirements: Python 3.10 environment from `PeTTaChainer/.venv`, working PeTTa
and SWI-Prolog runtime, and files under `dataset/`.

From the workspace containing `recommendation/`, `PeTTa/`, and
`PeTTaChainer/`:

```bash
export PYTHONPATH=PeTTa/python:PeTTaChainer
```

Fast deterministic demo:

```bash
PeTTaChainer/.venv/bin/python -m recommendation --fixture
```

Current prepared MIND workspace:

```bash
PeTTaChainer/.venv/bin/python -m recommendation \
  --replay-data recommendation/dataset/llm-workspace-canonical-v3.json.gz
```

Raw MIND-small replay:

```bash
PeTTaChainer/.venv/bin/python -m recommendation \
  --mind recommendation/dataset/MIND_small_x1.zip \
  --text-embeddings recommendation/dataset/MIND_small_x1.text-embeddings.npz \
  --max-train-cases 20000 --max-eval-impressions 500 --seed 7
```

Open <http://127.0.0.1:7070>. Browser supplies infinite proof-ranked feed,
click/skip feedback, state inspection, mining controls, and replay benchmark.

## Production startup

Mine and compile one immutable model artifact outside the request path:

```bash
PeTTaChainer/.venv/bin/python -m recommendation \
  --replay-data recommendation/dataset/llm-workspace-canonical-v3.json.gz \
  --config-file selected-model.json \
  --export-serving-model symbolic-serving-model.json \
  --export-only
```

Start the production gateway with two isolated scorers. Only port 7070 is
public; scorer ports bind to loopback:

```bash
PeTTaChainer/.venv/bin/python -m recommendation \
  --replay-data recommendation/dataset/llm-workspace-canonical-v3.json.gz \
  --serving-model symbolic-serving-model.json \
  --host 0.0.0.0 --port 7070 \
  --workers 2 --worker-start-port 7171
```

The gateway uses deterministic user/session affinity, bounded request admission,
body and upstream limits, readiness checks, and automatic scorer replacement.
After a scorer restart, its old browser sessions automatically receive a new
proof-ranked stream with `reset=true`. Serving workers are immutable: mining,
configuration, dataset replacement, tuning, and training-confirmation routes
are rejected. Publish a new digest-validated frozen model through the offline
control plane instead.

`/health/live` reports process liveness. `/health/ready` succeeds only when all
scorers have loaded and prewarmed every point and pair proof channel.
`/api/state` verifies model/config/dataset agreement across the pool and returns
aggregate counters. Put TLS, authentication, rate limiting, and durable event
storage in the platform ingress/data plane; the bundled gateway intentionally
owns only local inference orchestration.

On the current 20k-event MIND workspace, the optimized frozen scorer produced
a fresh 40-candidate, 780-comparison page in `205.3ms` median across 12 users
(`159.6–295.5ms`). Its next queued infinite-scroll page took `6.1ms`.
Incremental proof-backed skip feedback reranked 39 remaining candidates in
`133.3ms`, recomputing only affected point cases while replaying the existing
pair proofs. These are single-machine engineering measurements, not capacity
percentiles.

## Current measured result

Retained MIND-small development replay used 19,996 training exposures and 500
complete validation impressions containing 18,139 candidates:

- served AUC: `0.6890846301`
- proof-derived AUC: `0.6890548267`
- served-AUC impression-bootstrap 95% interval: `[0.6659, 0.7148]`
- pointwise symbolic AUC: `0.6129`
- pair-pipeline delta over pointwise: `+0.0762`, 95% interval
  `[0.0541, 0.0990]`
- MRR: `0.3416`
- nDCG@5: `0.3799`
- nDCG@10: `0.4387`
- proof coverage: `100%`

This is development evidence on logged MIND slates, not hidden-test, online
causal, transfer, or SOTA evidence. Exact direct reconstruction matched every
PeTTaChainer slate order and score signature, establishing faithful execution
of implemented proof semantics.

The latest bounded production-pool probe measured 75 uncached fresh-user
first-page requests across two isolated scorers. At concurrency 1/2/4/8, p50
latency was `220 / 216 / 337 / 671 ms`; p95 was
`345 / 425 / 736 / 1,288 ms`. Peak throughput was `6.09 requests/s`, with zero
failures. Against the preceding retained probe, p50 improved by `36% / 51% /
40% / 29%` and peak throughput improved by `50%`. The earlier full process-tree
run measured `3.00 GiB` peak RSS, `136%` mean CPU, and `373%` peak CPU. Worker
termination, automatic restart, readiness recovery, and browser-session reset
remain covered by the gateway tests.

Machine-readable evidence is retained in `results/benchmark-summary.json`.
The summary records SHA-256 identities for the full local artifacts, which stay
outside Git because they contain tens of thousands of per-case observations.

## Source layout

```text
adapters/       dataset adapters and causal replay construction
app/            server, live state, ranking, and orchestration
cli/            data extraction and runtime probes
core/           symbolic encodings, interests, and CTV estimation
evaluation/     replay comparison, parity, and promotion gates
features/       content, semantic, recency, and relational observations
integrations/   PeTTaChainer, NL2PLN, and LLM boundaries
miner/          executable MeTTa fpMiner programs
mining/         incremental and target-aware mining workspaces
pipelines/      reproducible dataset projections
tests/          unit, integration, and browser contracts
web/            browser client
```

`python -m recommendation` is the only supported server entry point.

## Tests

```bash
# Fast deterministic contracts
PeTTaChainer/.venv/bin/python -m pytest recommendation/tests/unit -q

# Real component boundaries
PeTTaChainer/.venv/bin/python -m pytest recommendation/tests/integration -q

# Requires Firefox, geckodriver, and working reasoner runtime
PeTTaChainer/.venv/bin/python -m pytest recommendation/tests/browser -q
```

Tests remain separated by behavior. Generated caches and duplicate historical
experiment fixtures are not retained.
