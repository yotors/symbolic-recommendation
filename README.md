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

Retained real-time probe measured uncached first-page requests on one scorer.
At concurrency 1/2/4/8, p50 latency was
`778 / 1,015 / 2,213 / 3,665 ms`; throughput flattened near `1.24 requests/s`.
This exposes the single active-Lab lock as serving bottleneck.

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

## Maintained documentation

- `docs/architecture/ARCHITECTURE.md`: logic, formulas, workspaces, inference,
  ranking, and evaluation contract.
- `docs/architecture/RELATIONAL_REASONING.md`: bounded relational proofs and
  provenance.
