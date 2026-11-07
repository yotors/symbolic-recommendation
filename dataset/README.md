# Local datasets

Dataset and generated replay files are intentionally excluded from Git.
The bundled `--fixture` demo needs none of these files, but it still requires
the real reasoning runtime described in [setup](../docs/SETUP.md).

## Download the raw MIND-small projection

Run from the parent workspace containing `recommendation/`:

```bash
python3 -m recommendation.cli.fetch_mind --help
```

Read the [Microsoft Research data license](https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf)
first. If you accept its terms, download the archive with:

```bash
python3 -m recommendation.cli.fetch_mind --accept-license
```

The downloader fetches the RecZoo MIND-small projection and verifies its
SHA-256 before placing it at `recommendation/dataset/MIND_small_x1.zip`.
It does not fetch a trained model or generate derived features.

## Files used by the retained demonstration

- `MIND_small_x1.zip`: raw benchmark projection, obtained by the downloader.
- `MIND_small_x1.text-embeddings.npz`: generated, provenanced article vectors.
- `llm-workspace-canonical-v3.json.gz`: prepared causal replay with frozen,
  canonicalized article observations.
- `.mind-replay-46d0a7c99fe3306f733d.json.gz`: optional loader cache from the
  retained local workflow. Cache names depend on the loading configuration.

The main README's raw-replay command expects the archive and embedding sidecar.
Its prepared-workspace command expects the canonical replay file. Neither
command recreates the missing inputs.

## Derived-data tools

The code contains separate builders and projectors. These help inspect the
available interfaces; they are not a complete recipe for reconstructing the
retained benchmark:

```bash
python3 -m recommendation.features.text_embeddings --help
python3 -m recommendation.cli.llm_extract_batch --help
python3 -m recommendation.cli.canonicalize_llm_annotations --help
python3 -m recommendation.pipelines.llm_data --help
```

These modules require NumPy. Building embeddings additionally uses
`sentence-transformers` and model weights. Running LLM extraction requires a
configured provider and can send article content to that provider and incur
API charges. The `--help` commands do not perform extraction.

Canonicalization and projection require matching replay, causal-history, and
annotation inputs. The repository does not include all retained inputs,
provider/model settings, or pinned dependency versions needed to reproduce the
published figures exactly.

The retained [benchmark summary](../results/benchmark-summary.json) identifies
local evidence and model artifacts by hash and states the evaluation limits.
Full per-case evidence, generated data, and serving models remain outside Git.
