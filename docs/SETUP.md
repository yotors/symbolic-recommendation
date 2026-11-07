# Setup and verification

This guide separates checks that work with the checked-in Python code from
the live recommender, which needs a separately configured reasoning stack.
The commands below use a POSIX shell.

## Workspace layout

Clone the repository as `recommendation`, because that is its Python package
name:

```bash
git clone https://github.com/yotors/symbolic-recommendation.git recommendation
```

Run module commands from the parent workspace:

```text
workspace/
├── recommendation/       # this repository
│   ├── __main__.py
│   ├── pyproject.toml
│   └── tests/
└── PeTTaChainer/          # separate, configured runtime for the live app
    ├── .venv/
    └── pettachainer/
```

`pyproject.toml` currently configures pytest only. It does not declare project
dependencies or an installable package. Do not treat `pip install -e .` as a
complete setup procedure.

## Python-only contract tests

Python 3.10 or newer can run this selection without pytest, NumPy, datasets,
API keys, SWI-Prolog, or PeTTaChainer:

```bash
python3 -m unittest \
  recommendation.tests.unit.ranking.test_symbolic \
  recommendation.tests.unit.mining.test_ctv_calibration \
  recommendation.tests.unit.mining.test_petta_workspace \
  recommendation.tests.unit.integrations.test_engine \
  -v
```

These tests cover deterministic numeric encoding, impression-weighted
confidence estimation, workspace update and rollback behavior, and the
semantic HTTP client's validation contract. Workspace and HTTP tests use
test doubles; they do not run the real reasoner or call a model service.

This selection was verified as **43 passing tests on Python 3.12.14**.
It also passes with `python3 -S`, which disables site-package loading.
This is a focused component check, not a full application or benchmark run.

## Live runtime

The retained application workflow uses Python 3.10 from
`PeTTaChainer/.venv/bin/python`, the compatible installed `petta` runtime,
the sibling `pettachainer` package, NumPy, and a working SWI-Prolog runtime.
The retained benchmark summary records Python 3.10.20.

**Current setup limitation:** this repository does not record a verified
PeTTa/PeTTaChainer revision pair, a complete dependency lock, or instructions
for provisioning that external stack from scratch. Use a known compatible
PeTTaChainer environment. A fresh clone alone is sufficient for the
Python-only tests above, but not for the live app. No dependency versions
should be inferred from the absence of a manifest.

Once that stack exists, run these checks from the parent workspace:

```bash
test -x PeTTaChainer/.venv/bin/python
swipl --version
export PYTHONPATH=PeTTaChainer
PeTTaChainer/.venv/bin/python -c \
  'import numpy; from petta import PeTTa; from pettachainer.pettachainer import PeTTaChainer; print("Runtime imports available")'
PeTTaChainer/.venv/bin/python -m recommendation --fixture
```

The import check confirms that the expected modules can be loaded. Successful
imports alone do not establish that the runtime versions are compatible.
The fixture launch exercises the actual mining and proof path.

Do not add `PeTTa/python` to `PYTHONPATH`. `app/server.py` deliberately
removes that known legacy path because it can shadow the compatible runtime
installed in the PeTTaChainer virtual environment. The module path above
exposes the sibling chainer package without selecting that old PeTTa source
tree.

The bundled fixture contains 8 articles and 4 users. It needs no external
dataset or LLM credentials. Open <http://127.0.0.1:7070> after startup.
Keep the default loopback binding for a local demo.

### Optional semantic preview

The HTTP semantic-preview path is separate from local proof scoring. Its
configuration is described by [`.env.example`](../.env.example) and validated
in [`integrations/engine.py`](../integrations/engine.py).

For that optional path, configure the PeTTaChainer service URL, API key, and
knowledge-base ID in environment variables or a local `recommendation/.env`.
The file is ignored by Git. Do not commit credentials. Missing semantic-preview
credentials do not replace or disable the live miner's local runtime
requirement.

## Full test suites

After configuring the external runtime and installing pytest in its
environment, run from the same parent workspace:

```bash
export PYTHONPATH=PeTTaChainer
PeTTaChainer/.venv/bin/python -m pytest recommendation/tests/unit -q
PeTTaChainer/.venv/bin/python -m pytest recommendation/tests/integration -q
```

Some unit tests import the server and therefore need the real runtime at
collection time. Selecting the `unit` marker is not a way to remove those
import dependencies.

Browser tests additionally require Firefox, geckodriver, and Selenium:

```bash
PeTTaChainer/.venv/bin/python -m pytest recommendation/tests/browser -q
```

The focused validation recorded above did not run the full suites, start the
live server, or reproduce the retained MIND benchmark.

## Data and reproducibility

See [dataset setup](../dataset/README.md) before running the MIND examples.
Downloading the raw archive does not create the text embeddings, frozen LLM
annotations, canonical workspace, or serving model used in the retained
results.

[benchmark-summary.json](../results/benchmark-summary.json) records measurements,
scope, and SHA-256 identities for local artifacts. Those hashes help identify
artifacts if they are available; they do not make the artifacts downloadable.
The full evidence files and prepared datasets are not checked in.

## Troubleshooting

- **`No module named recommendation`:** check that the directory is named
  `recommendation` and that the command is run from its parent.
- **Missing `PeTTaChainer/.venv/bin/python`:** the separate runtime environment
  has not been provisioned at the expected location. Use the Python-only
  checks until a compatible stack is available.
- **Missing `petta`, `pettachainer`, or `numpy`:** verify the selected
  interpreter and the import check above. An arbitrary similarly named package
  is not evidence of compatibility.
- **Syntax errors while loading the chainer library:** check for a legacy
  `PeTTa/python` path and use the compatible runtime installed in the chainer
  environment.
- **Missing dataset or serving-model files:** these are local artifacts, not
  files supplied by cloning this repository. Use `--fixture` for the bundled
  demo or provide the exact artifacts your command names.
- **`--workers > 1 requires an immutable --serving-model`:** export a frozen
  model before starting the scorer pool, as shown in the main README.
