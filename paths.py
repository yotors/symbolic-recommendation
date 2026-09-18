"""Authoritative filesystem locations for the recommendation package.

Runtime code must resolve repository resources through this module rather than
through the location of an individual implementation module.  Dataset, result,
miner, and environment locations deliberately remain stable so immutable
artifacts and operator workflows keep their meaning; relocatable UI and cache
resources live in their dedicated hierarchy.
"""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent

DATASET_DIR = PACKAGE_ROOT / "dataset"
RESULTS_DIR = PACKAGE_ROOT / "results"
MINER_DIR = PACKAGE_ROOT / "miner"
WEB_TEMPLATE = PACKAGE_ROOT / "web" / "templates" / "index.html"
ENV_FILE = PACKAGE_ROOT / ".env"

RUNTIME_DIR = PACKAGE_ROOT / "runtime"
RUNTIME_CACHE_DIR = RUNTIME_DIR / "cache" / "llm"
GATE_CLAIM_DIR = RESULTS_DIR / "training-confirmation-attempts"


__all__ = (
    "PACKAGE_ROOT",
    "WORKSPACE_ROOT",
    "DATASET_DIR",
    "RESULTS_DIR",
    "MINER_DIR",
    "WEB_TEMPLATE",
    "ENV_FILE",
    "RUNTIME_DIR",
    "RUNTIME_CACHE_DIR",
    "GATE_CLAIM_DIR",
)
