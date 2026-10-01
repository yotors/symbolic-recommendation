"""Live MeTTa-miner -> PeTTaChainer recommendation lab."""
from __future__ import annotations
import argparse, gc, gzip, hashlib, hmac, html as html_lib, ipaddress, json, math, multiprocessing as mp, os, re, sys, threading, time, uuid
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..integrations.engine import (
    PeTTaChainerClient, PeTTaChainerConfigurationError,
    PeTTaChainerInputError, PeTTaChainerProtocolError,
    PeTTaChainerTimeoutError, PeTTaChainerUpstreamError,
)
from ..adapters.mind import load_mind
from ..core.ctv_calibration import DEFAULT_EVIDENCE_K
from ..features.text_embeddings import load_text_embedding_sidecar, _canonical_corpus
from ..features.lexical_workspace import LEXICAL_FEATURES
from ..features.semantic_workspace import SEMANTIC_WORKSPACE_FEATURES
from ..features.recency_workspace import RECENCY_WORKSPACE_FEATURES, RECENCY_WORKSPACE_SCHEMA
from ..features.llm_workspace import LLM_NUMERIC_FEATURES, LLM_WORKSPACE_FEATURES, LLM_WORKSPACE_SCHEMA
from ..features.relational_workspace import (
    RELATIONAL_WORKSPACE_SCHEMA,
    REL_CONCEPT_CONTINUITY_PROOF_IDS,
    REL_CONCEPT_CONTINUITY_SCOPE,
    REL_ENTITY_CONTINUITY_PROOF_IDS,
    REL_ENTITY_CONTINUITY_SCOPE,
    validate_relational_projection,
)
from ..mining.lab_lifecycle import make_model_lifecycle_mixin
from ..mining.petta_workspace import PeTTaWorkspaceCache
from ..mining.incremental_fpminer import IncrementalFpMinerCache
from ..mining.retention import (
    DEFAULT_RETENTION_MAX_CASES,
    DEFAULT_RETENTION_MAX_UNITS,
    HARD_RETENTION_MAX_CASES,
    HARD_RETENTION_MAX_UNITS,
)
from ..paths import DATASET_DIR, MINER_DIR, WEB_TEMPLATE, WORKSPACE_ROOT
from ..evaluation.lab_benchmark import (
    PAIR_PROOF_PREPARATION_TIMERS, _proof_channel_preparation_profile,
    make_benchmark_mixin,
)
from ..ranking.lab_pairwise import make_pairwise_ranking_mixin
from .async_mining import AsyncMiningCoordinator
from .lab_serving import SemanticPreviewBusyError, make_serving_mixin

_CHAINER_ROOT = WORKSPACE_ROOT / "PeTTaChainer"
# A common invocation prepends ``PeTTa/python``.  That source tree is an older
# PeTTa runtime and shadows the compatible wheel installed in the chainer
# virtualenv, producing a misleading syntax error while loading the chainer
# library.  Keep the chainer package first and remove only that known legacy
# path; the installed runtime remains the single source of truth.
_LEGACY_PETTA_ROOT = (WORKSPACE_ROOT / "PeTTa" / "python").resolve()
sys.path[:] = [entry for entry in sys.path
               if (not entry or Path(entry).resolve() != _LEGACY_PETTA_ROOT)]
sys.path.insert(0, str(_CHAINER_ROOT))
try:
    from petta import PeTTa
except ImportError as exc:
    raise RuntimeError("Run with PeTTaChainer/.venv; see recommendation/README.md") from exc

from .reasoner import IsolatedPeTTaChainer
from ..mining.rule_parser import (
    FEATURES, POSITIVE, STV_RE, balanced_forms, parse_petta_target_rules,
    parse_rules, proof_tv,
)

LIVE_NEGATIVE_FEATURE = "recent_negative_match"
LIVE_NEGATIVE_CONCLUSION = "Live_Feedback_Click"
LIVE_NEGATIVE_CHAIN_STEPS = 4
LIVE_NEGATIVE_LEVELS = ("exact", "subcategory", "topic", "none")
LIVE_NEGATIVE_HISTORY_LIMIT = 20
LIVE_NEGATIVE_GENERALIZATION_WINDOW = 5
RELATIONAL_LIVE_HISTORY_LIMIT = 50
RELATIONAL_QUERY_STEPS_PER_ROOT = 32
RELATIONAL_PROOF_FIELDS = {
    REL_ENTITY_CONTINUITY_SCOPE: REL_ENTITY_CONTINUITY_PROOF_IDS,
    REL_CONCEPT_CONTINUITY_SCOPE: REL_CONCEPT_CONTINUITY_PROOF_IDS,
}
# A normal cursor request acknowledges the preceding page, so only one entry
# remains in steady state.  Keep a bounded tail for malformed/retried clients;
# the newest page is always retained and the HTTP page-size ceiling bounds it.
FEED_DELIVERY_HISTORY_LIMIT = 16
FEED_DELIVERY_ROW_LIMIT = 1000
LIVE_NEGATIVE_RULES = (
    # These are explicit online-feedback policies, not mined population
    # statistics.  The certain candidate fact says how closely an article
    # matches the user's recent skips; PeTTa's CTV revision decides how that
    # evidence combines with the currently mined click proofs.
    # A skip is evidence against the click target. Zero strength guarantees
    # every standalone feedback proof lies below every non-zero empirical click
    # prior; confidence controls how far exact/taxonomy matches demote it.
    ("feedback_skip_exact", "exact", 0.0, 0.95),
    ("feedback_skip_subcategory", "subcategory", 0.0, 0.80),
    ("feedback_skip_topic", "topic", 0.0, 0.65),
)
LIVE_NEGATIVE_RULE_IDS = frozenset(rule_id for rule_id, *_rest in LIVE_NEGATIVE_RULES)
LIVE_NEGATIVE_RULE_SOURCES = tuple(
    f'(: {rule_id} (Implication '
    f'({LIVE_NEGATIVE_FEATURE.title()} $case {json.dumps(level)}) '
    f'({LIVE_NEGATIVE_CONCLUSION} $case)) '
    f'(CTV (STV {strength} {confidence}) (STV 0.5 0.0)))'
    for rule_id, level, strength, confidence in LIVE_NEGATIVE_RULES
)
MULTI_INTEREST_NUMERIC_FEATURES = (
    "mi_topic_candidate_affinity_score",
    "mi_topic_candidate_recency_score",
    "mi_subcategory_candidate_affinity_score",
    "mi_subcategory_candidate_recency_score",
    "mi_entity_candidate_affinity_score",
    "mi_entity_candidate_recency_score",
    "mi_semantic_top1_similarity",
    "mi_semantic_topk_mean_similarity",
    "mi_semantic_weighted_similarity",
    "mi_semantic_attention_score",
)
TEXT_SEMANTIC_NUMERIC_FEATURES = (
    "text_semantic_coverage",
    "text_semantic_top1_similarity",
    "text_semantic_top3_mean_similarity",
    "text_semantic_top5_mean_similarity",
    "text_semantic_attention_t8_similarity",
    "text_semantic_attention_t12_similarity",
    "text_semantic_centroid_similarity",
    "text_semantic_recent5_centroid_similarity",
    "text_semantic_recent5_max_similarity",
    "text_semantic_last20_recency_decayed_similarity",
)
CONTEXT_FEATURES = (
    *FEATURES, LIVE_NEGATIVE_FEATURE, "topic_affinity", "recent_topic_affinity",
    "subcategory_affinity_score", "entity_recent_top1_similarity",
    "entity_long_mean_similarity", "title_history_idf_jaccard",
    "recent_subcategory_transition_score", "recent_subcategory_affinity_score",
    "topic_recency_score", "subcategory_recency_score",
    *MULTI_INTEREST_NUMERIC_FEATURES,
    *TEXT_SEMANTIC_NUMERIC_FEATURES,
    *LEXICAL_FEATURES,
    *SEMANTIC_WORKSPACE_FEATURES,
    *RECENCY_WORKSPACE_FEATURES,
    *LLM_WORKSPACE_FEATURES,
)
LLM_QUANTILE_PAIR_PREDICATES = tuple(
    f"pair_{name}_quantile" for name in LLM_NUMERIC_FEATURES
)
PAIR_CATEGORICAL_SIDE_FAMILIES = {
    "topic": ("pair_left_topic", "pair_right_topic"),
    "subcategory": ("pair_left_subcategory", "pair_right_subcategory"),
    "llm_format": ("pair_left_llm_format", "pair_right_llm_format"),
}
PAIR_CATEGORICAL_SIDE_PREDICATES = frozenset(
    predicate
    for predicates in PAIR_CATEGORICAL_SIDE_FAMILIES.values()
    for predicate in predicates
)
PAIR_FEATURES = (
    *(f"pair_{name}" for name in LLM_NUMERIC_FEATURES),
    *LLM_QUANTILE_PAIR_PREDICATES,
    *sorted(PAIR_CATEGORICAL_SIDE_PREDICATES),
    "pair_history_scope",
    *(f"pair_{name}" for name in LEXICAL_FEATURES),
    *(f"pair_{name}" for name in SEMANTIC_WORKSPACE_FEATURES),
    *(f"pair_{name}" for name in RECENCY_WORKSPACE_FEATURES),
    "pair_affinity", "pair_recent_affinity", "pair_long_affinity",
    "pair_entity_overlap", "pair_rel_entity_continuity_scope",
    "pair_rel_concept_continuity_scope",
    "pair_history_topic_count",
    "pair_recent_topic_count", "pair_topic_rank",
    "pair_subcategory_affinity", "pair_title_overlap", "pair_ctr",
    "pair_freshness", "pair_format", "pair_same_topic",
    "pair_same_subcategory", "pair_stable_dominance",
    "pair_entity_recent_top1_similarity",
    "pair_recent_subcategory_transition",
    "pair_entity_recent_top1_similarity_quantile",
    "pair_recent_subcategory_transition_quantile",
    "pair_long_topic_share_quantile", "pair_recent_topic_share_quantile",
    "pair_subcategory_share_quantile",
    "pair_title_history_idf_jaccard", "pair_entity_long_mean_similarity",
    "pair_recent_subcategory_affinity", "pair_topic_recency",
    "pair_subcategory_recency",
    "pair_recent_subcategory_affinity_quantile", "pair_topic_recency_quantile",
    "pair_subcategory_recency_quantile",
    "pair_mi_topic_affinity", "pair_mi_topic_recency",
    "pair_mi_subcategory_affinity", "pair_mi_subcategory_recency",
    "pair_mi_entity_affinity", "pair_mi_entity_recency",
    "pair_mi_semantic_top1", "pair_mi_semantic_topk_mean",
    "pair_mi_semantic_weighted", "pair_mi_semantic_attention",
    "pair_mi_topic_affinity_quantile", "pair_mi_topic_recency_quantile",
    "pair_mi_subcategory_affinity_quantile", "pair_mi_subcategory_recency_quantile",
    "pair_mi_entity_affinity_quantile", "pair_mi_entity_recency_quantile",
    "pair_mi_semantic_top1_quantile", "pair_mi_semantic_topk_mean_quantile",
    "pair_mi_semantic_weighted_quantile", "pair_mi_semantic_attention_quantile",
    "pair_text_semantic_top1", "pair_text_semantic_top3_mean",
    "pair_text_semantic_top5_mean", "pair_text_semantic_attention_t8",
    "pair_text_semantic_attention_t12", "pair_text_semantic_centroid",
    "pair_text_semantic_recent5_centroid", "pair_text_semantic_recent5_max",
    "pair_text_semantic_last20_decay",
    "pair_text_semantic_top1_quantile", "pair_text_semantic_top3_mean_quantile",
    "pair_text_semantic_top5_mean_quantile",
    "pair_text_semantic_attention_t8_quantile",
    "pair_text_semantic_attention_t12_quantile",
    "pair_text_semantic_centroid_quantile",
    "pair_text_semantic_recent5_centroid_quantile",
    "pair_text_semantic_recent5_max_quantile",
    "pair_text_semantic_last20_decay_quantile",
)
NUMERIC_PAIR_EVIDENCE = {
    **{name:f"pair_{name}_quantile" for name in LLM_NUMERIC_FEATURES},
    "entity_recent_top1_similarity":"pair_entity_recent_top1_similarity_quantile",
    "recent_subcategory_transition_score":"pair_recent_subcategory_transition_quantile",
    "topic_affinity":"pair_long_topic_share_quantile",
    "recent_topic_affinity":"pair_recent_topic_share_quantile",
    "subcategory_affinity_score":"pair_subcategory_share_quantile",
    "recent_subcategory_affinity_score":
        "pair_recent_subcategory_affinity_quantile",
    "topic_recency_score":"pair_topic_recency_quantile",
    "subcategory_recency_score":"pair_subcategory_recency_quantile",
    "mi_topic_candidate_affinity_score":"pair_mi_topic_affinity_quantile",
    "mi_topic_candidate_recency_score":"pair_mi_topic_recency_quantile",
    "mi_subcategory_candidate_affinity_score":"pair_mi_subcategory_affinity_quantile",
    "mi_subcategory_candidate_recency_score":"pair_mi_subcategory_recency_quantile",
    "mi_entity_candidate_affinity_score":"pair_mi_entity_affinity_quantile",
    "mi_entity_candidate_recency_score":"pair_mi_entity_recency_quantile",
    "mi_semantic_top1_similarity":"pair_mi_semantic_top1_quantile",
    "mi_semantic_topk_mean_similarity":"pair_mi_semantic_topk_mean_quantile",
    "mi_semantic_weighted_similarity":"pair_mi_semantic_weighted_quantile",
    "mi_semantic_attention_score":"pair_mi_semantic_attention_quantile",
    "text_semantic_top1_similarity":"pair_text_semantic_top1_quantile",
    "text_semantic_top3_mean_similarity":"pair_text_semantic_top3_mean_quantile",
    "text_semantic_top5_mean_similarity":"pair_text_semantic_top5_mean_quantile",
    "text_semantic_attention_t8_similarity":
        "pair_text_semantic_attention_t8_quantile",
    "text_semantic_attention_t12_similarity":
        "pair_text_semantic_attention_t12_quantile",
    "text_semantic_centroid_similarity":"pair_text_semantic_centroid_quantile",
    "text_semantic_recent5_centroid_similarity":
        "pair_text_semantic_recent5_centroid_quantile",
    "text_semantic_recent5_max_similarity":
        "pair_text_semantic_recent5_max_quantile",
    "text_semantic_last20_recency_decayed_similarity":
        "pair_text_semantic_last20_decay_quantile",
}
PAIR_REDUNDANT_INTEREST = frozenset({
    "pair_affinity", "pair_recent_affinity", "pair_long_affinity",
    "pair_history_topic_count", "pair_recent_topic_count", "pair_topic_rank",
    "pair_topic_recency",
})
PAIR_EVIDENCE_ALIASES = {
    # Descriptors are correlated views of the same article text/history, not
    # independent votes. Share the text dependency with the frozen encoder.
    **{f"pair_{name}":"pair_text_semantic_top3_mean" for name in LLM_NUMERIC_FEATURES},
    **{predicate:"pair_text_semantic_top3_mean"
       for predicate in LLM_QUANTILE_PAIR_PREDICATES},
    "pair_left_topic":"pair_long_affinity",
    "pair_right_topic":"pair_long_affinity",
    "pair_left_subcategory":"pair_subcategory_affinity",
    "pair_right_subcategory":"pair_subcategory_affinity",
    "pair_left_llm_format":"pair_text_semantic_top3_mean",
    "pair_right_llm_format":"pair_text_semantic_top3_mean",
    # All word-overlap summaries describe one lexical evidence source.
    **{f"pair_{name}":"pair_title_overlap" for name in LEXICAL_FEATURES},
    # Centering and concentration are views of the same frozen text encoder,
    # not additional independent witnesses for proof revision.
    **{f"pair_{name}":"pair_text_semantic_top3_mean" for name in (*SEMANTIC_WORKSPACE_FEATURES,*RECENCY_WORKSPACE_FEATURES)},
    "pair_entity_recent_top1_similarity_quantile":
        "pair_entity_recent_top1_similarity",
    "pair_recent_subcategory_transition_quantile":
        "pair_recent_subcategory_transition",
    "pair_long_topic_share_quantile":"pair_long_affinity",
    "pair_recent_topic_share_quantile":"pair_recent_affinity",
    "pair_subcategory_share_quantile":"pair_subcategory_affinity",
    "pair_recent_subcategory_affinity":"pair_subcategory_affinity",
    "pair_subcategory_recency":"pair_subcategory_affinity",
    "pair_topic_recency":"pair_long_affinity",
    "pair_title_history_idf_jaccard":"pair_title_overlap",
    "pair_recent_subcategory_affinity_quantile":
        "pair_recent_subcategory_affinity",
    "pair_topic_recency_quantile":"pair_topic_recency",
    "pair_subcategory_recency_quantile":"pair_subcategory_recency",
    # Recent-max and long-mean are two summaries of the same article-entity
    # embedding evidence.  Keep them in one dependency lineage so the proof
    # merger cannot count them as independent witnesses.
    "pair_entity_long_mean_similarity":
        "pair_entity_recent_top1_similarity",
    # Exact entity continuity refines the existing entity-overlap source with
    # a causal occurrence and recency.  It is not an independent witness.
    "pair_rel_entity_continuity_scope":"pair_entity_overlap",
    # Canonical concepts come from the same versioned content extraction as
    # the other LLM descriptors and therefore share its evidence owner.
    "pair_rel_concept_continuity_scope":"pair_text_semantic_top3_mean",
    "pair_mi_topic_affinity_quantile":"pair_mi_topic_affinity",
    "pair_mi_topic_recency_quantile":"pair_mi_topic_recency",
    "pair_mi_subcategory_affinity_quantile":"pair_mi_subcategory_affinity",
    "pair_mi_subcategory_recency_quantile":"pair_mi_subcategory_recency",
    "pair_mi_entity_affinity_quantile":"pair_mi_entity_affinity",
    "pair_mi_entity_recency_quantile":"pair_mi_entity_recency",
    "pair_mi_semantic_top1_quantile":"pair_mi_semantic_top1",
    "pair_mi_semantic_topk_mean_quantile":"pair_mi_semantic_top1",
    "pair_mi_semantic_weighted_quantile":"pair_mi_semantic_top1",
    "pair_mi_semantic_attention_quantile":"pair_mi_semantic_top1",
    "pair_mi_semantic_topk_mean":"pair_mi_semantic_top1",
    "pair_mi_semantic_weighted":"pair_mi_semantic_top1",
    "pair_mi_semantic_attention":"pair_mi_semantic_top1",
    "pair_mi_topic_recency":"pair_mi_topic_affinity",
    "pair_mi_subcategory_recency":"pair_mi_subcategory_affinity",
    "pair_mi_entity_recency":"pair_mi_entity_affinity",
    "pair_text_semantic_top1_quantile":"pair_text_semantic_top3_mean",
    "pair_text_semantic_top3_mean_quantile":"pair_text_semantic_top3_mean",
    "pair_text_semantic_top5_mean_quantile":"pair_text_semantic_top3_mean",
    "pair_text_semantic_attention_t8_quantile":"pair_text_semantic_top3_mean",
    "pair_text_semantic_attention_t12_quantile":"pair_text_semantic_top3_mean",
    "pair_text_semantic_centroid_quantile":"pair_text_semantic_top3_mean",
    "pair_text_semantic_recent5_centroid_quantile":"pair_text_semantic_top3_mean",
    "pair_text_semantic_recent5_max_quantile":"pair_text_semantic_top3_mean",
    "pair_text_semantic_last20_decay_quantile":"pair_text_semantic_top3_mean",
    "pair_text_semantic_top1":"pair_text_semantic_top3_mean",
    "pair_text_semantic_top5_mean":"pair_text_semantic_top3_mean",
    "pair_text_semantic_attention_t8":"pair_text_semantic_top3_mean",
    "pair_text_semantic_attention_t12":"pair_text_semantic_top3_mean",
    "pair_text_semantic_centroid":"pair_text_semantic_top3_mean",
    "pair_text_semantic_recent5_centroid":"pair_text_semantic_top3_mean",
    "pair_text_semantic_recent5_max":"pair_text_semantic_top3_mean",
    "pair_text_semantic_last20_decay":"pair_text_semantic_top3_mean",
}
PAIR_FEATURE_PROFILES = {
    **{f"text_semantic_recency_h{half_life}": (
        "pair_long_affinity", "pair_subcategory_affinity", "pair_title_overlap",
        "pair_entity_recent_top1_similarity", "pair_recent_subcategory_transition",
        f"pair_text_semantic_recency_attention_h{half_life}_similarity",
        "pair_same_topic", "pair_same_subcategory",
    ) for half_life in (8,16)},
    # Encoders observe content; only learned, proved implications can affect
    # recommendation. These profiles deliberately need no source entity model.
    "workspace_attention": (
        "pair_long_affinity", "pair_subcategory_affinity", "pair_title_overlap",
        "pair_recent_subcategory_transition", "pair_text_semantic_attention_t8",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "workspace_centered": (
        "pair_long_affinity", "pair_subcategory_affinity", "pair_title_overlap",
        "pair_recent_subcategory_transition",
        "pair_text_semantic_centered_attention_t8_similarity",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "workspace_contextual": (
        "pair_long_affinity", "pair_subcategory_affinity", "pair_title_overlap",
        "pair_recent_subcategory_transition",
        "pair_text_semantic_centered_attention_t8_similarity",
        "pair_text_semantic_effective_support_ratio", "pair_history_scope",
        "pair_same_topic", "pair_same_subcategory",
    ),
    # These profiles use only literal words, categorical metadata and causal
    # interaction counts. No source entity annotations or embedding evidence.
    "symbolic_baseline": (
        "pair_long_affinity", "pair_subcategory_affinity", "pair_title_overlap",
        "pair_recent_subcategory_transition", "pair_same_topic", "pair_same_subcategory",
    ),
    "symbolic_precision": (
        "pair_long_topic_share_quantile", "pair_subcategory_share_quantile",
        "pair_title_history_idf_jaccard", "pair_recent_subcategory_transition",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "symbolic_lexical": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_lexical_peak_match", "pair_recent_subcategory_transition",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "symbolic_coverage": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_lexical_history_coverage", "pair_recent_subcategory_transition",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "symbolic_rich": (
        "pair_long_topic_share_quantile", "pair_subcategory_share_quantile",
        "pair_lexical_peak_match", "pair_lexical_history_coverage",
        "pair_recent_subcategory_transition", "pair_same_topic", "pair_same_subcategory",
    ),
    "symbolic_contextual": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_lexical_history_coverage", "pair_recent_subcategory_transition",
        "pair_history_scope", "pair_same_topic", "pair_same_subcategory",
    ),
    # Causally portable signals: long-term interest, fine-grained interest,
    # lexical title overlap and recent-entity semantic proximity. CTR/freshness
    # are intentionally excluded
    # because training uses causal per-impression priors while MIND validation
    # necessarily sees the end-of-training prior, creating a measured shift.
    "stable_multi_interest": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_overlap", "pair_entity_recent_top1_similarity",
        "pair_recent_subcategory_transition",
        "pair_same_topic", "pair_same_subcategory",
    ),
    # Candidate-aware sequence facts retain where the matching topic and
    # subcategory occurred in ordered pre-impression history.  This is a
    # challenger rather than a hard-coded preference: fpMiner still has to
    # discover the direction and PeTTaChainer still has to prove it.
    "sequence_multi_interest": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_overlap", "pair_entity_recent_top1_similarity",
        "pair_recent_subcategory_transition",
        "pair_recent_subcategory_affinity", "pair_topic_recency",
        "pair_subcategory_recency",
        "pair_same_topic", "pair_same_subcategory",
    ),
    # Train-only quantiles refine ties inside coarse affinity buckets without
    # introducing dataset-specific category names.  Kept as an explicit
    # promotion candidate until its held-out proof AUC beats the stable view.
    "normalized_interest": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_overlap", "pair_entity_recent_top1_similarity",
        "pair_recent_subcategory_transition",
        "pair_long_topic_share_quantile",
        "pair_recent_topic_share_quantile",
        "pair_subcategory_share_quantile",
        "pair_recent_subcategory_affinity_quantile",
        "pair_topic_recency_quantile", "pair_subcategory_recency_quantile",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "full_quantile": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_overlap", "pair_entity_recent_top1_similarity",
        "pair_recent_subcategory_transition",
        "pair_entity_recent_top1_similarity_quantile",
        "pair_recent_subcategory_transition_quantile",
        "pair_long_topic_share_quantile",
        "pair_recent_topic_share_quantile",
        "pair_subcategory_share_quantile",
        "pair_recent_subcategory_affinity_quantile",
        "pair_topic_recency_quantile", "pair_subcategory_recency_quantile",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "semantic_consensus": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_history_idf_jaccard",
        "pair_entity_recent_top1_similarity",
        "pair_entity_long_mean_similarity",
        "pair_recent_subcategory_transition",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "candidate_aware_multi_interest": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_overlap", "pair_entity_recent_top1_similarity",
        "pair_recent_subcategory_transition",
        "pair_mi_topic_affinity", "pair_mi_topic_recency",
        "pair_mi_subcategory_affinity", "pair_mi_subcategory_recency",
        "pair_mi_entity_affinity", "pair_mi_entity_recency",
        "pair_mi_semantic_top1", "pair_mi_semantic_topk_mean",
        "pair_mi_semantic_weighted", "pair_mi_semantic_attention",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "candidate_aware_sparse": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_overlap", "pair_entity_recent_top1_similarity",
        "pair_recent_subcategory_transition",
        "pair_mi_semantic_top1", "pair_mi_subcategory_recency",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "candidate_aware_quantile": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_overlap", "pair_recent_subcategory_transition",
        "pair_mi_topic_affinity_quantile", "pair_mi_topic_recency_quantile",
        "pair_mi_subcategory_affinity_quantile",
        "pair_mi_subcategory_recency_quantile",
        "pair_mi_entity_affinity_quantile", "pair_mi_entity_recency_quantile",
        "pair_mi_semantic_top1_quantile",
        "pair_mi_semantic_topk_mean_quantile",
        "pair_mi_semantic_weighted_quantile",
        "pair_mi_semantic_attention_quantile",
        "pair_same_topic", "pair_same_subcategory",
    ),
    # Frozen sentence embeddings are observations, not a hidden recommender.
    # fpMiner must discover whether their candidate-relative direction predicts
    # the target, and PeTTaChainer must prove the grounded preference.
    "text_semantic_top3": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_overlap", "pair_entity_recent_top1_similarity",
        "pair_recent_subcategory_transition",
        "pair_text_semantic_top3_mean",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "text_semantic_attention": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_overlap", "pair_entity_recent_top1_similarity",
        "pair_recent_subcategory_transition",
        "pair_text_semantic_attention_t8",
        "pair_text_semantic_attention_t12",
        "pair_same_topic", "pair_same_subcategory",
    ),
    # The T=8 sensor was the stronger confirmation-split challenger and keeps
    # the proof graph compact: one semantic dependency plus the five stable
    # symbolic dependencies.  The two-temperature profile above remains an
    # explicit ablation, not the serving default.
    "text_semantic_attention_t8": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_overlap", "pair_entity_recent_top1_similarity",
        "pair_recent_subcategory_transition",
        "pair_text_semantic_attention_t8",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "text_semantic_multi_interest": (
        "pair_long_affinity", "pair_subcategory_affinity",
        "pair_title_overlap", "pair_entity_recent_top1_similarity",
        "pair_recent_subcategory_transition",
        "pair_text_semantic_top1", "pair_text_semantic_top3_mean",
        "pair_text_semantic_attention_t8", "pair_text_semantic_attention_t12",
        "pair_text_semantic_centroid", "pair_text_semantic_recent5_max",
        "pair_text_semantic_last20_decay",
        "pair_same_topic", "pair_same_subcategory",
    ),
    "all": PAIR_FEATURES,
}
PAIR_FEATURE_PROFILES["text_semantic_attention_t8_no_lexical"] = tuple(
    predicate for predicate in PAIR_FEATURE_PROFILES["text_semantic_attention_t8"]
    if predicate != "pair_title_overlap"
)
PAIR_FEATURE_PROFILES["text_semantic_attention_t8_quantile"] = tuple(
    "pair_text_semantic_attention_t8_quantile"
    if predicate=="pair_text_semantic_attention_t8" else predicate
    for predicate in PAIR_FEATURE_PROFILES["text_semantic_attention_t8"]
)
PAIR_FEATURE_PROFILES["text_semantic_attention_t8_magnitude_backoff"] = (
    *PAIR_FEATURE_PROFILES["text_semantic_attention_t8"],
    "pair_text_semantic_attention_t8_quantile",
)
PAIR_FEATURE_PROFILES["text_semantic_attention_t8_relational"] = (
    *PAIR_FEATURE_PROFILES["text_semantic_attention_t8"],
    "pair_rel_concept_continuity_scope",
)
PAIR_FEATURE_PROFILES["llm_content"] = (
    *PAIR_FEATURE_PROFILES["text_semantic_attention_t8"],
    *(f"pair_{name}" for name in LLM_NUMERIC_FEATURES),
)
PAIR_FEATURE_PROFILES["llm_content_no_lexical"] = (
    *PAIR_FEATURE_PROFILES["text_semantic_attention_t8_no_lexical"],
    *(f"pair_{name}" for name in LLM_NUMERIC_FEATURES),
)
PAIR_FEATURE_PROFILES["llm_only"] = tuple(f"pair_{name}" for name in LLM_NUMERIC_FEATURES)
PAIR_FEATURE_PROFILES["llm_content_quantile"] = (
    *PAIR_FEATURE_PROFILES["text_semantic_attention_t8"],
    *LLM_QUANTILE_PAIR_PREDICATES,
)
# Conditional mining needs orientation-invariant context.  History scope is
# deliberately absent from the ordinary LLM profile: by itself it cannot say
# whether the left or right candidate should win, but it can safely gate a
# semantic direction discovered by the real miner.
PAIR_FEATURE_PROFILES["llm_conditional"] = (
    *PAIR_FEATURE_PROFILES["llm_content"], "pair_history_scope",
)
PAIR_FEATURE_PROFILES["llm_conditional_quantile"] = (
    *PAIR_FEATURE_PROFILES["llm_content_quantile"], "pair_history_scope",
)
PAIR_FEATURE_PROFILES["llm_conditional_quantile_relational"] = (
    *PAIR_FEATURE_PROFILES["llm_conditional_quantile"],
    "pair_rel_concept_continuity_scope",
)
PAIR_FEATURE_PROFILES["llm_conditional_magnitude_backoff"] = (
    *PAIR_FEATURE_PROFILES["llm_conditional_quantile"],
    "pair_text_semantic_attention_t8_quantile",
)
PAIR_FEATURE_PROFILES["scoped_taxonomy"] = (
    *PAIR_FEATURE_PROFILES["text_semantic_attention_t8"],
    "pair_history_scope",
    "pair_left_topic", "pair_right_topic",
    "pair_left_subcategory", "pair_right_subcategory",
)
PAIR_FEATURE_PROFILES["llm_conditional_scoped_taxonomy"] = (
    *PAIR_FEATURE_PROFILES["llm_conditional_magnitude_backoff"],
    "pair_left_topic", "pair_right_topic",
    "pair_left_subcategory", "pair_right_subcategory",
    "pair_left_llm_format", "pair_right_llm_format",
)
LLM_PAIR_PREDICATES = frozenset(
    (*(f"pair_{name}" for name in LLM_NUMERIC_FEATURES),
     *LLM_QUANTILE_PAIR_PREDICATES)
)
CONDITIONAL_LLM_CONTEXT_PREDICATES = frozenset({
    "pair_history_scope", "pair_same_topic", "pair_same_subcategory",
})
PAIR_MARGIN_POWER_MIN = 0.05
PAIR_MARGIN_POWER_MAX = 4.0
CTV_EVIDENCE_K_DEFAULT = DEFAULT_EVIDENCE_K
DEFAULT_SERVING_REASONER_TIMEOUT_SECONDS = 30.0
DEFAULT_BENCHMARK_REASONER_TIMEOUT_SECONDS = 600.0
REASONER_TIMEOUT_SECONDS_MAX = 3600.0
# A pairwise reranker is quadratic when every candidate is compared with every
# other candidate.  Keep that cost explicit and bounded even if a caller
# accidentally admits a corpus-sized slate.  Larger experiments must select a
# finite cyclic opponent count instead of silently allocating O(n^2) work.
DEFAULT_MAX_PAIR_COMPARISONS = 32768
HARD_MAX_PAIR_COMPARISONS = 32768
DEFAULT_MAX_TOTAL_PAIR_COMPARISONS = 1_000_000
HARD_MAX_TOTAL_PAIR_COMPARISONS = 5_000_000
DEFAULT_MAX_PROOF_CACHE_ENTRIES = 250_000
HARD_MAX_PROOF_CACHE_ENTRIES = 1_000_000
# Architecture comparisons intentionally accept a narrow, audited override
# surface.  Evaluation cohort controls and unrelated serving settings remain
# frozen from the active champion.
CHALLENGER_CONFIG_KEYS = frozenset({
    "miner_strategy", "conjunctions", "pair_conjunctions",
    "pair_feature_profile", "pair_margin_power", "pair_max_rules",
    "ctv_evidence_k",
    "pair_ctv_mode", "pair_rule_selection_k", "pair_ctv_evidence_k",
    "pair_dependency_mode", "pair_family_fusion",
    "relational_evidence_mode",
})
MINING_CONFIG_KEYS = frozenset({
    "miner_strategy", "min_support", "max_rules", "conjunctions", "negative_ratio",
    "rule_rank", "max_feature_values", "feature_profile", "random_seed",
    "ctv_evidence_k",
    "pair_min_support", "pair_max_rules", "pair_negative_ratio",
    "pair_conjunctions", "pair_numeric_bins", "pair_min_effect",
    "pair_feature_profile",
    "pair_ctv_mode", "pair_rule_selection_k", "pair_ctv_evidence_k",
    "pair_dependency_mode",
    "relational_evidence_mode",
    "mining_retention_max_units", "mining_retention_max_cases",
})
FEATURE_PROFILES = {
    "symbolic_only": ("topic", "recent_affinity", "long_affinity",
                      "history_topic_count_bucket", "recent_topic_count_bucket"),
    "core": ("topic","format","affinity"),
    "accuracy": ("topic","recent_affinity","long_affinity","entity_overlap"),
    # Useful experimental ablation: title-length format adds candidate
    # resolution, but the full replay should decide whether its extra proofs
    # justify the latency for a given dataset.
    "accuracy_plus": ("topic","format","recent_affinity","long_affinity","entity_overlap"),
    "accuracy_detail": ("topic","recent_affinity","long_affinity","entity_overlap",
                         "entity_overlap_detail","history_topic_count_bucket",
                         "recent_topic_count_bucket"),
    "accuracy_detail_relational": (
        "topic","recent_affinity","long_affinity","entity_overlap",
        "entity_overlap_detail","history_topic_count_bucket",
        "recent_topic_count_bucket","rel_concept_continuity_scope",
    ),
    "accuracy_sequence": ("topic","recent_affinity","long_affinity","entity_overlap",
                            "entity_overlap_detail","history_topic_count_bucket",
                            "recent_topic_count_bucket","topic_recency_bucket",
                            "subcategory_recency_bucket"),
    # Source impression position is available only in offline MIND replay.
    # It is deliberately opt-in because a live candidate has no historical
    # position and would otherwise receive an uninformative ``unknown`` fact.
    "accuracy_position": ("topic","recent_affinity","long_affinity","entity_overlap",
                           "position_bucket"),
    "accuracy_content": ("topic","recent_affinity","long_affinity","entity_overlap",
                          "entity_overlap_detail","history_topic_count_bucket",
                          "recent_topic_count_bucket","title_overlap_detail"),
    "accuracy_rich": ("topic","subcategory","format","recent_affinity","long_affinity",
                      "entity_overlap","entity_overlap_detail","history_topic_count_bucket",
                      "recent_topic_count_bucket","topic_rank_bucket","subcategory_affinity"),
    "candidate_aware": (
        "topic", "subcategory", "recent_affinity", "long_affinity",
        "entity_overlap_detail", "mi_topic_candidate_match_rank",
        "mi_subcategory_candidate_match_rank", "mi_entity_candidate_match_rank",
    ),
    "all": FEATURES,
}

# Entity and canonical-concept continuity can be derived from the same
# historical click.  Until the final scorer can fuse dependencies by their
# shared origin key, activating both families would let one observation vote
# through two independently mined channels.  Keep this invariant at the
# effective point+pair profile boundary; checking only each profile in
# isolation would miss a point-entity / pair-concept combination.
RELATIONAL_ENTITY_SCORE_PREDICATES = frozenset({
    "rel_entity_continuity_scope",
    "pair_rel_entity_continuity_scope",
})
RELATIONAL_CONCEPT_SCORE_PREDICATES = frozenset({
    "rel_concept_continuity_scope",
    "pair_rel_concept_continuity_scope",
})


def _validate_relational_score_dependencies(
        point_predicates, pair_predicates):
    """Reject relational families that can reuse one interaction as two votes."""
    active=frozenset((*point_predicates,*pair_predicates))
    active_entity=RELATIONAL_ENTITY_SCORE_PREDICATES.intersection(active)
    active_concept=RELATIONAL_CONCEPT_SCORE_PREDICATES.intersection(active)
    if active_entity and active_concept:
        raise ValueError(
            "entity and concept relational scoring features cannot be active "
            "together until shared-origin dependency-aware fusion is "
            "implemented"
        )


INTERACTION_PAIRS = (
    ("topic","affinity"), ("format","affinity"),
    ("topic","affinity_level"), ("subcategory","affinity_level"),
    ("topic","recent_affinity"), ("topic","long_affinity"),
    ("recent_affinity","long_affinity"), ("entity_overlap","affinity_level"),
    ("ctr_bucket","freshness_bucket"), ("topic","time_bucket"),
    ("history_size_bucket","affinity_level"), ("format","affinity_level"),
    ("entity_overlap","recent_affinity"), ("ctr_bucket","recent_affinity"),
    ("history_size_bucket","recent_affinity"),
    ("topic","history_topic_count_bucket"), ("topic","recent_topic_count_bucket"),
    ("entity_overlap_detail","recent_topic_count_bucket"),
    ("entity_overlap_detail","history_topic_count_bucket"),
    ("title_overlap_detail","topic"),
    ("title_overlap_detail","recent_topic_count_bucket"),
    ("recent_topic_count_bucket","long_affinity"),
    ("subcategory_affinity","topic"),
    ("rel_entity_continuity_scope","recent_affinity"),
    ("rel_entity_continuity_scope","long_affinity"),
    ("rel_concept_continuity_scope","recent_affinity"),
    ("rel_concept_continuity_scope","long_affinity"),
)
INTERACTION_TRIPLES = (
    ("topic","recent_affinity","long_affinity"),
    ("topic","entity_overlap","affinity_level"),
    ("topic","ctr_bucket","freshness_bucket"),
    ("subcategory","entity_overlap","affinity_level"),
)
PAIR_INTERACTIONS = (
    ("pair_recent_affinity", "pair_long_affinity"),
    ("pair_recent_affinity", "pair_entity_overlap"),
    ("pair_long_affinity", "pair_history_topic_count"),
    ("pair_entity_overlap", "pair_title_overlap"),
    ("pair_recent_topic_count", "pair_topic_rank"),
    ("pair_subcategory_affinity", "pair_entity_overlap"),
    ("pair_ctr", "pair_freshness"),
    ("pair_same_topic", "pair_recent_affinity"),
    ("pair_long_affinity", "pair_title_overlap"),
    ("pair_long_affinity", "pair_subcategory_affinity"),
    ("pair_title_overlap", "pair_subcategory_affinity"),
    ("pair_entity_recent_top1_similarity", "pair_long_affinity"),
    ("pair_entity_recent_top1_similarity", "pair_subcategory_affinity"),
    ("pair_entity_recent_top1_similarity", "pair_title_overlap"),
    ("pair_recent_subcategory_transition", "pair_long_affinity"),
    ("pair_recent_subcategory_transition", "pair_subcategory_affinity"),
    ("pair_recent_subcategory_transition", "pair_title_overlap"),
    ("pair_subcategory_recency", "pair_recent_subcategory_transition"),
    ("pair_subcategory_recency", "pair_subcategory_affinity"),
    ("pair_topic_recency", "pair_long_affinity"),
    ("pair_recent_subcategory_affinity", "pair_subcategory_affinity"),
    ("pair_title_history_idf_jaccard", "pair_long_affinity"),
    ("pair_title_history_idf_jaccard", "pair_subcategory_affinity"),
    ("pair_entity_long_mean_similarity", "pair_long_affinity"),
    ("pair_entity_long_mean_similarity", "pair_subcategory_affinity"),
    ("pair_entity_long_mean_similarity", "pair_title_history_idf_jaccard"),
    ("pair_text_semantic_top3_mean", "pair_long_affinity"),
    ("pair_text_semantic_top3_mean", "pair_subcategory_affinity"),
    ("pair_text_semantic_top3_mean", "pair_recent_subcategory_transition"),
    ("pair_text_semantic_top3_mean", "pair_entity_recent_top1_similarity"),
    ("pair_text_semantic_attention_t8", "pair_long_affinity"),
    ("pair_text_semantic_attention_t8", "pair_subcategory_affinity"),
    ("pair_text_semantic_attention_t8", "pair_recent_subcategory_transition"),
    ("pair_text_semantic_attention_t8", "pair_entity_recent_top1_similarity"),
    ("pair_text_semantic_attention_t12", "pair_long_affinity"),
    ("pair_text_semantic_attention_t12", "pair_subcategory_affinity"),
    ("pair_text_semantic_attention_t12", "pair_recent_subcategory_transition"),
    ("pair_text_semantic_attention_t12", "pair_entity_recent_top1_similarity"),
    ("pair_rel_entity_continuity_scope", "pair_long_affinity"),
    ("pair_rel_entity_continuity_scope", "pair_subcategory_affinity"),
    ("pair_rel_entity_continuity_scope", "pair_text_semantic_attention_t8"),
    ("pair_rel_concept_continuity_scope", "pair_long_affinity"),
    ("pair_rel_concept_continuity_scope", "pair_subcategory_affinity"),
    ("pair_rel_concept_continuity_scope", "pair_text_semantic_attention_t8"),
)
PAIR_ORDERS = {
    "affinity": ("low", "high"),
    "recent_affinity": ("none", "low", "medium", "high"),
    "long_affinity": ("none", "low", "medium", "high"),
    "entity_overlap_detail": ("none", "one", "two", "three_plus"),
    "rel_entity_continuity_scope": ("none", "older", "recent"),
    "rel_concept_continuity_scope": ("none", "older", "recent"),
    "history_topic_count_bucket": ("zero", "one", "two", "three_plus"),
    "recent_topic_count_bucket": ("zero", "one", "two", "three_plus"),
    "topic_rank_bucket": ("none", "secondary", "top"),
    "subcategory_affinity": ("none", "low", "medium", "high"),
    "title_overlap_detail": ("none", "one", "two", "three_plus"),
    "ctr_bucket": ("cold", "low", "medium", "high"),
    "freshness_bucket": ("established", "recent", "new"),
    "format": ("short", "medium", "long"),
    "topic_recency_bucket": ("none", "older", "recent", "immediate"),
    "subcategory_recency_bucket": ("none", "older", "recent", "immediate"),
}
# Immutable pair-feature plans. These used to be rebuilt inside
# ``_pair_features`` for every orientation of every comparison.
PAIR_ORDERED_COMPARISONS = {
    "pair_affinity":"affinity", "pair_recent_affinity":"recent_affinity",
    "pair_long_affinity":"long_affinity", "pair_entity_overlap":"entity_overlap_detail",
    "pair_rel_entity_continuity_scope":"rel_entity_continuity_scope",
    "pair_rel_concept_continuity_scope":"rel_concept_continuity_scope",
    "pair_history_topic_count":"history_topic_count_bucket",
    "pair_recent_topic_count":"recent_topic_count_bucket",
    "pair_topic_rank":"topic_rank_bucket",
    "pair_subcategory_affinity":"subcategory_affinity",
    "pair_title_overlap":"title_overlap_detail", "pair_ctr":"ctr_bucket",
    "pair_freshness":"freshness_bucket", "pair_format":"format",
}
PAIR_LEXICAL_COMPARISONS = {
    f"pair_{source}":source for source in LEXICAL_FEATURES
}
PAIR_SEMANTIC_WORKSPACE_COMPARISONS = {
    f"pair_{source}":source for source in SEMANTIC_WORKSPACE_FEATURES
}
PAIR_DIRECT_NUMERIC_COMPARISONS = {
    "pair_entity_recent_top1_similarity":"entity_recent_top1_similarity",
    "pair_recent_subcategory_transition":"recent_subcategory_transition_score",
    "pair_recent_subcategory_affinity":"recent_subcategory_affinity_score",
    "pair_topic_recency":"topic_recency_score",
    "pair_subcategory_recency":"subcategory_recency_score",
    "pair_entity_long_mean_similarity":"entity_long_mean_similarity",
    "pair_title_history_idf_jaccard":"title_history_idf_jaccard",
}
PAIR_MULTI_INTEREST_COMPARISONS = {
    **{f"pair_{name}":name for name in LLM_NUMERIC_FEATURES},
    **{f"pair_{name}":name for name in RECENCY_WORKSPACE_FEATURES},
    "pair_mi_topic_affinity":"mi_topic_candidate_affinity_score",
    "pair_mi_topic_recency":"mi_topic_candidate_recency_score",
    "pair_mi_subcategory_affinity":"mi_subcategory_candidate_affinity_score",
    "pair_mi_subcategory_recency":"mi_subcategory_candidate_recency_score",
    "pair_mi_entity_affinity":"mi_entity_candidate_affinity_score",
    "pair_mi_entity_recency":"mi_entity_candidate_recency_score",
    "pair_mi_semantic_top1":"mi_semantic_top1_similarity",
    "pair_mi_semantic_topk_mean":"mi_semantic_topk_mean_similarity",
    "pair_mi_semantic_weighted":"mi_semantic_weighted_similarity",
    "pair_mi_semantic_attention":"mi_semantic_attention_score",
    "pair_text_semantic_top1":"text_semantic_top1_similarity",
    "pair_text_semantic_top3_mean":"text_semantic_top3_mean_similarity",
    "pair_text_semantic_top5_mean":"text_semantic_top5_mean_similarity",
    "pair_text_semantic_attention_t8":"text_semantic_attention_t8_similarity",
    "pair_text_semantic_attention_t12":"text_semantic_attention_t12_similarity",
    "pair_text_semantic_centroid":"text_semantic_centroid_similarity",
    "pair_text_semantic_recent5_centroid":"text_semantic_recent5_centroid_similarity",
    "pair_text_semantic_recent5_max":"text_semantic_recent5_max_similarity",
    "pair_text_semantic_last20_decay":"text_semantic_last20_recency_decayed_similarity",
}
PAIR_STABLE_DOMINANCE_SOURCES = {
    "pair_long_affinity":"long_affinity",
    "pair_subcategory_affinity":"subcategory_affinity",
    "pair_title_overlap":"title_overlap_detail",
}
PAIR_SIDE_PREDICATE_SWAP = {
    left:right for left,right in PAIR_CATEGORICAL_SIDE_FAMILIES.values()
} | {
    right:left for left,right in PAIR_CATEGORICAL_SIDE_FAMILIES.values()
}


@lru_cache(maxsize=128)
def _pair_feature_execution_plan(needed):
    """Compile the immutable predicate loops for one feature vocabulary."""
    selected=None if needed is None else frozenset(needed)
    choose=lambda mapping: tuple(
        (predicate,source) for predicate,source in mapping.items()
        if selected is None or predicate in selected
    )
    return {
        "lexical":choose(PAIR_LEXICAL_COMPARISONS),
        "semantic_workspace":choose(PAIR_SEMANTIC_WORKSPACE_COMPARISONS),
        "ordered":choose(PAIR_ORDERED_COMPARISONS),
        "direct_numeric":choose(PAIR_DIRECT_NUMERIC_COMPARISONS),
        "multi_interest":choose(PAIR_MULTI_INTEREST_COMPARISONS),
        "quantile":tuple(
            (source,predicate) for source,predicate in NUMERIC_PAIR_EVIDENCE.items()
            if selected is None or predicate in selected
        ),
    }


def _pair_candidate_source_features(predicates):
    """Return candidate facts required to ground promoted pair predicates."""
    selected=frozenset(predicates)
    plan=_pair_feature_execution_plan(tuple(sorted(selected)))
    sources={source for family in (
        "lexical","semantic_workspace","ordered","direct_numeric",
        "multi_interest",
    ) for _predicate,source in plan[family]}
    sources.update(source for source,_predicate in plan["quantile"])
    if "pair_history_scope" in selected:
        sources.add("history_size_bucket")
    if "pair_stable_dominance" in selected:
        sources.update(PAIR_STABLE_DOMINANCE_SOURCES.values())
    if selected & {
        "pair_left_llm_format","pair_right_llm_format",
    }:
        sources.add("llm_format")
    return frozenset(sources)


MINER_LOCK = threading.RLock()
MINING_WORKSPACE_SCHEMA_VERSION = 1
MINING_WORKSPACE_MODE = "incremental_fpminer_support_full_population_ctv_estimation"
SERVING_MODEL_SCHEMA = "mindplex-symbolic-serving-model-v1"
BACKGROUND_MINING_BUILD_MODE = (
    "incremental_fpminer_support_staged_snapshot"
)
BACKGROUND_MODEL_FIELDS = (
    "_feature_vocabulary", "_pair_feature_vocabulary",
    "_pair_categorical_labels", "_numeric_pair_encoders",
    "mined_rules", "mined_output", "_point_rule_sources",
    "_point_channel_sources",
    "pair_rules", "_pair_rule_sources",
    "_active_rule_ids", "_active_pair_rule_ids", "last_pair_mining",
    "_click_base_rate", "_tie_break_stats",
    "_active_mining_workspace_plans",
)
SEMANTIC_CACHE_MAX_BYTES = 32 * 1024 * 1024
SEMANTIC_CACHE_MAX_ENTRIES = 512
SEMANTIC_CACHE_TTL_SECONDS = 3600.0
SEMANTIC_SINGLE_FLIGHT_WAIT_SECONDS = 1.0


def fixture():
    articles = [
        {"id":"n1","title":"A practical guide to neural networks","topic":"ai","format":"guide"},
        {"id":"n2","title":"AI policy and public trust","topic":"ai","format":"news"},
        {"id":"n3","title":"How scientists study galaxies","topic":"science","format":"explainer"},
        {"id":"n4","title":"New evidence about ocean warming","topic":"climate","format":"news"},
        {"id":"n5","title":"A short history of climate policy","topic":"climate","format":"essay"},
        {"id":"n6","title":"The history of computing","topic":"history","format":"essay"},
        {"id":"n7","title":"A beginner map of quantum physics","topic":"science","format":"guide"},
        {"id":"n8","title":"What ancient cities teach us","topic":"history","format":"explainer"},
    ]
    users = {"u1":["ai","science"],"u2":["climate"],"u3":["history","ai"],"u4":["science","climate"]}
    events = []
    for repeat in range(4):
        for user, interests in users.items():
            for article in articles:
                high = article["topic"] in interests
                clicked = high and (article["format"] != "essay" or repeat % 2 == 0)
                events.append({"user":user,"article":article["id"],
                               "action":"click" if clicked else "skip",
                               "impression":f"fixture_{repeat}_{user}"})
    tests = [{"user":u,"candidates":[a["id"] for a in articles],
              "relevant":[a["id"] for a in articles if a["topic"] in topics and a["format"] != "essay"]}
             for u, topics in users.items()]
    return {"users":users,"articles":articles,"events":events,"tests":tests}


def script_json(value):
    """Serialize bootstrap data without allowing an inline-script close tag."""
    return json.dumps(value,ensure_ascii=False).replace("<","\\u003c")


def first_paint_feed(page):
    """Render a small progressive first paint before JavaScript takes over."""
    cards=[]
    for row in page.get("feed",[]):
        article=row["article"]; stv=row.get("stv",{})
        details=" · ".join(str(article.get(key)) for key in ("topic","format","subcategory")
                           if article.get(key))
        cards.append(
            '<article class="card">'
            f'<span class="score">{float(row.get("ranking_score",row.get("score",0))):.6f}</span>'
            f'<b>{html_lib.escape(str(article.get("title",article["id"])))}</b>'
            f'<div class="muted">{html_lib.escape(details)} · decision STV '
            f'{float(stv.get("strength",0)):.3f}/{float(stv.get("confidence",0)):.3f}</div>'
            '</article>'
        )
    return "".join(cards)


BenchmarkMixin = make_benchmark_mixin(
    CHALLENGER_CONFIG_KEYS=CHALLENGER_CONFIG_KEYS,
    CONTEXT_FEATURES=CONTEXT_FEATURES,
    MINING_CONFIG_KEYS=MINING_CONFIG_KEYS,
    PAIR_FEATURE_PROFILES=PAIR_FEATURE_PROFILES,
    PAIR_MARGIN_POWER_MAX=PAIR_MARGIN_POWER_MAX,
    PAIR_MARGIN_POWER_MIN=PAIR_MARGIN_POWER_MIN,
)
PairwiseRankingMixin = make_pairwise_ranking_mixin(
    CONTEXT_FEATURES=CONTEXT_FEATURES,
    DEFAULT_MAX_PROOF_CACHE_ENTRIES=DEFAULT_MAX_PROOF_CACHE_ENTRIES,
    LIVE_NEGATIVE_FEATURE=LIVE_NEGATIVE_FEATURE,
    LIVE_NEGATIVE_RULE_IDS=LIVE_NEGATIVE_RULE_IDS,
    PAIR_CATEGORICAL_SIDE_PREDICATES=PAIR_CATEGORICAL_SIDE_PREDICATES,
    PAIR_SIDE_PREDICATE_SWAP=PAIR_SIDE_PREDICATE_SWAP,
    RELATIONAL_PROOF_FIELDS=RELATIONAL_PROOF_FIELDS,
)
ServingMixin = make_serving_mixin(
    CONTEXT_FEATURES=CONTEXT_FEATURES,
    DEFAULT_MAX_PROOF_CACHE_ENTRIES=DEFAULT_MAX_PROOF_CACHE_ENTRIES,
    FEATURE_PROFILES=FEATURE_PROFILES,
    FEED_DELIVERY_HISTORY_LIMIT=FEED_DELIVERY_HISTORY_LIMIT,
    FEED_DELIVERY_ROW_LIMIT=FEED_DELIVERY_ROW_LIMIT,
    LIVE_NEGATIVE_FEATURE=LIVE_NEGATIVE_FEATURE,
    LIVE_NEGATIVE_GENERALIZATION_WINDOW=LIVE_NEGATIVE_GENERALIZATION_WINDOW,
    LIVE_NEGATIVE_RULE_IDS=LIVE_NEGATIVE_RULE_IDS,
    pair_candidate_source_features=_pair_candidate_source_features,
    RELATIONAL_LIVE_HISTORY_LIMIT=RELATIONAL_LIVE_HISTORY_LIMIT,
    RELATIONAL_PROOF_FIELDS=RELATIONAL_PROOF_FIELDS,
    RELATIONAL_QUERY_STEPS_PER_ROOT=RELATIONAL_QUERY_STEPS_PER_ROOT,
    SEMANTIC_CACHE_MAX_BYTES=SEMANTIC_CACHE_MAX_BYTES,
    SEMANTIC_CACHE_MAX_ENTRIES=SEMANTIC_CACHE_MAX_ENTRIES,
    SEMANTIC_CACHE_TTL_SECONDS=SEMANTIC_CACHE_TTL_SECONDS,
    SEMANTIC_SINGLE_FLIGHT_WAIT_SECONDS=SEMANTIC_SINGLE_FLIGHT_WAIT_SECONDS,
)
ModelLifecycleMixin = make_model_lifecycle_mixin(
    BACKGROUND_MINING_BUILD_MODE=BACKGROUND_MINING_BUILD_MODE,
    BACKGROUND_MODEL_FIELDS=BACKGROUND_MODEL_FIELDS,
    CONDITIONAL_LLM_CONTEXT_PREDICATES=CONDITIONAL_LLM_CONTEXT_PREDICATES,
    FEATURE_PROFILES=FEATURE_PROFILES,
    INTERACTION_PAIRS=INTERACTION_PAIRS,
    INTERACTION_TRIPLES=INTERACTION_TRIPLES,
    LIVE_NEGATIVE_RULE_SOURCES=LIVE_NEGATIVE_RULE_SOURCES,
    LLM_PAIR_PREDICATES=LLM_PAIR_PREDICATES,
    MINER_LOCK=MINER_LOCK,
    MINING_WORKSPACE_MODE=MINING_WORKSPACE_MODE,
    MINING_WORKSPACE_SCHEMA_VERSION=MINING_WORKSPACE_SCHEMA_VERSION,
    NUMERIC_PAIR_EVIDENCE=NUMERIC_PAIR_EVIDENCE,
    PAIR_CATEGORICAL_SIDE_FAMILIES=PAIR_CATEGORICAL_SIDE_FAMILIES,
    PAIR_CATEGORICAL_SIDE_PREDICATES=PAIR_CATEGORICAL_SIDE_PREDICATES,
    PAIR_EVIDENCE_ALIASES=PAIR_EVIDENCE_ALIASES,
    PAIR_FEATURES=PAIR_FEATURES,
    PAIR_FEATURE_PROFILES=PAIR_FEATURE_PROFILES,
    PAIR_INTERACTIONS=PAIR_INTERACTIONS,
    PAIR_ORDERS=PAIR_ORDERS,
    PAIR_REDUNDANT_INTEREST=PAIR_REDUNDANT_INTEREST,
    PAIR_STABLE_DOMINANCE_SOURCES=PAIR_STABLE_DOMINANCE_SOURCES,
    RELATIONAL_PROOF_FIELDS=RELATIONAL_PROOF_FIELDS,
    SERVING_MODEL_SCHEMA=SERVING_MODEL_SCHEMA,
    _pair_feature_execution_plan=_pair_feature_execution_plan,
)


class Lab(
    ServingMixin, ModelLifecycleMixin, PairwiseRankingMixin, BenchmarkMixin,
):
    def __init__(self, data=None, *, symbolic_only=False, config=None,
                 serving_model=None, serving_only=False):
        self.symbolic_only=bool(symbolic_only)
        self.serving_only=bool(serving_only)
        if self.symbolic_only and data is not None:
            from ..pipelines.symbolic_data import strip_neural_evidence
            data=strip_neural_evidence(data)
        self.lock = threading.RLock(); self.data = data or fixture()
        self._semantic_workspace_model=self.data.get("semantic_workspace_model") or {}
        self._recency_workspace=(self.data.get("metadata") or {}).get("recency_workspace") or {}
        self._llm_workspace=(self.data.get("metadata") or {}).get("llm_workspace") or {}
        self._relational_workspace=(
            (self.data.get("metadata") or {}).get("relational_workspace") or {}
        )
        self._llm_article_annotations=self.data.get("llm_article_annotations") or {}
        if self._llm_workspace or self._llm_article_annotations:
            annotation_hash=hashlib.sha256(json.dumps(
                self._llm_article_annotations,ensure_ascii=False,sort_keys=True,
                separators=(",",":"),allow_nan=False,
            ).encode("utf-8")).hexdigest()
            if (self._llm_workspace.get("observation_schema")!=LLM_WORKSPACE_SCHEMA
                    or annotation_hash!=self._llm_workspace.get("annotation_records_sha256")):
                raise ValueError("LLM annotations do not match frozen workspace provenance")
            corpus=_canonical_corpus(
                ((article["id"],article.get("source_id",article["id"]),
                  article.get("title"),article.get("abstract"))
                 for article in self.data["articles"]),source={"kind":"llm-runtime"},
            )
            if corpus.content_sha256!=self._llm_workspace.get("article_corpus_content_sha256"):
                raise ValueError("LLM workspace article content does not match its frozen corpus")
        # Keep presentation metadata outside the scoring workspace.  These
        # labels make the real MIND/LLM corpus useful to a reader-facing UI,
        # but never become hidden ranking inputs.
        self._article_presentation={
            str(article_id):self._presentation_from_annotation(annotation)
            for article_id,annotation in self._llm_article_annotations.items()
        }
        if self._relational_workspace:
            if self._relational_workspace.get("schema")!=RELATIONAL_WORKSPACE_SCHEMA:
                raise ValueError("unsupported relational workspace schema")
            # The projection owns both its proof ledger and every categorical
            # reference stored in historical contexts. Validate before mining
            # so a tampered fact can never become a behavioral rule.
            validate_relational_projection(self.data)
        self._replay_snapshot=bool((self.data.get("metadata") or {}).get("replay_snapshot"))
        self.instance_id=uuid.uuid4().hex
        # Offline replay labels are immutable evidence.  Live feedback is kept
        # visible to the online miner, but tracked separately so an evaluator
        # cannot silently train on interactions involving its own MIND users
        # and candidate articles.
        self._offline_event_count=len(self.data.get("events",[]))
        self._online_events=[]
        try:
            self._candidate_feature_workers=int(os.environ.get(
                "RECOMMENDATION_CANDIDATE_THREADS","1"
            ))
        except ValueError as exc:
            raise ValueError(
                "RECOMMENDATION_CANDIDATE_THREADS must be an integer"
            ) from exc
        if not 1<=self._candidate_feature_workers<=16:
            raise ValueError(
                "RECOMMENDATION_CANDIDATE_THREADS must be between 1 and 16"
            )
        self._candidate_feature_executor=(ThreadPoolExecutor(
            max_workers=self._candidate_feature_workers,
            thread_name_prefix=f"recommendation-features-{self.instance_id[:8]}",
        ) if self._candidate_feature_workers>1 else None)
        self.config = {"miner_strategy":"fixed_combinations",
                       "min_support":16,"max_rules":30,"top_k":5,"conjunctions":2,"chain_steps":10,
                       "mine_interval":8,"max_candidates":0,"random_seed":7,"query_batch_size":512,
                       "mining_retention_max_units":DEFAULT_RETENTION_MAX_UNITS,
                       "mining_retention_max_cases":DEFAULT_RETENTION_MAX_CASES,
                       "serving_reasoner_timeout_seconds":DEFAULT_SERVING_REASONER_TIMEOUT_SECONDS,
                       "benchmark_reasoner_timeout_seconds":DEFAULT_BENCHMARK_REASONER_TIMEOUT_SECONDS,
                       "feed_window":40,"negative_ratio":4,"aggregation":"weighted",
                       "rule_rank":"quality","max_feature_values":24,
                       "ctv_evidence_k":CTV_EVIDENCE_K_DEFAULT,
                       "feature_profile":"accuracy_detail",
                       "relational_evidence_mode":"disabled",
                       "ranking_mode":"pairwise","pairwise_weight":1.0,
                       "pairwise_fusion":"rank",
                       "pair_aggregation":"proof_margin",
                       "pair_family_fusion":"flat_margin",
                       "pair_margin_transform":"log_odds",
                       "pair_margin_power":1.0,
                       "pair_feature_profile":"stable_multi_interest",
                       "pair_min_support":12,"pair_max_rules":40,
                       "pair_negative_ratio":8,"pair_conjunctions":2,
                       "pair_numeric_bins":4,
                       "pairwise_opponents":0,"pair_chain_steps":12,
                       "max_pair_comparisons":DEFAULT_MAX_PAIR_COMPARISONS,
                       "max_total_pair_comparisons":(
                           DEFAULT_MAX_TOTAL_PAIR_COMPARISONS
                       ),
                       "max_proof_cache_entries":(
                           DEFAULT_MAX_PROOF_CACHE_ENTRIES
                       ),
                       "pair_min_effect":0.03}
        self.config.update(
            pair_ctv_mode="raw_pairs",
            pair_rule_selection_k=20.0,
            # Deprecated configuration alias retained so frozen experiment
            # artifacts remain loadable. It is never a PeTTa CTV evidence K.
            pair_ctv_evidence_k=20.0,
            pair_dependency_mode="clustered",
        )
        if self.symbolic_only:
            self.config.update(feature_profile="symbolic_only",
                               pair_feature_profile="symbolic_baseline")
        elif self._semantic_workspace_model:
            self.config.update(feature_profile="symbolic_only",
                               pair_feature_profile="workspace_attention")
        self.runs=[]; self.pending_events=0; self.version=0; self.feed_cache=OrderedDict(); self.last_mined_at=None
        # Cumulative observability counters distinguish a real scoring pass
        # from an HTTP response served by the final ranked-feed cache.  They do
        # not affect cache keys, rules, or ranking semantics.
        self._feed_rank_cache_hits=0; self._feed_rank_cache_misses=0
        self._event_sequence=0; self._last_mined_event_sequence=0
        self._closed=False; self._background_mining=None
        self._articles={str(article["id"]):article for article in self.data["articles"]}
        self._article_entity_vectors={
            str(article_id):tuple(float(value) for value in vector)
            for article_id,vector in self.data.get("article_entity_vectors",{}).items()
        }
        self._article_text_vectors={
            str(article_id):tuple(float(value) for value in vector)
            for article_id,vector in self.data.get("article_text_vectors",{}).items()
        }
        self._text_embedding_metadata=None
        text_sidecar=(self.data.get("metadata") or {}).get("text_embedding_sidecar")
        if self._recency_workspace:
            if (self._recency_workspace.get("schema")!="mindplex-preserved-recency-projection-v1"
                    or self._recency_workspace.get("observation_schema")!=RECENCY_WORKSPACE_SCHEMA):
                raise ValueError("unsupported recency workspace formula/schema")
            if self._article_text_vectors:
                raise ValueError("recency workspace rejects unverified inline text vectors")
            recency_expected=self._recency_workspace.get("embedding_file_sha256")
            if (not text_sidecar or not recency_expected
                    or hashlib.sha256(Path(text_sidecar).read_bytes()).hexdigest()!=recency_expected):
                raise ValueError("recency workspace sidecar does not match its frozen provenance")
        if self._semantic_workspace_model:
            provenance=(self.data.get("metadata") or {}).get("semantic_workspace",{})
            if self._article_text_vectors or self._article_entity_vectors:
                raise ValueError("semantic workspace rejects unverified inline vector payloads")
            model_hash=hashlib.sha256(json.dumps(
                self._semantic_workspace_model,ensure_ascii=False,sort_keys=True,
                separators=(",",":"),
            ).encode("utf-8")).hexdigest()
            if model_hash!=provenance.get("model_sha256"):
                raise ValueError("semantic workspace background model does not match its frozen provenance")
            # Validate schema, dimensions and formula before any mining. An
            # empty observation needs no encoder or evidence to do this.
            build_semantic_workspace_facts(None,(),{},self._semantic_workspace_model)
            expected=provenance.get("embedding_file_sha256")
            if (not text_sidecar or not expected
                    or hashlib.sha256(Path(text_sidecar).read_bytes()).hexdigest()!=expected):
                raise ValueError("semantic workspace embedding sidecar does not match its frozen provenance")
        if text_sidecar and not self.symbolic_only:
            loaded_text_sidecar=load_text_embedding_sidecar(text_sidecar)
            if self._recency_workspace and hashlib.sha256(Path(text_sidecar).read_bytes()).hexdigest()!=recency_expected:
                raise ValueError("recency workspace embedding sidecar changed while loading")
            if self._semantic_workspace_model and hashlib.sha256(Path(text_sidecar).read_bytes()).hexdigest()!=expected:
                raise ValueError("semantic workspace embedding sidecar changed while loading")
            source_vectors=loaded_text_sidecar.as_mapping("source_id")
            for article_id,article in self._articles.items():
                source_id=str(article.get("source_id",article_id))
                vector=source_vectors.get(source_id)
                if vector is not None:
                    self._article_text_vectors[article_id]=vector
            self._text_embedding_metadata=dict(loaded_text_sidecar.metadata)
        self._title_idf_model=self.data.get("title_idf_model") or {}
        self._lexical_idf_model=self.data.get("lexical_idf_model") or {}
        self._popularity=Counter(event["article"] for event in self.data["events"]
                                 if event["action"] in POSITIVE)
        base=MINER_DIR
        # PeTTa's named spaces are process-global. Each deterministic mining plan
        # gets a private, persistent space whose unchanged cases can be reused or
        # extended by delta; MINER_LOCK still serializes all PeTTa mutations.
        with MINER_LOCK:
            self.petta=PeTTa()
            self.petta.load_metta_file(str(base/"helpers.metta"))
            self.petta.load_metta_file(str(base/"fpMiner.metta"))
            self._mining_workspaces=PeTTaWorkspaceCache(
                self.petta,namespace=self.instance_id,batch_size=1000,
                lock=MINER_LOCK,
            )
            self._incremental_fpminer=IncrementalFpMinerCache(
                self.petta,namespace=self.instance_id,batch_size=1000,
                lock=MINER_LOCK,
            )
        self._active_mining_workspace_plans=frozenset()
        self._staged_background_build=False
        self.engine=IsolatedPeTTaChainer()
        # This client is a separate, optional semantic-ingestion boundary.  It
        # does not replace the embedded scoring worker above: the former asks
        # NL2PLN what an article means; the latter proves mined ranking rules.
        # Invalid optional HTTP configuration must not prevent the embedded
        # miner/PeTTa proof ranker from starting.  The preview endpoint reports
        # an explicit 503 until the semantic configuration is corrected.
        self._semantic_configuration_error=None
        self.semantic_client=None
        try:
            if not self.symbolic_only:
                self.semantic_client=PeTTaChainerClient()
        except PeTTaChainerConfigurationError as exc:
            self.semantic_client=None
            self._semantic_configuration_error=str(exc)
        self._semantic_lock=threading.Lock()
        self._semantic_preview_cache=OrderedDict()
        self._semantic_preview_cache_bytes=0
        self._active_rule_ids=set(); self._loaded_candidates=set(); self._proof_cache={}
        self._loaded_point_channels=set(); self._point_channel_proof_cache={}
        self._point_channel_templates={}
        self._active_pair_rule_ids=set(); self._loaded_pairs=set(); self._pair_proof_cache={}
        self._loaded_pair_channels=set(); self._pair_channel_proof_cache={}
        self._pair_channel_templates={}
        self._pair_proof_origins={}
        self._pair_feature_vocabulary={}; self._pair_case_attrs={}; self._pair_margin_cache={}
        self._pair_categorical_labels={}
        self._numeric_pair_encoders={}
        self._point_rule_sources=[]; self._point_channel_sources=[]
        self._point_query_calls=0; self._point_query_roots=0
        self._point_pruned_query_roots=0
        self._point_channel_activations=0
        self._point_reused_channel_activations=0
        self._last_point_completeness={}
        self.pair_rules=[]; self._pair_rule_sources=[]; self.last_pair_mining=None
        self._pair_query_calls=0
        self._pair_query_roots=0; self._pair_pruned_query_roots=0
        self._pair_channel_activations=0; self._pair_reused_channel_activations=0
        self._last_point_cache_stats={}
        self._last_pair_cache_stats={}
        self._feature_vocabulary={}
        self._relational_feature_cache={}
        self._loaded_relational_statements=set()
        self._live_relational_proof_ledger={}
        self._candidate_relational_proof_refs={}
        self._relational_query_calls=0
        self._relational_query_roots=0
        self._click_base_rate=0.5
        self._tie_break_stats={}
        self._startup_mode="mine"
        self._serving_model_sha256=None
        self._startup_prewarm={"status":"not_run"}
        self._last_live_score_profile={}
        self._feed_sessions={}
        self.mined_rules=[]; self.mined_output=[]; self.last_mining=None
        if serving_model is not None:
            model_config=serving_model.get("config")
            if not isinstance(model_config,dict):
                raise ValueError("serving model has no valid config")
            if config and config!=model_config:
                raise ValueError(
                    "serving model config conflicts with requested config"
                )
            config=model_config
        if config:
            self.configure(config)
        if serving_model is None:
            self.mine()
        else:
            self._load_serving_model(serving_model)
        self._background_mining=AsyncMiningCoordinator(
            capture=lambda: self._capture_background_mining(),
            build=lambda snapshot: self._build_background_mining(snapshot),
            promote=lambda snapshot,built: self._promote_background_mining(
                snapshot,built
            ),
            should_retrigger=lambda: self._background_remine_due(),
            dispose=lambda built: self._dispose_background_mining(built),
            thread_name=f"recommendation-miner-{self.instance_id[:8]}",
        )

    def users(self):
        evaluation_users=list(dict.fromkeys(
            case.get("user",case.get("user_id"))
            for case in self.data.get("eval_impressions",self.data.get("evaluation",self.data.get("tests",[])))
            if case.get("user",case.get("user_id")) in self.data["users"]
        ))
        visible=evaluation_users or list(self.data["users"])
        return [{"id":user,"topics":self.user_topics(user)} for user in visible]
    def dataset_info(self):
        metadata=self.data.get("metadata")
        if not metadata:
            return {"name":"built-in fixture","path":"bundled","train_cases":len(self.data["events"]),
                    "eval_impressions":len(self.evaluation_cases()),"articles":len(self.data["articles"]),
                    "users":len(self.data["users"])}
        return {**metadata,"name":metadata.get("dataset","MIND"),"path":metadata.get("root"),
                "train_cases":len(self.data["events"]),"eval_impressions":len(self.evaluation_cases()),
                "articles":len(self.data["articles"]),"users":len(self.data["users"])}
    def state(self):
        semantic_configured=bool(
            self.semantic_client is not None
            and self.semantic_client.api_key
            and self.semantic_client.knowledge_base
        )
        background=(self._background_mining.status()
                    if self._background_mining is not None else {
                        "state":"initializing","running":False,
                        "last_error":None,
                    })
        background.update({
            "build_mode":BACKGROUND_MINING_BUILD_MODE,
            "event_sequence":self._event_sequence,
            "model_event_sequence":self._last_mined_event_sequence,
            "pending_events":self.pending_events,
            "active_rule_version":self.version,
            "serving_only":getattr(self,"serving_only",False),
        })
        last_mining=self.last_mining or {}
        last_pair=last_mining.get("pairwise",{}) or {}
        support_updates=[
            *last_mining.get("fpminer_incremental",[]),
            *last_pair.get("fpminer_incremental",[]),
        ]
        mining_workspace={
            "mode":MINING_WORKSPACE_MODE,
            "active_plans":len(self._active_mining_workspace_plans),
            "last_sync":list(last_mining.get("workspace_sync",[])),
            "last_support_updates":support_updates,
            "last_prune":dict(last_mining.get(
                "workspace_prune",{"removed":[],"error":None,"deferred":False}
            )),
            "full_structure_research":bool(
                last_mining.get("full_structure_research",True)
                or last_pair.get("full_structure_research",True)
            ),
            "full_population_ctv_estimation":True,
        }
        return {"instance_id":self.instance_id,
                "users":self.users(),"rules":self.mined_rules,"pair_rules":self.pair_rules,
                "runs":self.runs,"config":self.config,
                "version":self.version,"pending_events":self.pending_events,"last_mined_at":self.last_mined_at,
                "last_mining":self.last_mining,"last_pair_mining":self.last_pair_mining,
                "background_mining":background,
                "mining_workspace":mining_workspace,
                "dataset":self.dataset_info(),"engine":{"status":"PeTTaChainer live",
                "symbolic_only":self.symbolic_only,
                "serving_only":getattr(self,"serving_only",False),
                "semantic_workspace":bool(self._semantic_workspace_model),
                "recency_workspace":bool(self._recency_workspace),
                "llm_workspace":bool(self._llm_workspace),
                "relational_workspace":bool(self._relational_workspace),
                "relational_evidence_mode":self.config.get(
                    "relational_evidence_mode","disabled"
                ),
                "llm_annotated_articles":len(self._llm_article_annotations),
                "worker_pid":self.engine.pid,
                "startup_mode":self._startup_mode,
                "serving_model_sha256":self._serving_model_sha256,
                "startup_prewarm":dict(self._startup_prewarm),
                "last_live_score_profile":dict(
                    self._last_live_score_profile
                ),
                "worker_recoveries":self.engine.recovery_count,
                "last_worker_timeout":self.engine.last_timeout,
                "serving_outer_deadline_seconds":self.config[
                    "serving_reasoner_timeout_seconds"
                ],
                "benchmark_outer_deadline_seconds":self.config[
                    "benchmark_reasoner_timeout_seconds"
                ],
                "candidate_contexts":len(self._loaded_candidates),
                "point_proof_channels":len(self.mined_rules),
                "point_channel_templates":len(self._point_channel_templates),
                "point_channel_cache_entries":len(
                    self._point_channel_proof_cache
                ),
                "point_channel_completeness":dict(
                    self._last_point_completeness
                ),
                "benchmark_clean":not self._online_events,
                "online_events":len(self._online_events),
                "feed_rank_cache":{
                    "entries":len(self.feed_cache),
                    "hits":getattr(self,"_feed_rank_cache_hits",0),
                    "misses":getattr(self,"_feed_rank_cache_misses",0),
                },
                "point_case_cache_entries":len(self._proof_cache),
                "pair_case_cache_entries":len(self._pair_proof_cache),
                "pair_proof_channels":len(self.pair_rules),
                "pair_channel_cache_entries":len(
                    self._pair_channel_proof_cache
                ),
                "point_reasoner_query_calls":self._point_query_calls,
                "pair_reasoner_query_calls":self._pair_query_calls,
                "relational_reasoner_query_calls":self._relational_query_calls,
                "relational_reasoner_query_roots":self._relational_query_roots,
                "relational_feature_cache_entries":len(
                    self._relational_feature_cache
                ),
                "live_relational_proof_records":len(
                    self._live_relational_proof_ledger
                ),
                "candidate_relational_provenance_entries":len(
                    self._candidate_relational_proof_refs
                ),
                },
                "semantic_parser":{
                    "status":("disabled by symbolic-only policy" if self.symbolic_only else
                              "PeTTaChainer HTTP + NL2PLN configured"
                              if semantic_configured else
                              "invalid configuration"
                              if self._semantic_configuration_error else
                              "not configured"),
                    "configured":semantic_configured,
                    "mode":"bounded on-demand preview",
                    "timeout_seconds":(self.semantic_client.timeout
                                       if self.semantic_client is not None else None),
                    "cache_entries":len(self._semantic_preview_cache),
                    "cache_bytes":self._semantic_preview_cache_bytes,
                    "cache_max_bytes":SEMANTIC_CACHE_MAX_BYTES,
                    "cache_ttl_seconds":SEMANTIC_CACHE_TTL_SECONDS,
                }}
    def _serving_reasoner_timeout(self):
        return float(self.config.get(
            "serving_reasoner_timeout_seconds",
            DEFAULT_SERVING_REASONER_TIMEOUT_SECONDS,
        ))

    def _benchmark_reasoner_timeout(self):
        return float(self.config.get(
            "benchmark_reasoner_timeout_seconds",
            DEFAULT_BENCHMARK_REASONER_TIMEOUT_SECONDS,
        ))

    def _ensure_candidate_specs(self,specs,*,timeout_sec=None):
        """Materialize only candidate facts consumed by the selected proof path.

        Weighted point scoring uses alpha-normalized rule-channel templates, so its
        ordinary mined proofs never read the per-user candidate atoms.  Live
        feedback is deliberately case-specific and still needs the grounded
        ``recent_negative_match`` fact.  Shared-target aggregation continues to
        materialize every candidate because its Engagement query consumes them.
        """
        started=time.perf_counter()
        timeout_sec=(self._serving_reasoner_timeout()
                     if timeout_sec is None else float(timeout_sec))
        facts=[]; missing=set()
        factorized=self.config["aggregation"]=="weighted"
        for aid,case,attrs,raw_attrs in specs:
            if (factorized
                    and raw_attrs.get(LIVE_NEGATIVE_FEATURE,"none")=="none"):
                continue
            if case in self._loaded_candidates or case in missing: continue
            missing.add(case)
            for predicate,value in attrs.items():
                facts.append(f'(: fact_{case}_{predicate} ({predicate.title()} {case} {json.dumps(value)}) (STV 1.0 1.0))')
        cache_limit=int(self.config.get(
            "max_proof_cache_entries",DEFAULT_MAX_PROOF_CACHE_ENTRIES
        ))
        if len(self._loaded_candidates)+len(missing)>cache_limit:
            raise RuntimeError(
                "point candidate-fact cache limit exceeded before PeTTa "
                "mutation; promote a fresh rule snapshot"
            )
        if facts: self.engine.add_atoms_no_check(facts,timeout_sec=timeout_sec)
        self._loaded_candidates.update(missing)
        self._last_point_materialization_profile={
            "factorized_channels":factorized,
            "candidate_cases_requested":len(specs),
            "candidate_cases_materialized":len(missing),
            "candidate_atoms_inserted":len(facts),
            "total_seconds":time.perf_counter()-started,
        }

    @staticmethod
    def point_channel_case(rule_id,channel,premises):
        """Return one alpha-normalized case for an isolated point channel."""
        signature=json.dumps({
            "rule_id":str(rule_id),"channel":str(channel),
            "premises":[list(item) for item in premises],
        },sort_keys=True,separators=(",",":"),ensure_ascii=False)
        digest=hashlib.blake2s(
            signature.encode("utf-8"),digest_size=20
        ).hexdigest()
        return f"point_channel_{digest}"

    def _point_channel_topology(self):
        """Validate the compiler-issued isolated point-channel contract."""
        sources=getattr(self,"_point_channel_sources",None)
        signature=(
            tuple(sources) if isinstance(sources,list) else None,
            tuple((
                rule.get("id"),rule.get("target"),
                tuple(tuple(item) for item in rule.get("premises",())),
                rule.get("strength"),rule.get("confidence"),
                rule.get("negative_strength"),rule.get("negative_confidence"),
                rule.get("point_proof_channel_id"),
                rule.get("point_variant_id"),rule.get("point_decision_id"),
                json.dumps(rule.get("point_proof_factorization",{}),
                           sort_keys=True,separators=(",",":")),
            ) for rule in self.mined_rules),
        )
        if signature==getattr(self,"_point_topology_signature",None):
            return self._point_topology_cache
        if (not isinstance(sources,list)
                or len(sources)!=2*len(self.mined_rules)
                or not all(isinstance(source,str) for source in sources)
                or len(sources)!=len(set(sources))):
            raise RuntimeError(
                "compiled isolated point-channel sources are unavailable"
            )
        source_set=set(sources); consumed=set(); topology=[]; roots=set()
        statement_ids=set()
        for rule in self.mined_rules:
            rule_id=rule.get("id"); target=rule.get("target")
            channel=rule.get("point_proof_channel_id")
            variant_id=rule.get("point_variant_id")
            decision_id=rule.get("point_decision_id")
            premises=tuple(tuple(item) for item in rule.get("premises",()))
            if (target!="click" or not premises
                    or any(len(item)!=2 or not isinstance(item[0],str)
                           or not re.fullmatch(r"[a-z][a-z0-9_]*",item[0])
                           or not isinstance(item[1],str)
                           for item in premises)
                    or len({predicate for predicate,_value in premises})
                       !=len(premises)
                    or not all(isinstance(value,str)
                               and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",value)
                               for value in (rule_id,channel,variant_id,decision_id))
                    or len({rule_id,variant_id,decision_id})!=3
                    or any(identifier in statement_ids for identifier in (
                        rule_id,variant_id,decision_id
                    ))
                    or (rule_id,channel) in roots):
                raise RuntimeError(
                    f"point rule {rule_id!r} is not safely factorable"
                )
            roots.add((rule_id,channel))
            statement_ids.update((rule_id,variant_id,decision_id))
            terms=[f'({predicate.title()} $case {json.dumps(value)})'
                   for predicate,value in premises]
            premise=terms[0] if len(terms)==1 else f"(And {' '.join(terms)})"
            ctv=(f'(CTV (STV {rule["strength"]} {rule["confidence"]}) '
                 f'(STV {rule["negative_strength"]} '
                 f'{rule["negative_confidence"]}))')
            encoded_rule=json.dumps(rule_id); encoded_channel=json.dumps(channel)
            mined_signal=(
                f'(MinedPointPreference $case {encoded_rule} '
                f'{encoded_channel})'
            )
            point_signal=(
                f'(PointSignal $case {encoded_rule} {encoded_channel})'
            )
            variant_source=(
                f'(: {variant_id} (Implication {premise} {mined_signal}) '
                f'{ctv})'
            )
            decision_source=(
                f'(: {decision_id} (Implication {mined_signal} '
                f'{point_signal}) '
                '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))'
            )
            contract=rule.get("point_proof_factorization",{})
            if (variant_source not in source_set
                    or decision_source not in source_set
                    or contract.get("schema")
                       !="isolated_extensional_point_channel_v1"
                    or contract.get("case_variable")!="$case"
                    or contract.get("premise_tv")!=[1.0,1.0]
                    or contract.get("single_channel_producer") is not True
                    or contract.get("rule_id")!=rule_id
                    or contract.get("variant_id")!=variant_id
                    or contract.get("decision_id")!=decision_id
                    or contract.get("proof_channel_id")!=channel
                    or contract.get("premises")!=[list(item) for item in premises]
                    or contract.get("rule_source_sha256")!=hashlib.sha256(
                        variant_source.encode("utf-8")
                    ).hexdigest()
                    or contract.get("decision_source_sha256")!=hashlib.sha256(
                        decision_source.encode("utf-8")
                    ).hexdigest()):
                raise RuntimeError(
                    f"point channel {(rule_id,channel)!r} failed its compiler contract"
                )
            consumed.update((variant_source,decision_source))
            topology.append((rule_id,channel,premises,variant_id,decision_id))
        if consumed!=source_set:
            raise RuntimeError(
                "compiled isolated point-channel source set contains unknown rules"
            )
        topology=tuple(sorted(topology,key=lambda item:(item[0],item[1])))
        self._point_topology_signature=signature
        self._point_topology_cache=topology
        return topology

    def _weighted_point_proofs(self,specs,*,batch,cache_limit,timeout_sec):
        """Return every active PeTTa-proven isolated point channel."""
        topology=self._point_channel_topology()
        proof_roots=tuple((rule_id,channel,premises)
                          for rule_id,channel,premises,_variant,_decision
                          in topology)
        inference_key=(self.version,"isolated-point",int(
            self.config["chain_steps"]),batch,proof_roots)
        unique={}
        for _aid,case,attrs,_raw_attrs in specs:
            previous=unique.setdefault(case,attrs)
            if previous!=attrs:
                raise RuntimeError(
                    "one normalized candidate case has conflicting attributes"
                )
        missing=[case for case in unique
                 if (*inference_key,case) not in self._proof_cache]
        if len(self._proof_cache)+len(missing)>cache_limit:
            raise RuntimeError(
                "point proof cache limit exceeded; promote a fresh rule snapshot"
            )
        stats={
            "case_root_requests":len(unique),
            "case_root_hits":len(unique)-len(missing),
            "case_root_misses":len(missing),
            "channel_root_requests":0,"channel_root_hits":0,
            "channel_root_misses":0,
            "feedback_root_requests":0,"feedback_root_hits":0,
            "feedback_root_misses":0,
            "mode":"alpha_normalized_isolated_channels",
        }
        active_by_case={case:[] for case in unique}
        for case in unique:
            attrs=unique[case]
            for item in topology:
                rule_id,channel,premises,_variant,_decision=item
                if all(attrs.get(predicate)==value
                       for predicate,value in premises):
                    active_by_case[case].append(item)
        requested_active_uses=sum(map(len,active_by_case.values()))
        active_uses=sum(len(active_by_case[case]) for case in missing)
        active_templates=list(dict.fromkeys(
            item for case in missing for item in active_by_case[case]
        ))
        channel_key=lambda item:(
            self.version,"isolated-point-channel",
            int(self.config["chain_steps"]),batch,item[0],item[1],item[2]
        )
        channel_cache=self._point_channel_proof_cache
        query_templates=[item for item in active_templates
                         if channel_key(item) not in channel_cache]
        if len(channel_cache)+len(query_templates)>cache_limit:
            raise RuntimeError(
                "point proof-channel cache limit exceeded; promote a fresh "
                "rule snapshot"
            )
        stats.update(
            channel_root_requests=len(active_templates),
            channel_root_hits=len(active_templates)-len(query_templates),
            channel_root_misses=len(query_templates),
        )
        known_template_identities={
            item["case"]:(
                item["rule_id"],item["proof_channel_id"],
                tuple(tuple(fact) for fact in item["facts"]),
            )
            for item in self._point_channel_templates.values()
        }
        facts=[]; pending_loaded=set(); pending_audit={}
        for rule_id,channel,premises,variant_id,decision_id in query_templates:
            template=self.point_channel_case(rule_id,channel,premises)
            loaded_key=(self.version,rule_id,channel,premises)
            identity=(rule_id,channel,premises)
            previous=known_template_identities.setdefault(template,identity)
            if previous!=identity:
                raise RuntimeError(
                    f"point proof template hash collision at {template}"
                )
            pending_audit[loaded_key]={
                "case":template,"rule_id":rule_id,
                "proof_channel_id":channel,
                "variant_id":variant_id,"decision_id":decision_id,
                "facts":[list(item) for item in premises],
            }
            if loaded_key in self._loaded_point_channels:
                continue
            for index,(predicate,value) in enumerate(premises,1):
                facts.append(
                    f'(: fact_{template}_{index}_{predicate} '
                    f'({predicate.title()} {template} {json.dumps(value)}) '
                    '(STV 1.0 1.0))'
                )
            pending_loaded.add(loaded_key)
        for offset in range(0,len(facts),1000):
            self.engine.add_atoms_no_check(
                facts[offset:offset+1000],timeout_sec=timeout_sec
            )
        staged_results={}; calls=0
        for offset in range(0,len(query_templates),batch):
            items=query_templates[offset:offset+batch]
            queries=[
                f'(: $proof (PointSignal '
                f'{self.point_channel_case(rule_id,channel,premises)} '
                f'{json.dumps(rule_id)} {json.dumps(channel)}) $tv)'
                for rule_id,channel,premises,_variant,_decision in items
            ]
            # Every root has one producer and is queried only after the host's
            # exact categorical join says its certain premises are present.
            # Failure to prove any such root is an explicit completeness error,
            # never a silently truncated weighted rule set.
            steps=max(12,int(self.config["chain_steps"]))*len(items)
            results=self.engine.query_many(
                queries,steps=steps,timeout_sec=timeout_sec
            ); calls+=1
            for item,proofs in zip(items,results):
                staged_results[channel_key(item)]=proofs
        candidate_cache={**channel_cache,**staged_results}
        unproved=[(item[0],item[1]) for item in active_templates
                  if not candidate_cache.get(channel_key(item))]
        if unproved:
            raise RuntimeError(
                "active isolated point channels returned no PeTTa proof: "
                +", ".join(f"{rule_id}/{channel}"
                            for rule_id,channel in unproved[:8])
            )
        channel_cache.update(staged_results)
        self._loaded_point_channels.update(pending_loaded)
        self._point_channel_templates.update(pending_audit)
        for case in missing:
            active=active_by_case[case]
            proofs=[]
            for item in active:
                proofs.extend(channel_cache[channel_key(item)])
            self._proof_cache[(*inference_key,case)]=proofs
        self._point_query_calls+=calls
        self._point_query_roots+=len(query_templates)
        self._point_pruned_query_roots+=(
            len(missing)*len(topology)-active_uses
        )
        self._point_channel_activations+=active_uses
        self._point_reused_channel_activations+=max(
            0,active_uses-len(query_templates)
        )
        self._last_point_completeness={
            "mode":"fail_closed_active_channel_completeness",
            "requested_candidate_cases":len(unique),
            "cached_candidate_cases":len(unique)-len(missing),
            "candidate_case_misses":len(missing),
            "compiled_channels":len(topology),
            "expected_active_channel_uses":requested_active_uses,
            "proven_active_channel_uses":requested_active_uses,
            "missing_active_channel_uses":0,
            "unique_active_templates":len(active_templates),
            "new_reasoner_roots":len(query_templates),
            "complete":True,
        }
        return unique,inference_key,stats,calls

    def _proofs_for_specs(self,specs,*,timeout_sec=None):
        timeout_sec=(self._serving_reasoner_timeout()
                     if timeout_sec is None else float(timeout_sec))
        batch=max(1,int(self.config["query_batch_size"])); calls=0
        cache_limit=int(self.config.get(
            "max_proof_cache_entries",DEFAULT_MAX_PROOF_CACHE_ENTRIES
        ))
        if self.config["aggregation"]=="weighted":
            unique,inference_key,point_stats,calls=(
                self._weighted_point_proofs(
                    specs,batch=batch,cache_limit=cache_limit,
                    timeout_sec=timeout_sec,
                )
            )
        else:
            unique={case:case for _aid,case,_attrs,_raw_attrs in specs}
            inference_key=(self.version,"shared-engagement",
                           int(self.config["chain_steps"]),batch)
            missing=[case for case in unique
                     if (*inference_key,case) not in self._proof_cache]
            if len(self._proof_cache)+len(missing)>cache_limit:
                raise RuntimeError(
                    "point proof cache limit exceeded; promote a fresh rule snapshot"
                )
            point_stats={
                "case_root_requests":len(unique),
                "case_root_hits":len(unique)-len(missing),
                "case_root_misses":len(missing),
                "channel_root_requests":0,"channel_root_hits":0,
                "channel_root_misses":0,
                "feedback_root_requests":0,
                "feedback_root_hits":0,
                "feedback_root_misses":0,
                "mode":"shared_engagement_revision",
            }
            self._last_point_completeness={
                "mode":"bounded_shared_target_not_channel_complete",
                "complete":None,
            }
            for offset in range(0,len(missing),batch):
                cases=missing[offset:offset+batch]
                queries=[f'(: $proof (Engagement {case} "click") $tv)'
                         for case in cases]
                # Max/hybrid deliberately retain PeTTa's canonical bounded
                # shared-target revision behavior.
                steps=max(1,int(self.config["chain_steps"]))*len(cases)
                results=self.engine.query_many(
                    queries,steps=steps,timeout_sec=timeout_sec
                ); calls+=1
                for case,proofs in zip(cases,results):
                    self._proof_cache[(*inference_key,case)]=proofs
        # Live feedback has its own conclusion predicate and query roots.  If
        # it shared the busy Engagement target, a finite backward-search budget
        # could be consumed by dozens of mined roots before the three online
        # policy rules were visited.  This explicit PeTTaChainer query makes
        # feedback availability independent of mined-rule count; the returned
        # proof is appended to the normal proof group and is still the sole
        # source of the feedback score and provenance in ``_rank``.
        feedback_cases=list(dict.fromkeys(
            case for _aid,case,_attrs,raw_attrs in specs
            if raw_attrs.get(LIVE_NEGATIVE_FEATURE,"none")!="none"
        ))
        feedback_key=("live-feedback",*inference_key)
        feedback_missing=[
            case for case in feedback_cases
            if (*feedback_key,case) not in self._proof_cache
        ]
        # Ordinary misses have already been published above. Count only the
        # additional feedback roots here; adding ``missing`` again would make
        # the configured bound fail at roughly half capacity.
        if len(self._proof_cache)+len(feedback_missing)>cache_limit:
            raise RuntimeError(
                "point proof cache limit exceeded by live-feedback roots; "
                "promote a fresh rule snapshot"
            )
        point_stats.update(
            feedback_root_requests=len(feedback_cases),
            feedback_root_hits=len(feedback_cases)-len(feedback_missing),
            feedback_root_misses=len(feedback_missing),
        )
        for offset in range(0,len(feedback_missing),batch):
            cases=feedback_missing[offset:offset+batch]
            queries=[
                f'(: $proof ({LIVE_NEGATIVE_CONCLUSION} {case}) $tv)'
                for case in cases
            ]
            # One grounded fact plus one implication needs four scheduler
            # steps in the current PeTTaChainer runtime.  This fixed policy
            # depth is independent of the configurable mined-rule budget.
            steps=max(LIVE_NEGATIVE_CHAIN_STEPS,
                      int(self.config["chain_steps"]))*len(cases)
            results=self.engine.query_many(
                queries,steps=steps,timeout_sec=timeout_sec
            ); calls+=1
            for case,proofs in zip(cases,results):
                self._proof_cache[(*feedback_key,case)]=proofs
        groups=[]
        for _aid,case,_attrs,raw_attrs in specs:
            proofs=list(self._proof_cache[(*inference_key,case)])
            if raw_attrs.get(LIVE_NEGATIVE_FEATURE,"none")!="none":
                proofs.extend(self._proof_cache[(*feedback_key,case)])
            groups.append(proofs)
        point_stats["hits"]=(point_stats["case_root_hits"]
                             +point_stats["channel_root_hits"]
                             +point_stats["feedback_root_hits"])
        point_stats["misses"]=(point_stats["case_root_misses"]
                               +point_stats["channel_root_misses"]
                               +point_stats["feedback_root_misses"])
        point_stats["requests"]=point_stats["hits"]+point_stats["misses"]
        self._last_point_cache_stats=point_stats
        return groups,calls

    def event(self,user,article,action,context=None,impression=None,*,
              feed_session=None,queue_revision=None,feed_position=None):
        if user not in self.data["users"] or str(article) not in self._articles: raise ValueError("unknown user or article")
        if action not in {"click","skip","like","complete"}: raise ValueError("invalid action")
        protocol_values=(feed_session,queue_revision,feed_position)
        protocol_supplied=any(value is not None for value in protocol_values)
        if protocol_supplied and not all(value is not None for value in protocol_values):
            raise ValueError(
                "live feedback requires session, queue_revision, and feed_position"
            )
        live_session_id=None; live_session=None
        delivery_recovery=None
        if impression and str(impression).startswith("live_"):
            match=re.fullmatch(r"live_([0-9a-f]{32})_\d+_\d+",str(impression))
            live_session_id=match.group(1) if match else None
            if protocol_supplied and str(feed_session)!=str(live_session_id):
                raise ValueError("feedback session does not match served impression")
            live_session=(self._feed_session_for_user(live_session_id,user)
                          if live_session_id else None)
            feedback_key=(str(impression),str(article))
            expected=(live_session.get("feedback_contexts",{}).get(feedback_key)
                      if live_session else None)
            if expected is None: raise ValueError("feedback does not match an active served feed item")
            if protocol_supplied:
                accepted_end=live_session.get("feedback_positions",{}).get(
                    feedback_key
                )
                if (type(feed_position) is not int
                        or accepted_end is None or accepted_end>feed_position):
                    raise ValueError("feedback item was not accepted by the client")
                delivery_recovery=self._recover_unaccepted_feed_deliveries(
                    live_session,accepted_position=feed_position,
                    accepted_revision=queue_revision,
                )
            context=expected
        elif protocol_supplied:
            raise ValueError("feed acceptance fields require a live impression")
        elif isinstance(context,dict) and any(
                key in context for key in (
                    *RELATIONAL_PROOF_FIELDS,
                    *RELATIONAL_PROOF_FIELDS.values(),
                )):
            # Relational provenance is server-issued state. A direct caller
            # may submit ordinary context observations, but cannot inject an
            # old or otherwise valid proof into the online mining stream.
            raise ValueError(
                "relational context requires a verified live feed item"
            )
        relational_references=(
            self._validate_relational_context_provenance(
                context,user=user,article=article
            ) if isinstance(context,dict) else {}
        )
        attrs=self.contextual_features(user,self.article(article),context if isinstance(context,dict) else None)
        event={"user":user,"article":str(article),"action":action,**attrs}
        recorded_context=dict(attrs)
        for proof_field,references in relational_references.items():
            event[proof_field]=list(references)
            recorded_context[proof_field]=list(references)
        if impression: event["impression"]=str(impression)[:200]
        self.data["events"].append(event)
        self._online_events.append(event)
        # Editorial tie priors are part of the active model snapshot. Keep them
        # frozen until a complete mining build is atomically promoted;
        # otherwise one online event could change every user's order outside
        # PeTTaChainer. The live user's symbolic profile still changes below.
        self.feed_cache.clear()
        profile=self._mutable_user_profile(user)
        negative_history=profile.setdefault("negative_history",[])
        if action in POSITIVE:
            self._popularity[str(article)]+=1
            # A later positive outcome overrides an older exact skip. Other
            # skipped items in the same taxonomy remain independent evidence.
            negative_history[:]=[
                aid for aid in negative_history if str(aid)!=str(article)
            ]
            topics=profile.setdefault("topics",[])
            topic=self.article(article).get("topic",self.article(article).get("category"))
            if isinstance(topics,list) and topic and topic not in topics: topics.append(topic)
            recent_subcategories=profile.setdefault("recent_subcategories",[])
            subcategory=self.article(article).get("subcategory")
            if subcategory:
                recent_subcategories.append(str(subcategory))
                del recent_subcategories[:-5]
            history=profile.setdefault("history",[])
            history.append(str(article)); del history[:-200]
        else:
            # Move a repeated skip to the newest position instead of treating
            # browser retries as independent preference evidence. HTTP-level
            # idempotency belongs in the production event ledger, but this
            # bounded profile must not amplify an identical article locally.
            negative_history[:]=[
                aid for aid in negative_history if str(aid)!=str(article)
            ]
            negative_history.append(str(article))
            del negative_history[:-LIVE_NEGATIVE_HISTORY_LIMIT]
        queue_update=None
        if live_session is not None:
            feedback_key=(str(impression),str(article))
            live_session["feedback_contexts"].pop(feedback_key,None)
            live_session.get("feedback_positions",{}).pop(feedback_key,None)
            if not protocol_supplied:
                # Legacy in-process callers have no abort boundary; preserve
                # their historical behaviour by treating deliveries as seen.
                live_session["deliveries"]=[]
            queue_update=self._rerank_unserved_queue(
                live_session_id,live_session,action=action,article=article
            )
            if delivery_recovery is not None:
                queue_update["delivery_recovery"]=delivery_recovery
        self._event_sequence+=1; self.pending_events+=1
        mining_scheduled=False
        if (not getattr(self,"serving_only",False)
                and self.pending_events>=self.config["mine_interval"]):
            mining_scheduled=self._background_mining.schedule()
        mining_status=self._background_mining.status()
        mining_status["serving_only"]=getattr(self,"serving_only",False)
        return {"pending_events":self.pending_events,"mined":None,
                "mining_scheduled":mining_scheduled,
                "background_mining":mining_status,
                "recorded_context":recorded_context,
                "profile_changed":True,"queue_revision":queue_update,
                "negative_profile":{
                    "articles":list(negative_history),
                    "limit":LIVE_NEGATIVE_HISTORY_LIMIT,
                    "generalization_window":LIVE_NEGATIVE_GENERALIZATION_WINDOW,
                    "provenance":"bounded_online_feedback_policy_not_mined",
                }}

    def configure(self,values):
        minimums={"min_support":1,"max_rules":1,"top_k":1,"conjunctions":2,"chain_steps":1,
                  "mine_interval":1,"max_candidates":0,"random_seed":0,"query_batch_size":1,
                  "feed_window":1,"negative_ratio":0,"max_feature_values":2,
                  "pair_min_support":1,"pair_max_rules":1,"pair_negative_ratio":0,
                  "pair_conjunctions":2,"pair_numeric_bins":1,
                  "pairwise_opponents":0,"pair_chain_steps":2,
                  "max_pair_comparisons":1,
                  "max_proof_cache_entries":1,
                  "max_total_pair_comparisons":1,
                  "mining_retention_max_units":1,
                  "mining_retention_max_cases":1}
        updates={key:int(value) for key,value in values.items() if key in minimums}
        for key,value in updates.items():
            if value<minimums[key]: raise ValueError(f"{key} must be >= {minimums[key]}")
        if updates.get("conjunctions",self.config["conjunctions"])>4:
            raise ValueError("conjunctions must be <= 4 for the bounded rich feature miner")
        if updates.get("pair_conjunctions",self.config["pair_conjunctions"])>4:
            raise ValueError("pair_conjunctions must be <= 4")
        if updates.get("pair_numeric_bins",self.config["pair_numeric_bins"])>8:
            raise ValueError("pair_numeric_bins must be <= 8")
        # The independently checked shallow topology is deliberately bounded
        # to 1,024 compiled point rules and 1,024 pair channels. Reject an
        # impossible configuration at the API boundary instead of allowing an
        # unbounded rule snapshot or letting a post-ranking diagnostic abort
        # only after the expensive proof pass.
        if updates.get("max_rules",self.config["max_rules"])>1024:
            raise ValueError("max_rules must be <= 1024")
        if updates.get("pair_max_rules",self.config["pair_max_rules"])>1024:
            raise ValueError("pair_max_rules must be <= 1024")
        if updates.get(
            "max_proof_cache_entries",self.config["max_proof_cache_entries"]
        )>HARD_MAX_PROOF_CACHE_ENTRIES:
            raise ValueError(
                f"max_proof_cache_entries must be <= "
                f"{HARD_MAX_PROOF_CACHE_ENTRIES}"
            )
        if updates.get(
                "max_pair_comparisons",
                self.config["max_pair_comparisons"],
        )>HARD_MAX_PAIR_COMPARISONS:
            raise ValueError(
                f"max_pair_comparisons must be <= {HARD_MAX_PAIR_COMPARISONS}"
            )
        if updates.get(
                "max_total_pair_comparisons",
                self.config["max_total_pair_comparisons"],
        )>HARD_MAX_TOTAL_PAIR_COMPARISONS:
            raise ValueError(
                "max_total_pair_comparisons must be <= "
                f"{HARD_MAX_TOTAL_PAIR_COMPARISONS}"
            )
        if updates.get(
                "mining_retention_max_units",
                self.config["mining_retention_max_units"],
        )>HARD_RETENTION_MAX_UNITS:
            raise ValueError(
                "mining_retention_max_units must be <= "
                f"{HARD_RETENTION_MAX_UNITS}"
            )
        if updates.get(
                "mining_retention_max_cases",
                self.config["mining_retention_max_cases"],
        )>HARD_RETENTION_MAX_CASES:
            raise ValueError(
                "mining_retention_max_cases must be <= "
                f"{HARD_RETENTION_MAX_CASES}"
            )
        string_updates={}
        if "miner_strategy" in values:
            miner_strategy=str(values["miner_strategy"])
            if miner_strategy not in {
                    "fixed_combinations","target_aware","conditional_llm",
                    "conditional_llm_seed_only",
                    "petta_conditional_seed_only","petta_mdl_seed_only",
                    "petta_hierarchical_seed_only"}:
                raise ValueError(
                    "miner_strategy must be fixed_combinations, target_aware "
                    "conditional_llm, conditional_llm_seed_only or "
                    "petta_conditional_seed_only/petta_mdl_seed_only"
                    "/petta_hierarchical_seed_only"
                )
            string_updates["miner_strategy"]=miner_strategy
        effective_strategy=string_updates.get(
            "miner_strategy",self.config["miner_strategy"]
        )
        effective_point_depth=updates.get(
            "conjunctions",self.config["conjunctions"]
        )
        effective_pair_depth=updates.get(
            "pair_conjunctions",self.config["pair_conjunctions"]
        )
        if (effective_strategy=="target_aware"
                and max(effective_point_depth,effective_pair_depth)<3):
            raise ValueError(
                "target_aware requires conjunctions >= 3 or "
                "pair_conjunctions >= 3 so it can expand beyond fpMiner unaries"
            )
        if (effective_strategy in {
                "conditional_llm","conditional_llm_seed_only",
                "petta_conditional_seed_only","petta_mdl_seed_only",
                "petta_hierarchical_seed_only"}
                and effective_pair_depth<3):
            raise ValueError(
                "conditional LLM mining requires pair_conjunctions >= 3 so it can "
                "test a semantic seed with an invariant context gate"
            )
        if "aggregation" in values:
            aggregation=str(values["aggregation"])
            if aggregation not in {"max","weighted","hybrid"}: raise ValueError("aggregation must be max, weighted, or hybrid")
            string_updates["aggregation"]=aggregation
        if "rule_rank" in values:
            rule_rank=str(values["rule_rank"])
            if rule_rank not in {"support","quality"}: raise ValueError("rule_rank must be support or quality")
            string_updates["rule_rank"]=rule_rank
        if "feature_profile" in values:
            feature_profile=str(values["feature_profile"])
            if feature_profile not in FEATURE_PROFILES:
                raise ValueError(
                    "feature_profile must be one of: "
                    + ", ".join(sorted(FEATURE_PROFILES))
                )
            string_updates["feature_profile"]=feature_profile
        if "ranking_mode" in values:
            ranking_mode=str(values["ranking_mode"])
            if ranking_mode not in {"pointwise","pairwise"}:
                raise ValueError("ranking_mode must be pointwise or pairwise")
            string_updates["ranking_mode"]=ranking_mode
        if "pairwise_fusion" in values:
            pairwise_fusion=str(values["pairwise_fusion"])
            if pairwise_fusion not in {"rank","probability"}:
                raise ValueError("pairwise_fusion must be rank or probability")
            string_updates["pairwise_fusion"]=pairwise_fusion
        if "pair_aggregation" in values:
            pair_aggregation=str(values["pair_aggregation"])
            if pair_aggregation not in {"proof_margin","posterior"}:
                raise ValueError("pair_aggregation must be proof_margin or posterior")
            string_updates["pair_aggregation"]=pair_aggregation
        if "pair_family_fusion" in values:
            pair_family_fusion=str(values["pair_family_fusion"])
            if pair_family_fusion not in {
                    "flat_margin","balanced_rank","balanced_margin",
                    "symbolic_balanced"}:
                raise ValueError(
                    "pair_family_fusion must be flat_margin, balanced_rank, "
                    "balanced_margin or symbolic_balanced"
                )
            string_updates["pair_family_fusion"]=pair_family_fusion
        if "pair_ctv_mode" in values:
            pair_ctv_mode=str(values["pair_ctv_mode"])
            if pair_ctv_mode not in {
                    "raw_pairs","impression_macro",
                    "raw_strength_effective_confidence",
                    "conditional_effective_backoff"}:
                raise ValueError(
                    "pair_ctv_mode must be raw_pairs, impression_macro or "
                    "raw_strength_effective_confidence or "
                    "conditional_effective_backoff"
                )
            string_updates["pair_ctv_mode"]=pair_ctv_mode
        if "pair_dependency_mode" in values:
            pair_dependency_mode=str(values["pair_dependency_mode"])
            if pair_dependency_mode not in {"clustered","residual_hypergraph"}:
                raise ValueError(
                    "pair_dependency_mode must be clustered or residual_hypergraph"
                )
            string_updates["pair_dependency_mode"]=pair_dependency_mode
        if "pair_margin_transform" in values:
            pair_margin_transform=str(values["pair_margin_transform"])
            if pair_margin_transform not in {"linear","log_odds"}:
                raise ValueError("pair_margin_transform must be linear or log_odds")
            string_updates["pair_margin_transform"]=pair_margin_transform
        if "pair_feature_profile" in values:
            pair_feature_profile=str(values["pair_feature_profile"])
            if pair_feature_profile not in PAIR_FEATURE_PROFILES:
                raise ValueError(
                    "pair_feature_profile must be one of: "
                    + ", ".join(sorted(PAIR_FEATURE_PROFILES))
                )
            string_updates["pair_feature_profile"]=pair_feature_profile
        if "relational_evidence_mode" in values:
            relational_mode=str(values["relational_evidence_mode"])
            if relational_mode not in {"disabled","facts_only","chained"}:
                raise ValueError(
                    "relational_evidence_mode must be disabled, facts_only, "
                    "or chained"
                )
            string_updates["relational_evidence_mode"]=relational_mode
        float_updates={}
        if "pairwise_weight" in values:
            pairwise_weight=float(values["pairwise_weight"])
            if not 0.0<=pairwise_weight<=1.0:
                raise ValueError("pairwise_weight must be between 0 and 1")
            float_updates["pairwise_weight"]=pairwise_weight
        if "ctv_evidence_k" in values:
            try:
                ctv_evidence_k=float(values["ctv_evidence_k"])
            except (TypeError,ValueError) as exc:
                raise ValueError(
                    "ctv_evidence_k must be a finite number greater than 0 and at most 1e9"
                ) from exc
            if (not math.isfinite(ctv_evidence_k)
                    or not 0.0<ctv_evidence_k<=1e9):
                raise ValueError(
                    "ctv_evidence_k must be a finite number greater than 0 and at most 1e9"
                )
            float_updates["ctv_evidence_k"]=ctv_evidence_k
        if ("pair_rule_selection_k" in values
                or "pair_ctv_evidence_k" in values):
            selection_key=("pair_rule_selection_k"
                           if "pair_rule_selection_k" in values
                           else "pair_ctv_evidence_k")
            try:
                pair_rule_selection_k=float(values[selection_key])
            except (TypeError,ValueError) as exc:
                raise ValueError(
                    f"{selection_key} must be a finite number greater than 0 "
                    "and at most 1e9"
                ) from exc
            if (not math.isfinite(pair_rule_selection_k)
                    or not 0.0<pair_rule_selection_k<=1e9):
                raise ValueError(
                    f"{selection_key} must be a finite number greater than 0 "
                    "and at most 1e9"
                )
            if ("pair_rule_selection_k" in values
                    and "pair_ctv_evidence_k" in values):
                try:
                    legacy_value=float(values["pair_ctv_evidence_k"])
                except (TypeError,ValueError) as exc:
                    raise ValueError(
                        "pair_ctv_evidence_k must equal pair_rule_selection_k"
                    ) from exc
                if legacy_value!=pair_rule_selection_k:
                    raise ValueError(
                        "pair_ctv_evidence_k is a deprecated alias and must "
                        "equal pair_rule_selection_k when both are supplied"
                    )
            float_updates["pair_rule_selection_k"]=pair_rule_selection_k
            # Preserve round-tripping of old frozen configurations without
            # allowing this legacy name to control PeTTa confidence encoding.
            float_updates["pair_ctv_evidence_k"]=pair_rule_selection_k
        if "pair_min_effect" in values:
            pair_min_effect=float(values["pair_min_effect"])
            if not 0.0<=pair_min_effect<=1.0:
                raise ValueError("pair_min_effect must be between 0 and 1")
            float_updates["pair_min_effect"]=pair_min_effect
        if "pair_margin_power" in values:
            try:
                pair_margin_power=float(values["pair_margin_power"])
            except (TypeError,ValueError) as exc:
                raise ValueError(
                    "pair_margin_power must be a finite number between "
                    f"{PAIR_MARGIN_POWER_MIN} and {PAIR_MARGIN_POWER_MAX}"
                ) from exc
            if (not math.isfinite(pair_margin_power)
                    or not PAIR_MARGIN_POWER_MIN<=pair_margin_power<=PAIR_MARGIN_POWER_MAX):
                raise ValueError(
                    "pair_margin_power must be a finite number between "
                    f"{PAIR_MARGIN_POWER_MIN} and {PAIR_MARGIN_POWER_MAX}"
                )
            float_updates["pair_margin_power"]=pair_margin_power
        for key,label in (
            ("serving_reasoner_timeout_seconds","serving reasoner timeout"),
            ("benchmark_reasoner_timeout_seconds","benchmark reasoner timeout"),
        ):
            if key not in values:
                continue
            try:
                timeout=float(values[key])
            except (TypeError,ValueError) as exc:
                raise ValueError(
                    f"{label} must be a finite number greater than 0 and at "
                    f"most {REASONER_TIMEOUT_SECONDS_MAX:g}"
                ) from exc
            if (not math.isfinite(timeout) or timeout<=0.0
                    or timeout>REASONER_TIMEOUT_SECONDS_MAX):
                raise ValueError(
                    f"{label} must be a finite number greater than 0 and at "
                    f"most {REASONER_TIMEOUT_SECONDS_MAX:g}"
                )
            float_updates[key]=timeout
        changed_values={**updates,**string_updates,**float_updates}
        if getattr(self,"symbolic_only",False):
            point_profile=changed_values.get("feature_profile",self.config["feature_profile"])
            pair_profile=changed_values.get("pair_feature_profile",self.config["pair_feature_profile"])
            predicates=(*FEATURE_PROFILES[point_profile],*PAIR_FEATURE_PROFILES[pair_profile])
            if any("semantic" in p or "entity" in p or "llm_" in p for p in predicates):
                raise ValueError("symbolic-only mode rejects semantic/vector/entity feature profiles")
        selected_pair=string_updates.get("pair_feature_profile",self.config.get("pair_feature_profile",""))
        selected_pair_predicates=PAIR_FEATURE_PROFILES.get(selected_pair,())
        selected_point=string_updates.get(
            "feature_profile",self.config.get("feature_profile","")
        )
        selected_point_predicates=FEATURE_PROFILES.get(selected_point,())
        _validate_relational_score_dependencies(
            selected_point_predicates,selected_pair_predicates
        )
        relational_predicates={
            "rel_entity_continuity_scope","rel_concept_continuity_scope",
            "pair_rel_entity_continuity_scope",
            "pair_rel_concept_continuity_scope",
        }
        relational_selected=bool(
            relational_predicates.intersection(
                (*selected_point_predicates,*selected_pair_predicates)
            )
        )
        effective_relational_mode=string_updates.get(
            "relational_evidence_mode",
            self.config.get("relational_evidence_mode","disabled"),
        )
        if relational_selected and effective_relational_mode!="chained":
            raise ValueError(
                "relational feature profiles require "
                "relational_evidence_mode=chained"
            )
        if (effective_relational_mode in {"facts_only","chained"}
                and not getattr(self,"_relational_workspace",{})):
            raise ValueError(
                "relational evidence modes require a prepared relational workspace"
            )
        if (any(predicate in PAIR_CATEGORICAL_SIDE_PREDICATES
                for predicate in selected_pair_predicates)
                and effective_pair_depth<3):
            raise ValueError(
                "scoped categorical profiles require pair_conjunctions >= 3; "
                "side categories are never mined as unconditioned unary rules"
            )
        if (selected_pair.startswith("llm_")
                and any(predicate.endswith("_quantile")
                        for predicate in selected_pair_predicates)):
            effective_bins=updates.get(
                "pair_numeric_bins",self.config["pair_numeric_bins"]
            )
            effective_value_cap=updates.get(
                "max_feature_values",self.config["max_feature_values"]
            )
            minimum_value_cap=2*effective_bins+2
            if effective_value_cap<minimum_value_cap:
                raise ValueError(
                    "LLM quantile profiles require max_feature_values >= "
                    f"{minimum_value_cap} for mirror-closed left/right bins"
                )
        if selected_pair.startswith("llm_") and not getattr(self,"_llm_article_annotations",{}):
            raise ValueError("LLM profiles require a prepared annotation workspace")
        if (selected_pair.startswith("llm_")
                and changed_values.get(
                    "pair_aggregation",self.config["pair_aggregation"]
                )!="proof_margin"):
            raise ValueError(
                "LLM profiles require proof_margin so correlated variants use "
                "isolated PeTTa proof channels"
            )
        if (effective_strategy in {
                "conditional_llm","conditional_llm_seed_only",
                "petta_conditional_seed_only","petta_mdl_seed_only",
                "petta_hierarchical_seed_only"}
                and selected_pair not in {
                    "llm_conditional","llm_conditional_quantile",
                    "llm_conditional_quantile_relational",
                    "llm_conditional_magnitude_backoff",
                    "llm_conditional_scoped_taxonomy"
                }):
            raise ValueError(
                "conditional LLM mining requires pair_feature_profile "
                "llm_conditional, llm_conditional_quantile or "
                "llm_conditional_quantile_relational or "
                "llm_conditional_magnitude_backoff or "
                "llm_conditional_scoped_taxonomy"
            )
        effective_pair_ctv=string_updates.get(
            "pair_ctv_mode",self.config.get("pair_ctv_mode","raw_pairs")
        )
        if (effective_pair_ctv=="conditional_effective_backoff"
                and effective_strategy not in {
                    "conditional_llm_seed_only","petta_conditional_seed_only",
                    "petta_mdl_seed_only",
                    "petta_hierarchical_seed_only",
                }):
            raise ValueError(
                "conditional_effective_backoff requires "
                "a seed-only conditional miner strategy"
            )
        if (getattr(self,"_llm_workspace",{})
                and changed_values.get("pair_dependency_mode",self.config["pair_dependency_mode"])!="clustered"):
            raise ValueError("LLM workspace requires clustered dependencies for correlated text evidence")
        if selected_pair.startswith("workspace_") and not getattr(self,"_semantic_workspace_model",{}):
            raise ValueError("workspace profiles require a prepared semantic-data snapshot")
        if selected_pair.startswith("text_semantic_recency_") and not getattr(self,"_recency_workspace",{}):
            raise ValueError("recency profiles require a prepared recency workspace snapshot")
        if (getattr(self,"_recency_workspace",{})
                and changed_values.get("pair_dependency_mode",self.config["pair_dependency_mode"])!="clustered"):
            raise ValueError("recency workspace requires clustered dependencies for shared encoder evidence")
        if (getattr(self,"_semantic_workspace_model",{})
                and changed_values.get("pair_dependency_mode",self.config["pair_dependency_mode"])!="clustered"):
            raise ValueError("semantic workspace requires clustered dependencies to avoid counting encoder variants twice")
        changed=any(self.config.get(key)!=value for key,value in changed_values.items())
        inference_changed=any(key in {
                                  "aggregation","chain_steps",
                                  "pair_chain_steps","query_batch_size"
                              }
                              and self.config.get(key)!=value
                              for key,value in changed_values.items())
        pair_margin_changed=any(
            key in changed_values and self.config.get(key)!=changed_values[key]
            for key in ("pair_margin_transform","pair_margin_power")
        )
        self.config.update(updates); self.config.update(string_updates); self.config.update(float_updates)
        if changed:
            self._feed_sessions.clear(); self.feed_cache.clear()
        if inference_changed:
            self._proof_cache.clear(); self._pair_proof_cache.clear()
            self._point_channel_proof_cache.clear()
            self._pair_channel_proof_cache.clear(); self._pair_margin_cache.clear()
            self._pair_proof_origins.clear()
            self._point_query_calls=0; self._point_query_roots=0
            self._point_pruned_query_roots=0
            self._point_channel_activations=0
            self._point_reused_channel_activations=0
            self._last_point_completeness={}
            self._last_point_cache_stats={}
        elif pair_margin_changed:
            self._pair_margin_cache.clear()
        return self.config

# ``spawn`` imports this module in every scoring child. Never construct the
# default Lab there (or while running this file before CLI arguments are
# parsed), otherwise worker creation would recurse indefinitely. Imported test
# callers retain the historical ready-to-use fixture singleton.
LAB=(Lab() if (__name__!="__main__" and mp.current_process().name=="MainProcess"
              and os.getenv("RECOMMENDATION_DISABLE_DEFAULT_LAB")!="1")
     else None)
LAB_SWAP_LOCK=threading.Lock()
TRAINING_CONFIRMATION_HTTP_LOCK=threading.Lock()
TRAINING_CONFIRMATION_MAX_BODY_BYTES=32_768
HTML=WEB_TEMPLATE.read_text(encoding="utf-8")


def _loopback_name(value):
    name=str(value or "").lower()
    if name=="localhost": return True
    try: return ipaddress.ip_address(name).is_loopback
    except ValueError: return False


def _authority_parts(authority,default_port):
    try:
        parsed=urlparse("//"+str(authority or ""))
        if (not parsed.hostname or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment):
            return None
        return parsed.hostname.lower(),parsed.port or default_port
    except ValueError:
        return None


def _training_confirmation_request_error(headers,client_host,server_port):
    """Return an HTTP error for an unsafe mutation request, otherwise None."""
    if not _loopback_name(client_host):
        return 403,"training confirmation is restricted to loopback clients"
    host=_authority_parts(headers.get("Host"),server_port)
    if host is None or not _loopback_name(host[0]) or host[1]!=server_port:
        return 403,"training confirmation requires this localhost server origin"
    content_type=str(headers.get("Content-Type","")).split(";",1)[0].strip().lower()
    if content_type!="application/json":
        return 415,"training confirmation requires application/json"

    configured_token=os.getenv("RECOMMENDATION_ADMIN_TOKEN","").strip()
    if configured_token:
        supplied=str(headers.get("X-Admin-Token","")).strip()
        authorization=str(headers.get("Authorization","")).strip()
        if authorization.lower().startswith("bearer "):
            supplied=authorization[7:].strip()
        if not supplied or not hmac.compare_digest(supplied,configured_token):
            return 403,"training confirmation requires the configured admin token"

    origin=str(headers.get("Origin","")).strip()
    if origin:
        try:
            parsed=urlparse(origin)
            origin_host=(parsed.hostname.lower(),parsed.port or 80)
        except (AttributeError,ValueError):
            return 403,"training confirmation requires a same-origin request"
        if (parsed.scheme!="http" or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path not in {"","/"}
                or origin_host!=host):
            return 403,"training confirmation requires a same-origin request"
    return None

@contextmanager
def pinned_lab():
    """Pin the active Lab and acquire its lock without racing a dataset swap."""
    with LAB_SWAP_LOCK:
        lab=LAB
        if lab is None: raise RuntimeError("recommendation Lab is not initialized")
        lab.lock.acquire()
    try:
        yield lab
    finally:
        lab.lock.release()

class Handler(BaseHTTPRequestHandler):
    protocol_version="HTTP/1.1"

    def send_json(self,value,status=200):
        body=json.dumps(value).encode(); self.send_response(status); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(body))); self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError): pass
    def do_GET(self):
        try:
            parsed=urlparse(self.path); path=parsed.path; query=parse_qs(parsed.query)
            if path=="/health/live":
                self.send_json({"status":"ok"}); return
            if path=="/health/ready":
                with pinned_lab() as lab:
                    ready=(not lab._closed and lab.engine.pid is not None
                           and bool(lab.mined_rules))
                    payload={
                        "status":"ready" if ready else "not_ready",
                        "instance_id":lab.instance_id,
                        "rule_version":lab.version,
                        "worker_pid":lab.engine.pid,
                        "point_rules":len(lab.mined_rules),
                        "pair_rules":len(lab.pair_rules),
                    }
                self.send_json(payload,200 if ready else 503); return
            if path=="/":
                with pinned_lab() as lab:
                    first_user=lab.users()[0]["id"]; state=lab.state()
                    first_page=lab.feed_page(first_user,limit=lab.config["top_k"])
                bootstrap=(f"<script>window.__INITIAL_STATE__={script_json(state)};"
                           f"window.__INITIAL_FEED_PAGE__={script_json(first_page)};</script><script>")
                options="".join(f'<option value="{html_lib.escape(str(user["id"]),quote=True)}">'
                                f'{html_lib.escape(str(user["id"]))}</option>' for user in state["users"])
                rendered=(HTML.replace('<select id="user"></select>',f'<select id="user">{options}</select>',1)
                          .replace('<div id="feed"></div>',f'<div id="feed">{first_paint_feed(first_page)}</div>',1)
                          .replace('Using the built-in fixture.',f'{html_lib.escape(state["dataset"]["name"])} is loaded and mined.',1)
                          .replace("<script>",bootstrap,1))
                body=rendered.encode(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(body))); self.end_headers()
                try: self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError): pass
                return
            with pinned_lab() as lab:
                if path=="/api/state": self.send_json(lab.state()); return
                if path=="/api/feed":
                    user=(query.get("user") or [self.headers.get("X-User",lab.users()[0]["id"])])[0]
                    if {"cursor","session","limit","page_size"}.intersection(query):
                        limit=(query.get("limit") or query.get("page_size") or [None])[0]
                        self.send_json(lab.feed_page(user,cursor=(query.get("cursor") or [None])[0],
                                                     session=(query.get("session") or [None])[0],limit=limit))
                    else: self.send_json({"feed":lab.score(user)})
                    return
            self.send_json({"error":"not found"},404)
        except ValueError as exc: self.send_json({"error":str(exc)},400)
        except TimeoutError:
            # Recovery is best-effort and is recorded in engine.last_timeout;
            # never tell the caller it succeeded when startup itself failed.
            self.send_json({"error":"reasoner deadline exceeded; request failed"},504)
        except Exception as exc: self.send_json({"error":f"engine failure: {exc}"},500)
    def do_POST(self):
        global LAB
        path=urlparse(self.path).path
        if path=="/api/training-confirmation":
            request_error=_training_confirmation_request_error(
                self.headers,self.client_address[0],self.server.server_port
            )
            if request_error is not None:
                status,message=request_error
                self.send_json({"error":message},status); return
        try:
            raw_length=self.headers.get("Content-Length")
            if path=="/api/training-confirmation" and raw_length is None:
                self.send_json({"error":"Content-Length is required"},411); return
            content_length=int(raw_length or 0)
            if content_length<0: raise ValueError
        except (TypeError,ValueError):
            self.send_json({"error":"invalid Content-Length"},400); return
        if (path=="/api/training-confirmation"
                and content_length>TRAINING_CONFIRMATION_MAX_BODY_BYTES):
            self.send_json({"error":"training confirmation request body is too large"},413)
            return
        try: body=json.loads(self.rfile.read(content_length) or b"{}")
        except (json.JSONDecodeError,UnicodeDecodeError):
            self.send_json({"error":"invalid json"},400); return
        training_lock_claimed=False
        if path=="/api/training-confirmation":
            if not TRAINING_CONFIRMATION_HTTP_LOCK.acquire(blocking=False):
                self.send_json({"error":"training confirmation is already running"},409)
                return
            training_lock_claimed=True
        try:
            if path in {
                    "/api/config","/api/dataset/load","/api/mine",
                    "/api/training-confirmation","/api/tune",
            }:
                with pinned_lab() as active:
                    immutable=bool(getattr(active,"serving_only",False))
                if immutable:
                    self.send_json({
                        "error":(
                            "serving-only scorer is immutable; publish a new "
                            "validated frozen model"
                        )
                    },409)
                    return
            if path=="/api/semantic/preview":
                # Copy the source record while holding the lab lock, then
                # release it before the slower external model call so feed and
                # benchmark requests continue to use the embedded reasoner.
                with pinned_lab() as lab:
                    article=dict(lab.article(body["article"]))
                    semantic_lab=lab
                semantic_result=semantic_lab.preview_semantics(article)
                with LAB_SWAP_LOCK:
                    if LAB is not semantic_lab:
                        self.send_json({"error":"semantic preview belongs to a stale dataset"},409)
                        return
                self.send_json(semantic_result)
                return
            if path=="/api/dataset/load":
                with pinned_lab() as active:
                    symbolic_only=active.symbolic_only
                    semantic_workspace=bool(getattr(active,"_semantic_workspace_model",{})) and not symbolic_only
                    recency_workspace=bool(getattr(active,"_recency_workspace",{})) and not symbolic_only
                    replay_snapshot=bool(getattr(active,"_replay_snapshot",False))
                    # Transfer the selected architecture, not a silently reset
                    # baseline, when changing a strict-symbolic dataset.
                    transferred_config=active.config.copy() if symbolic_only or semantic_workspace or recency_workspace or replay_snapshot else None
                if symbolic_only:
                    data=load_symbolic_snapshot(body["path"])
                elif semantic_workspace:
                    data=load_semantic_snapshot(body["path"])
                elif recency_workspace:
                    data=load_symbolic_snapshot(body["path"])
                    if not data.get("metadata",{}).get("recency_workspace"):
                        raise ValueError("recency mode requires a prepared recency workspace snapshot")
                elif replay_snapshot:
                    data=load_symbolic_snapshot(body["path"])
                else:
                    data=load_mind(body["path"],max_train_cases=int(body.get("max_train_cases",20000)),
                                   max_eval_impressions=int(body.get("max_eval_impressions",500)),
                                   seed=int(body.get("random_seed",body.get("seed",7))),
                                   text_embedding_path=body.get("text_embedding_path"))
                replacement=Lab(
                    data=data,symbolic_only=symbolic_only,
                    config=transferred_config,
                    serving_only=active.serving_only,
                )
                with LAB_SWAP_LOCK:
                    previous=LAB; LAB=replacement
                # No new request can pin the old Lab after the swap. Waiting
                # for its lock lets an already-pinned request finish before its
                # private scorer process is retired.
                if previous is not None:
                    with previous.lock: previous.close()
                self.send_json({"message":"Dataset loaded and mined.","dataset":replacement.dataset_info(),
                                "state":replacement.state()})
                return
            with pinned_lab() as lab:
                if path=="/api/event": result=lab.event(
                    body["user"],body["article"],body["action"],
                    body.get("context"),body.get("impression"),
                    feed_session=body.get("session"),
                    queue_revision=body.get("queue_revision"),
                    feed_position=body.get("feed_position"),
                )
                elif path=="/api/mine": result=lab.mine()
                elif path=="/api/config":
                    previous=lab.config.copy(); configured=lab.configure(body)
                    should_mine=any(key in body and previous.get(key)!=configured.get(key)
                                    for key in MINING_CONFIG_KEYS)
                    if should_mine:
                        try: mined=lab.mine()
                        except BaseException:
                            lab.configure(previous)
                            raise
                    else: mined=None
                    result={"config":lab.config,"mined":mined}
                elif path=="/api/benchmark": result=lab.benchmark(body)
                elif path=="/api/compare": result=lab.compare_profiles(body)
                elif path=="/api/training-confirmation": result=lab.training_confirmation(body)
                elif path=="/api/tune": result=lab.tune(body)
                else: self.send_json({"error":"not found"},404); return
            self.send_json(result)
        except SemanticPreviewBusyError:
            self.send_json({"error":"semantic parser is busy; retry shortly"},503)
        except PeTTaChainerTimeoutError:
            self.send_json({"error":"semantic parser upstream timed out"},504)
        except PeTTaChainerConfigurationError:
            self.send_json({"error":"semantic parser is not configured"},503)
        except (PeTTaChainerUpstreamError,PeTTaChainerProtocolError):
            self.send_json({"error":"semantic parser upstream failed"},502)
        except PeTTaChainerInputError as exc:
            self.send_json({"error":str(exc)},400)
        except TimeoutError:
            self.send_json({"error":"reasoner deadline exceeded; request failed"},504)
        except (KeyError,ValueError,OSError) as exc: self.send_json({"error":str(exc)},400)
        except Exception as exc: self.send_json({"error":f"engine failure: {exc}"},500)
        finally:
            if training_lock_claimed:
                TRAINING_CONFIRMATION_HTTP_LOCK.release()
def load_symbolic_snapshot(path):
    path=Path(path)
    if not str(path).endswith((".json",".json.gz")):
        raise ValueError("symbolic-only mode requires a prepared .json or .json.gz causal snapshot")
    with (gzip.open(path,"rt") if path.suffix==".gz" else path.open()) as stream:
        data=json.load(stream)
    data.setdefault("metadata",{})["root"]=str(path.resolve())
    data["metadata"]["replay_snapshot"]=True
    return data


def load_semantic_snapshot(path):
    data=load_symbolic_snapshot(path)
    if not data.get("semantic_workspace_model") or not data.get("metadata",{}).get("semantic_workspace"):
        raise ValueError("semantic-data requires a prepared semantic_workspace snapshot")
    return data


def load_serving_model(path):
    path=Path(path)
    if path.suffix!=".json":
        raise ValueError("serving model must be a .json artifact")
    with path.open(encoding="utf-8") as stream:
        model=json.load(stream)
    if not isinstance(model,dict):
        raise ValueError("serving model must contain a JSON object")
    return model


def write_serving_model(path,model):
    """Create, fsync and atomically publish a model without overwriting one."""
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x",encoding="utf-8") as stream:
            json.dump(model,stream,ensure_ascii=False,sort_keys=True,
                      separators=(",",":"),allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary,target)
    finally:
        temporary.unlink(missing_ok=True)


class ProductionHTTPServer(ThreadingHTTPServer):
    daemon_threads=True
    allow_reuse_address=True
    request_queue_size=128


def _scorer_worker_command(args,port):
    command=[sys.executable,"-m","recommendation",
             "--host","127.0.0.1","--port",str(port),"--workers","1",
             "--max-train-cases",str(args.max_train_cases),
             "--max-eval-impressions",str(args.max_eval_impressions),
             "--seed",str(args.seed)]
    if args.workers>1 or args.serving_only:
        command.append("--serving-only")
    for option,value in (
        ("--symbolic-data",args.symbolic_data),
        ("--semantic-data",args.semantic_data),
        ("--replay-data",args.replay_data),
        ("--config-file",args.config_file),
        ("--serving-model",args.serving_model),
        ("--text-embeddings",args.text_embeddings),
    ):
        if value:
            command.extend((option,str(value)))
    if args.fixture:
        command.append("--fixture")
    elif (not args.symbolic_data and not args.semantic_data
          and not args.replay_data and args.mind):
        command.extend(("--mind",str(args.mind)))
    return tuple(command)


def main():
    global LAB
    local_archive=DATASET_DIR/"MIND_small_x1.zip"
    parser=argparse.ArgumentParser()
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=7070)
    parser.add_argument(
        "--workers",type=int,default=1,
        help="Isolated scorer processes behind a session-sticky gateway",
    )
    parser.add_argument(
        "--worker-start-port",type=int,
        help="First loopback scorer port (default: gateway port + 1)",
    )
    parser.add_argument("--worker-ready-timeout",type=float,default=180.0)
    parser.add_argument("--backend-timeout",type=float,default=120.0)
    parser.add_argument("--gateway-threads",type=int,default=64)
    parser.add_argument("--gateway-queue",type=int,default=256)
    parser.add_argument("--mind",default=str(local_archive) if local_archive.is_file() else None,
                        help="Extracted raw MIND root or RecZoo MIND_small_x1.zip")
    parser.add_argument("--fixture",action="store_true",help="Use only the bundled deterministic fixture")
    evidence=parser.add_mutually_exclusive_group()
    evidence.add_argument("--symbolic-data",help="Prepared causal JSON snapshot; enforces no NN, vectors or NL2PLN")
    evidence.add_argument("--semantic-data",help="Prepared frozen content-evidence workspace; only miner/PeTTa rules rank candidates")
    evidence.add_argument("--replay-data",help="Prepared JSON replay snapshot; preserve its existing content/entity evidence")
    parser.add_argument("--config-file",help="JSON configuration or saved symbolic_experiment artifact")
    parser.add_argument(
        "--serving-model",
        help="Validated frozen serving-model JSON; skips fpMiner at startup",
    )
    parser.add_argument(
        "--serving-only",action="store_true",
        help="Disable online mining and require frozen model publication",
    )
    parser.add_argument(
        "--export-serving-model",
        help="Write the mined/loaded serving model to a new JSON file",
    )
    parser.add_argument(
        "--export-only",action="store_true",
        help="Exit after --export-serving-model is written",
    )
    parser.add_argument(
        "--text-embeddings",
        help="Optional provenanced article text-embedding NPZ sidecar",
    )
    parser.add_argument("--max-train-cases",type=int,default=20000); parser.add_argument("--max-eval-impressions",type=int,default=500)
    parser.add_argument("--seed",type=int,default=7); args=parser.parse_args()
    if args.workers<1:
        parser.error("--workers must be positive")
    if not (1<=args.port<=65535):
        parser.error("--port must be between 1 and 65535")
    if min(args.worker_ready_timeout,args.backend_timeout)<=0:
        parser.error("pool timeouts must be positive")
    if min(args.gateway_threads,args.gateway_queue)<1:
        parser.error("gateway thread and queue limits must be positive")
    if args.export_only and not args.export_serving_model:
        parser.error("--export-only requires --export-serving-model")
    if args.config_file and args.serving_model:
        parser.error("--config-file cannot override --serving-model")
    if args.serving_only and not args.serving_model:
        parser.error("--serving-only requires --serving-model")
    if args.workers>1:
        if not args.serving_model:
            parser.error("--workers > 1 requires an immutable --serving-model")
        if args.export_serving_model or args.export_only:
            parser.error("model export must run before starting a scorer pool")
        start_port=(args.worker_start_port
                    if args.worker_start_port is not None else args.port+1)
        worker_ports=list(range(start_port,start_port+args.workers))
        if (start_port<1 or worker_ports[-1]>65535
                or args.port in worker_ports):
            parser.error("worker port range is invalid or overlaps gateway port")
        from .production_pool import serve_pool
        serve_pool(
            host=args.host,port=args.port,
            worker_commands=[_scorer_worker_command(args,port)
                             for port in worker_ports],
            worker_ports=worker_ports,
            ready_timeout=args.worker_ready_timeout,
            backend_timeout=args.backend_timeout,
            gateway_threads=args.gateway_threads,
            gateway_queue=args.gateway_queue,
        )
        return
    config={}
    if args.config_file:
        saved=json.loads(Path(args.config_file).read_text())
        config=saved.get("result",saved).get("config",saved)
    serving_model=(load_serving_model(args.serving_model)
                   if args.serving_model else None)
    if args.symbolic_data:
        LAB=Lab(load_symbolic_snapshot(args.symbolic_data),symbolic_only=True,
                config=config,serving_model=serving_model,
                serving_only=args.serving_only)
    elif args.semantic_data:
        LAB=Lab(load_semantic_snapshot(args.semantic_data),config=config,
                serving_model=serving_model,serving_only=args.serving_only)
    elif args.replay_data:
        LAB=Lab(load_symbolic_snapshot(args.replay_data),config=config,
                serving_model=serving_model,serving_only=args.serving_only)
    elif args.mind and not args.fixture:
        data=load_mind(args.mind,max_train_cases=args.max_train_cases,
                       max_eval_impressions=args.max_eval_impressions,seed=args.seed,
                       text_embedding_path=args.text_embeddings)
        LAB=Lab(data=data,config=config,serving_model=serving_model,
                serving_only=args.serving_only)
        print("Loaded:",json.dumps(LAB.dataset_info(),ensure_ascii=False))
    else:
        LAB=Lab(config=config,serving_model=serving_model,
                serving_only=args.serving_only)
    if args.export_serving_model:
        write_serving_model(args.export_serving_model,LAB.serving_model())
        print(f"Serving model: {args.export_serving_model}")
    if args.export_only:
        LAB.close(); LAB=None; return
    if args.serving_only:
        # The dataset, frozen model and compiled proof topology are immutable for
        # this process. Move that large object graph out of later cyclic-GC scans;
        # request/session objects remain normally reference-counted and collected.
        gc.collect()
        gc.freeze()
    print(f"Recommendation lab: http://{args.host}:{args.port}")
    server=ProductionHTTPServer((args.host,args.port),Handler)
    try: server.serve_forever()
    finally:
        server.server_close()
        if LAB is not None: LAB.close()
if __name__=="__main__": main()
