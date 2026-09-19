"""Live MeTTa-miner -> PeTTaChainer recommendation lab."""
from __future__ import annotations
import copy
import argparse, gzip, hashlib, hmac, html as html_lib, ipaddress, json, math, multiprocessing as mp, os, random, re, sys, threading, time, traceback, unicodedata, uuid
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..integrations.engine import (
    EXPECTED_NL2PLN_CONTRACT, RECOMMENDATION_PREDICATE_SCHEMA,
    PeTTaChainerClient, PeTTaChainerConfigurationError,
    PeTTaChainerInputError, PeTTaChainerProtocolError,
    PeTTaChainerTimeoutError, PeTTaChainerUpstreamError,
)
from ..adapters.mind import (
    history_feature_context,load_mind,prepare_history_feature_workspace,
    subcategory_transition_score,
)
from ..core.ctv_calibration import (
    DEFAULT_EVIDENCE_K,
    CTVObservation,
    calibrate_ctv,
    reencode_ctv_confidence,
)
from ..core.symbolic import QuantileNumericEvidence
from ..mining.target_miner import TargetMinerConfig, mine_target_patterns
from ..features.text_embeddings import load_text_embedding_sidecar, _canonical_corpus
from ..features.lexical_workspace import LEXICAL_FEATURES, build_lexical_workspace_facts
from ..features.semantic_workspace import SEMANTIC_WORKSPACE_FEATURES, build_semantic_workspace_facts
from ..features.recency_workspace import RECENCY_WORKSPACE_FEATURES, RECENCY_WORKSPACE_SCHEMA, build_recency_workspace_facts
from ..features.llm_workspace import LLM_NUMERIC_FEATURES, LLM_WORKSPACE_FEATURES, LLM_WORKSPACE_SCHEMA, build_llm_workspace_facts
from ..features.relational_workspace import (
    RELATIONAL_STRUCTURAL_RULES,
    RELATIONAL_WORKSPACE_SCHEMA,
    REL_CONCEPT_CONTINUITY_PROOF_IDS,
    REL_CONCEPT_CONTINUITY_SCOPE,
    REL_ENTITY_CONTINUITY_PROOF_IDS,
    REL_ENTITY_CONTINUITY_SCOPE,
    build_relational_plans,
    reduce_concept_relational_proofs,
    reduce_relational_proofs,
    validate_relational_projection,
)
from ..mining.conditional_llm_mining import (
    ConditionalMiningConfig, FpMinerUnary, mine_conditional_llm_patterns,
)
from ..mining.petta_workspace import PeTTaWorkspaceCache
from ..mining.incremental_fpminer import IncrementalFpMinerCache
from ..mining.retention import (
    DEFAULT_RETENTION_MAX_CASES,
    DEFAULT_RETENTION_MAX_UNITS,
    HARD_RETENTION_MAX_CASES,
    HARD_RETENTION_MAX_UNITS,
    retain_complete_units,
)
from ..paths import DATASET_DIR, MINER_DIR, WEB_TEMPLATE, WORKSPACE_ROOT
from .async_mining import AsyncMiningCoordinator, MiningSnapshot

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
    from pettachainer.pettachainer import PeTTaChainer
except ImportError as exc:
    raise RuntimeError("Run with PeTTaChainer/.venv; see recommendation/README.md") from exc

POSITIVE = {"click", "like", "complete"}
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
FEATURES = (
    "topic", "subcategory", "format", "affinity", "affinity_level",
    "recent_affinity", "long_affinity", "history_size_bucket",
    "entity_overlap", "entity_overlap_detail", "history_topic_count_bucket",
    "recent_topic_count_bucket", "topic_rank_bucket", "subcategory_affinity",
    "topic_recency_bucket", "subcategory_recency_bucket",
    "time_bucket", "ctr_bucket", "freshness_bucket", "position_bucket",
    "title_overlap_detail",
    # Bounded candidate-level conclusions produced by the relational
    # workspace. ``none`` is explicit negative knowledge only when both the
    # candidate and its causal history have complete source annotations;
    # incomplete coverage abstains by omitting the feature.
    "rel_entity_continuity_scope",
    "rel_concept_continuity_scope",
    "mi_topic_candidate_match_rank", "mi_subcategory_candidate_match_rank",
    "mi_entity_candidate_match_rank",
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
# PeTTa's in-process timeout is deliberately disabled inside the isolated
# scorer.  These finite parent-process deadlines are therefore the hard abort
# boundary: an expired worker is terminated and reconstructed from its last
# committed rule/fact journal before another request is accepted.
DEFAULT_REASONER_RPC_TIMEOUT_SECONDS = 30.0
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
# A scorer worker is reconstructed by replaying every acknowledged fact
# mutation since the last atomic rule-snapshot replacement. Bound that replay
# journal and the host proof-result caches explicitly: a long-lived process
# must promote a fresh snapshot instead of growing without limit.
DEFAULT_MAX_REASONER_JOURNAL_STATEMENTS = 250_000
DEFAULT_MAX_REASONER_JOURNAL_MUTATIONS = 50_000
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


PAIR_PROOF_PREPARATION_TIMERS = (
    "proof_topology_validation_seconds",
    "proof_activation_join_seconds",
    "proof_template_serialization_seconds",
    "proof_atomspace_insertion_seconds",
)


def _proof_channel_preparation_profile(reasoning_profile):
    """Project reasoner profiling to non-overlapping pre-query preparation."""
    preparation_seconds=math.fsum(
        float(reasoning_profile.get(key,0.0))
        for key in PAIR_PROOF_PREPARATION_TIMERS
    )
    return {
        **{key:(round(value,6) if isinstance(value,float) else value)
           for key,value in reasoning_profile.items()
           if key not in {"proof_query_seconds","total_seconds"}},
        "pre_query_component_sum_seconds":round(preparation_seconds,6),
        "proof_query_seconds_excluded":round(float(
            reasoning_profile.get("proof_query_seconds",0.0)
        ),6),
        "definition":(
            "Validates proof topology, joins active rules, serializes "
            "alpha-normalized templates and inserts only new template "
            "atoms. The separately reported proof-query time is not "
            "pair preparation."
        ),
    }
STV_RE = re.compile(r"\(STV\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\)")
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


class SemanticPreviewBusyError(RuntimeError):
    """The one bounded semantic conversion slot is already occupied."""


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


def balanced_forms(text, head="supportOf"):
    forms = []
    for start in (m.start() for m in re.finditer(r"\(" + head + r"\b", text)):
        depth = 0; quoted = False; escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if quoted:
                if escaped: escaped = False
                elif ch == "\\": escaped = True
                elif ch == '"': quoted = False
            elif ch == '"': quoted = True
            elif ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0: forms.append(text[start:i+1]); break
    return forms


def parse_rules(raw, limit=None, features=FEATURES):
    clause_re = re.compile(
        r'\((' + "|".join((*features, "engagement")) + r')\s+[^\s()]+\s+"([^"]+)"\)'
    )
    unique = {}
    for form in balanced_forms(" ".join(raw)):
        clauses = clause_re.findall(form)
        target = next((v for p,v in clauses if p == "engagement"), None)
        premises = tuple((p,v) for p,v in clauses if p in features)
        tvs = STV_RE.findall(form); support = re.search(r"\)\s+(\d+)\)$", form)
        if target in POSITIVE and premises and tvs and support:
            strength, confidence = map(float, tvs[0])
            negative_strength,negative_confidence=(
                map(float,tvs[1]) if len(tvs)>1 else (0.0,0.0)
            )
            unique[(premises,target)] = {
                "premises":premises,"target":target,
                "support":int(support.group(1)),
                "strength":strength,"confidence":confidence,
                "discovery_ctv":{
                    "positive":{"strength":strength,"confidence":confidence},
                    "negative":{
                        "strength":negative_strength,
                        "confidence":negative_confidence,
                    },
                    "complete":len(tvs)>1,
                },
            }
    rules = sorted(unique.values(), key=lambda r:(-r["support"],-r["strength"],r["premises"]))
    if limit is not None: rules=rules[:limit]
    for i, rule in enumerate(rules, 1): rule.update(id=f"mined_{i}", source="recommendation/miner/fpMiner.metta")
    return rules


def proof_tv(proof):
    values = STV_RE.findall(proof)
    return tuple(map(float, values[-1])) if values else (0.0,0.0)


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


def _scoring_worker(connection, statements, mutations=()):
    """Own one PeTTaChainer runtime in an isolated spawned process.

    PeTTa's compiled rule/index spaces are process-global even though each
    ``PeTTaChainer`` instance has a distinct KB name.  Keeping the live scorer
    behind this small RPC boundary prevents an old Lab/KB from consuming a
    later Lab's finite backward-search budget.
    """
    try:
        engine=PeTTaChainer()
        engine.set_backward_premise_prefilter(True)
        engine.add_atoms_no_check(list(statements))
        # Replaying only parent-acknowledged mutations reconstructs the exact
        # committed scorer state after an outer-deadline abort.  The command
        # that timed out is intentionally absent: whether it reached PeTTa is
        # unknowable, so retrying it implicitly would violate at-most-once RPC
        # semantics.
        for command,payload in mutations:
            if command=="add":
                engine.add_atoms_no_check(list(payload))
            elif command=="remove":
                engine.remove_statement(str(payload))
            else:
                raise ValueError(f"unknown scoring-worker bootstrap mutation: {command}")
        connection.send(("ok",None))
    except BaseException as exc:  # The parent must retain its previous worker.
        try: connection.send(("error",f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
        finally: connection.close()
        return
    try:
        while True:
            command,payload=connection.recv()
            try:
                if command=="add":
                    result=engine.add_atoms_no_check(payload)
                elif command=="remove":
                    result=engine.remove_statement(payload)
                elif command=="query_many":
                    queries,steps=payload
                    result=engine.query_many(queries,steps=steps,timeout_sec=0)
                elif command=="stop":
                    connection.send(("ok",None)); break
                else:
                    raise ValueError(f"unknown scoring-worker command: {command}")
                connection.send(("ok",result))
            except BaseException as exc:
                connection.send(("error",f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    except (EOFError,BrokenPipeError):
        pass
    finally:
        connection.close()


class IsolatedPeTTaChainer:
    """A replaceable PeTTaChainer whose global MeTTa state cannot leak.

    A remine starts a clean worker with the complete new rule snapshot and
    swaps it in only after compilation succeeds. Candidate facts are then
    loaded lazily into that snapshot. This is deliberately a process boundary:
    constructing a second PeTTaChainer in this process would still share the
    old global rule indexes.
    """
    def __init__(self, *, worker_target=None,
                 default_timeout_seconds=DEFAULT_REASONER_RPC_TIMEOUT_SECONDS,
                 startup_timeout_seconds=120.0,
                 max_journal_statements=DEFAULT_MAX_REASONER_JOURNAL_STATEMENTS,
                 max_journal_mutations=DEFAULT_MAX_REASONER_JOURNAL_MUTATIONS):
        self._context=mp.get_context("spawn")
        self._lock=threading.RLock()
        self._connection=None; self._process=None
        self._worker_target=worker_target or _scoring_worker
        self._default_timeout_seconds=self._finite_timeout(
            default_timeout_seconds,"default reasoner RPC timeout"
        )
        self._startup_timeout_seconds=self._finite_timeout(
            startup_timeout_seconds,"reasoner worker startup timeout"
        )
        self._max_journal_statements=self._positive_integer(
            max_journal_statements,"max reasoner journal statements"
        )
        self._max_journal_mutations=self._positive_integer(
            max_journal_mutations,"max reasoner journal mutations"
        )
        self._bootstrap_statements=()
        self._has_snapshot=False
        self._mutation_journal=[]
        self._journal_statements=0
        self._recovery_count=0
        self._last_timeout=None

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    @property
    def recovery_count(self):
        return self._recovery_count

    @property
    def last_timeout(self):
        return dict(self._last_timeout) if self._last_timeout is not None else None

    @property
    def journal_audit(self):
        return {
            "mutations":len(self._mutation_journal),
            "statements":self._journal_statements,
            "mutation_limit":self._max_journal_mutations,
            "statement_limit":self._max_journal_statements,
        }

    @staticmethod
    def _positive_integer(value,label):
        if isinstance(value,bool):
            raise ValueError(f"{label} must be a positive integer")
        try:
            parsed=int(value)
        except (TypeError,ValueError) as exc:
            raise ValueError(f"{label} must be a positive integer") from exc
        if parsed<=0 or parsed!=value:
            raise ValueError(f"{label} must be a positive integer")
        return parsed

    @staticmethod
    def _finite_timeout(value,label):
        try:
            timeout=float(value)
        except (TypeError,ValueError) as exc:
            raise ValueError(f"{label} must be a finite number greater than 0") from exc
        if not math.isfinite(timeout) or timeout<=0.0:
            raise ValueError(f"{label} must be a finite number greater than 0")
        return timeout

    @staticmethod
    def _shutdown(connection,process):
        if connection is not None:
            try:
                connection.send(("stop",None))
                if connection.poll(2): connection.recv()
            except (BrokenPipeError,EOFError,OSError):
                pass
            finally:
                try: connection.close()
                except OSError: pass
        if process is not None and process.pid is not None:
            process.join(5)
            if process.is_alive():
                process.terminate(); process.join(5)

    @staticmethod
    def _terminate(connection,process):
        """Abort a possibly wedged worker without waiting for its protocol."""
        if connection is not None:
            try: connection.close()
            except OSError: pass
        if process is not None and process.pid is not None:
            if process.is_alive(): process.terminate()
            process.join(5)
            if process.is_alive() and hasattr(process,"kill"):
                process.kill(); process.join(5)

    @staticmethod
    def _receive(connection,process,timeout=None,operation="request"):
        deadline=time.monotonic()+timeout if timeout is not None else None
        while True:
            wait_seconds=1.0
            if deadline is not None:
                remaining=deadline-time.monotonic()
                if remaining<=0.0:
                    raise TimeoutError(
                        f"PeTTaChainer worker {operation} exceeded its "
                        f"{timeout:g}s outer deadline"
                    )
                wait_seconds=min(wait_seconds,remaining)
            if connection.poll(wait_seconds):
                break
            if not process.is_alive():
                raise RuntimeError(
                    f"PeTTaChainer worker exited unexpectedly ({process.exitcode})"
                )
        try:
            status,payload=connection.recv()
        except EOFError as exc:
            exit_code=process.exitcode
            raise RuntimeError(f"PeTTaChainer worker exited unexpectedly ({exit_code})") from exc
        if status!="ok": raise RuntimeError(f"PeTTaChainer worker failure: {payload}")
        return payload

    def _launch_locked(self,statements,mutations):
        parent,child=self._context.Pipe()
        process=self._context.Process(
            target=self._worker_target,
            args=(child,tuple(statements),tuple(mutations)),
            name="recommendation-pettachainer",daemon=True,
        )
        try:
            process.start(); child.close()
            self._receive(
                parent,process,timeout=self._startup_timeout_seconds,
                operation="startup",
            )
        except BaseException:
            try: child.close()
            except OSError: pass
            self._terminate(parent,process)
            raise
        return parent,process

    def _recover_locked(self):
        if not self._has_snapshot:
            raise RuntimeError("PeTTaChainer worker has no recoverable snapshot")
        connection,process=self._launch_locked(
            self._bootstrap_statements,self._mutation_journal
        )
        self._connection,self._process=connection,process
        self._recovery_count+=1

    def replace(self,statements):
        """Atomically replace the scorer with a clean compiled snapshot."""
        with self._lock:
            statements=tuple(statements)
            parent,process=self._launch_locked(statements,())
            old_connection,old_process=self._connection,self._process
            self._connection,self._process=parent,process
            self._bootstrap_statements=statements
            self._has_snapshot=True
            self._mutation_journal=[]
            self._journal_statements=0
            self._shutdown(old_connection,old_process)

    def _rpc(self,command,payload,*,timeout_sec=None,journal=False):
        with self._lock:
            if self._connection is None or self._process is None:
                raise RuntimeError("PeTTaChainer worker has no compiled rule snapshot")
            if timeout_sec is None:
                timeout=self._default_timeout_seconds
            else:
                try: requested_timeout=float(timeout_sec)
                except (TypeError,ValueError) as exc:
                    raise ValueError(
                        "reasoner RPC timeout must be a finite number greater than 0"
                    ) from exc
                # Zero was the historical spelling for "no in-process
                # timeout". Preserve call compatibility without preserving an
                # unbounded outer wait: it now selects the finite default.
                timeout=(self._default_timeout_seconds
                         if requested_timeout==0.0 else self._finite_timeout(
                             requested_timeout,"reasoner RPC timeout"
                         ))
            journal_units=0
            if journal:
                journal_units=(len(payload) if command=="add" else 1)
                if len(self._mutation_journal)+1>self._max_journal_mutations:
                    raise RuntimeError(
                        "PeTTaChainer mutation journal limit exceeded before "
                        "worker mutation; promote a fresh rule snapshot"
                    )
                if (self._journal_statements+journal_units
                        > self._max_journal_statements):
                    raise RuntimeError(
                        "PeTTaChainer statement journal limit exceeded before "
                        "worker mutation; promote a fresh rule snapshot"
                    )
            connection,process=self._connection,self._process
            try:
                connection.send((command,payload))
                result=self._receive(
                    connection,process,timeout=timeout,operation=command
                )
                if journal:
                    committed=(tuple(payload) if command=="add" else str(payload))
                    self._mutation_journal.append((command,committed))
                    self._journal_statements+=journal_units
                return result
            except TimeoutError as exc:
                # A deadline makes the worker state unknowable. Kill it rather
                # than consuming a late response, then restore the last fully
                # acknowledged snapshot. The failed caller still receives a
                # timeout and decides whether its operation is safe to retry.
                self._connection=None; self._process=None
                self._terminate(connection,process)
                recovered=False; recovery_error=None
                try:
                    self._recover_locked(); recovered=True
                except BaseException as recovery_exc:
                    recovery_error=(
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    )
                self._last_timeout={
                    "command":str(command),"timeout_seconds":timeout,
                    "worker_recovered":recovered,
                    "recovery_error":recovery_error,
                    "occurred_at":time.time(),
                }
                suffix=("worker snapshot recovered" if recovered else
                        f"worker recovery failed ({recovery_error})")
                raise TimeoutError(f"{exc}; {suffix}") from exc
            except BaseException:
                # A failed RPC may leave unread protocol bytes or partial MeTTa
                # state. Retire it; the caller can remine a clean snapshot.
                self._connection=None; self._process=None
                self._shutdown(connection,process)
                raise

    def add_atoms_no_check(self,atoms,timeout_sec=None):
        return self._rpc(
            "add",list(atoms),timeout_sec=timeout_sec,journal=True
        )

    def remove_statement(self,atom_name,timeout_sec=None):
        return self._rpc(
            "remove",str(atom_name),timeout_sec=timeout_sec,journal=True
        )

    def query_many(self,atoms,steps=100,timeout_sec=None):
        # The worker is itself the timeout/isolation boundary; its in-process
        # PeTTa query must not fork a second runtime.
        return self._rpc(
            "query_many",(list(atoms),int(steps)),timeout_sec=timeout_sec
        )

    def close(self):
        with self._lock:
            connection,process=self._connection,self._process
            self._connection=None; self._process=None
            self._shutdown(connection,process)

    def __del__(self):
        try: self.close()
        except BaseException: pass


class Lab:
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
    def article(self, aid):
        try: return self._articles[str(aid)]
        except KeyError as exc: raise ValueError(f"unknown article: {aid}") from exc

    @staticmethod
    def _presentation_from_annotation(annotation):
        """Return bounded, human-readable semantic labels for feed cards."""
        if not isinstance(annotation,dict):
            return {}
        canonical=(annotation.get("provenance",{}).get("canonicalization",{})
                   .get("mappings",{}))

        def labels(field,limit):
            result=[]
            for item in canonical.get(field,[]) or []:
                value=(item.get("lexical") if isinstance(item,dict) else None)
                if value and value not in result:
                    result.append(str(value))
                if len(result)>=limit:
                    break
            if result:
                return result
            raw_values=annotation.get(field,[]) or []
            if isinstance(raw_values,str):
                raw_values=[raw_values]
            for value in raw_values:
                # Portable IDs retain a readable slug before the content hash.
                text=str(value).split(":",1)[-1].split("~",1)[0]
                text=text.replace("-"," ").strip()
                if text and text not in result:
                    result.append(text)
                if len(result)>=limit:
                    break
            return result

        formats=labels("format",1)
        if not formats and annotation.get("format"):
            value=str(annotation["format"]).split(":",1)[-1].split("~",1)[0]
            formats=[value.replace("-"," ")]
        return {
            "concepts":labels("concepts",5),
            "audiences":labels("audiences",3),
            "events":labels("event_types",2),
            "intents":labels("intents",2),
            "semantic_format":formats[0] if formats else None,
        }

    def public_article(self, aid):
        """Article content plus non-ranking metadata safe for the browser."""
        article=dict(self.article(aid))
        presentation=self._article_presentation.get(str(aid))
        if presentation:
            article["presentation"]=presentation
        return article

    def preview_semantics(self, article):
        """Parse immutable content through one bounded, versioned preview slot."""
        if self.symbolic_only:
            raise PeTTaChainerConfigurationError("NL2PLN is disabled in symbolic-only mode")
        if self.semantic_client is None:
            raise PeTTaChainerConfigurationError(
                "semantic parser configuration is invalid"
            )
        article_id=str(article.get("id",""))
        payload=json.dumps({
            "title":str(article.get("title","")),
            "abstract":str(article.get("abstract","")),
            "base_url":self.semantic_client.base_url,
            "knowledge_base":self.semantic_client.knowledge_base,
            "predicate_schema":RECOMMENDATION_PREDICATE_SCHEMA,
            "contract":EXPECTED_NL2PLN_CONTRACT,
            "semantic_cache_version":self.semantic_client.semantic_cache_version,
        },ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
        key=hashlib.sha256(payload).hexdigest()
        # The semantic lock is deliberately separate from ``Lab.lock``. It is
        # a single-flight/model-quota boundary, but callers wait at most one
        # second rather than consuming unbounded HTTP threads behind a slow
        # provider request.
        if not self._semantic_lock.acquire(timeout=SEMANTIC_SINGLE_FLIGHT_WAIT_SECONDS):
            raise SemanticPreviewBusyError("semantic preview is busy")
        try:
            now=time.monotonic()
            while self._semantic_preview_cache:
                oldest_key,oldest=next(iter(self._semantic_preview_cache.items()))
                if oldest["expires_at"]>now:
                    break
                self._semantic_preview_cache.pop(oldest_key)
                self._semantic_preview_cache_bytes-=oldest["bytes"]
            cached=self._semantic_preview_cache.get(key)
            if cached is not None:
                if cached["expires_at"]<=now:
                    self._semantic_preview_cache.pop(key)
                    self._semantic_preview_cache_bytes-=cached["bytes"]
                else:
                    self._semantic_preview_cache.move_to_end(key)
                    return {**cached["result"],"article_id":article_id,
                            "cached":True,"content_hash":key,
                            "lab_instance_id":self.instance_id}
            result={**self.semantic_client.parse_article(article),"cached":False,
                    "content_hash":key,"lab_instance_id":self.instance_id}
            cache_result={**result,"cached":False}
            cache_bytes=len(json.dumps(
                cache_result,ensure_ascii=False,separators=(",",":")
            ).encode("utf-8"))
            if cache_bytes<=SEMANTIC_CACHE_MAX_BYTES:
                self._semantic_preview_cache[key]={
                    "result":cache_result,"bytes":cache_bytes,
                    "expires_at":now+SEMANTIC_CACHE_TTL_SECONDS,
                }
                self._semantic_preview_cache_bytes+=cache_bytes
                while (len(self._semantic_preview_cache)>SEMANTIC_CACHE_MAX_ENTRIES
                       or self._semantic_preview_cache_bytes>SEMANTIC_CACHE_MAX_BYTES):
                    _discarded_key,discarded=self._semantic_preview_cache.popitem(last=False)
                    self._semantic_preview_cache_bytes-=discarded["bytes"]
            return result
        finally:
            self._semantic_lock.release()
    def user_topics(self,user):
        try: profile=self.data["users"][user]
        except KeyError as exc: raise ValueError(f"unknown user: {user}") from exc
        if isinstance(profile,dict): return profile.get("topics",profile.get("interests",[]))
        return profile

    def _mutable_user_profile(self,user):
        """Return a mutable live profile without requiring one dataset schema.

        Small fixtures historically store a bare topic list, while MIND
        adapters store a mapping with causal history.  Normalizing only when
        the user actually sends feedback keeps both input contracts valid and
        gives live negative evidence one explicit, bounded home.
        """
        profile=self.data["users"][user]
        if isinstance(profile,dict): return profile
        profile={"topics":list(profile or []),"history":[],"negative_history":[]}
        self.data["users"][user]=profile
        return profile

    def _negative_feedback_features(self,user,article):
        """Project recent skips into one exclusive symbolic candidate fact.

        Exact, subcategory, and topic matches are deliberately mutually
        exclusive.  They are correlated consequences of one observation and
        must not be revised by PeTTa as three independent pieces of evidence.
        """
        profile=self.data["users"].get(user,{})
        negative_history=(profile.get("negative_history",[])
                          if isinstance(profile,dict) else [])
        negative_history=[str(aid) for aid in negative_history
                          if str(aid) in self._articles]
        aid=str(article["id"])
        if aid in negative_history:
            match="exact"
        else:
            recent=[self._articles[item]
                    for item in negative_history[-LIVE_NEGATIVE_GENERALIZATION_WINDOW:]]
            subcategory=str(article.get("subcategory","unknown"))
            topic=str(article.get("topic",article.get("category","unknown")))
            if (subcategory!="unknown" and any(
                    str(item.get("subcategory","unknown"))==subcategory
                    for item in recent)):
                match="subcategory"
            elif (topic!="unknown" and any(
                    str(item.get("topic",item.get("category","unknown")))==topic
                    for item in recent)):
                match="topic"
            else:
                match="none"
        return {LIVE_NEGATIVE_FEATURE:match}

    def features(self,user,article,history_workspace=None):
        topic=article.get("topic",article.get("category","unknown"))
        article_format=article.get("format",article.get("subcategory","article"))
        profile=self.data["users"].get(user,{})
        history=(profile.get("history",[]) if isinstance(profile,dict) else [])
        if isinstance(profile,dict) and "history" in profile:
            attrs=history_feature_context(
                article,history,self._articles,
                entity_vectors=self._article_entity_vectors,
                text_semantic_vectors=self._article_text_vectors,
                title_idf_model=self._title_idf_model,
                transition_model=self.data.get("subcategory_transition_model"),
                workspace=history_workspace,
            )
            if self._lexical_idf_model:
                attrs.update(build_lexical_workspace_facts(
                    article, (self._articles.get(aid,{}) for aid in history),
                    self._lexical_idf_model,
                ))
            if self._semantic_workspace_model:
                attrs.update(build_semantic_workspace_facts(
                    str(article["id"]),history,self._article_text_vectors,
                    self._semantic_workspace_model,
                ))
            if self._recency_workspace:
                attrs.update(build_recency_workspace_facts(
                    str(article["id"]),history,self._article_text_vectors,
                ))
            if self._llm_workspace:
                attrs.update(build_llm_workspace_facts(
                    str(article["id"]),history,self._llm_article_annotations,
                ))
            attrs.update(self._negative_feedback_features(user,article))
            return attrs
        interested=topic in self.user_topics(user)
        recent_subcategories=(profile.get("recent_subcategories",[])
                              if isinstance(profile,dict) else [])
        transition=subcategory_transition_score(
            article.get("subcategory"),recent_subcategories,
            self.data.get("subcategory_transition_model"),
        )
        return {"topic":topic,"subcategory":article.get("subcategory","unknown"),
                "format":article_format,"affinity":"high" if interested else "low",
                "topic_affinity":1.0 if interested else 0.0,
                "recent_topic_affinity":1.0 if interested else 0.0,
                "subcategory_affinity_score":0.0,
                "affinity_level":"high" if interested else "none",
                "recent_affinity":"high" if interested else "none",
                "long_affinity":"high" if interested else "none",
                "history_size_bucket":"unknown","entity_overlap":"unknown",
                "entity_overlap_detail":"unknown","history_topic_count_bucket":"unknown",
                "recent_topic_count_bucket":"unknown","topic_rank_bucket":"unknown",
                "subcategory_affinity":"unknown",
                "time_bucket":"unknown","ctr_bucket":article.get("ctr_bucket","unknown"),
                "freshness_bucket":article.get("freshness_bucket","unknown"),
                "position_bucket":article.get("position_bucket","unknown"),
                "title_overlap_detail":article.get("title_overlap_detail","unknown"),
                "entity_long_mean_similarity":None,
                "title_history_idf_jaccard":None,
                "recent_subcategory_transition_score":transition,
                **self._negative_feedback_features(user,article)}

    def contextual_features(self,user,article,context=None,
                            history_workspace=None):
        # A persisted pre-impression snapshot is authoritative, including
        # missing values. Never fill its absent evidence from a later live
        # profile: that can import future history into cold-start replay.
        if context and ("history_size_bucket" in context
                        or any(key in context for key in (*RECENCY_WORKSPACE_FEATURES,*LLM_WORKSPACE_FEATURES))):
            attrs={"topic":article.get("topic",article.get("category","unknown")),
                   "subcategory":article.get("subcategory","unknown"),
                   "format":article.get("format","article")}
        else:
            attrs=self.features(
                user,article,history_workspace=history_workspace
            )
        if context:
            attrs.update({
                key:str(context[key])[:200] for key in CONTEXT_FEATURES
                if key in context and context[key] not in (None,"","unknown")
                and (not key.startswith("rel_")
                     or self.config.get("relational_evidence_mode")=="chained")
            })
        return {key:str(value) for key,value in attrs.items()
                if key in CONTEXT_FEATURES and value not in (None,"","unknown")}

    def event_features(self,event):
        # Dataset adapters persist the exact pre-outcome context on each replay
        # event.  Recomputing it from the user's latest profile is both
        # temporally wrong and, for a large corpus, needlessly expensive.
        # Minimal fixtures do not carry such a snapshot and use the live path.
        if any(key in event for key in (
            *RECENCY_WORKSPACE_FEATURES,
            *LLM_WORKSPACE_FEATURES,
            "recent_affinity", "long_affinity", "history_size_bucket",
            "history_topic_count_bucket", "recent_topic_count_bucket",
        )):
            article=self.article(event["article"])
            attrs={
                "topic":article.get("topic",article.get("category","unknown")),
                "subcategory":article.get("subcategory","unknown"),
                "format":article.get("format",article.get("subcategory","article")),
            }
            attrs.update({
                key:str(event[key])[:200] for key in CONTEXT_FEATURES
                if key in event and event[key] not in (None,"","unknown")
                and (not key.startswith("rel_")
                     or self.config.get("relational_evidence_mode")=="chained")
            })
            return {key:str(value) for key,value in attrs.items()
                    if key in CONTEXT_FEATURES and value not in (None,"","unknown")}
        return self.contextual_features(
            event["user"],self.article(event["article"]),event
        )

    def _bounded_features(self,attrs):
        if not self._feature_vocabulary: return attrs
        bounded={}
        for predicate,value in attrs.items():
            allowed=self._feature_vocabulary.get(predicate)
            if not allowed: continue
            bounded[predicate]=value if value in allowed else "other"
        return bounded

    @staticmethod
    def _positive_label(value):
        return value is True or value == 1 or str(value).lower() in POSITIVE|{"1","true","positive"}

    def _case_candidates(self,case):
        raw=case.get("candidates",case.get("candidate_ids",case.get("articles",[])))
        labels=case.get("labels")
        label_map={str(key):value for key,value in labels.items()} if isinstance(labels,dict) else {}
        ids=[]; embedded_relevant=[]
        for index,item in enumerate(raw):
            label=None
            if isinstance(item,dict):
                aid=item.get("article",item.get("article_id",item.get("news_id",item.get("item",item.get("id")))))
                label=item.get("label",item.get("clicked",item.get("relevant",item.get("action"))))
            elif isinstance(item,(list,tuple)) and len(item)==2:
                aid,label=item
            else:
                aid=item
                # Accept raw MIND impression tokens such as N12345-1, but only
                # strip the suffix when the stripped id exists in this dataset.
                match=re.fullmatch(r"(.+)-([01])",str(aid))
                if match and str(aid) not in self._articles and match.group(1) in self._articles:
                    aid,label=match.group(1),match.group(2)
            if aid is None: continue
            aid=str(aid)
            if aid not in self._articles: raise ValueError(f"unknown candidate article: {aid}")
            if isinstance(labels,list) and index<len(labels): label=labels[index]
            elif aid in label_map: label=label_map[aid]
            if self._positive_label(label): embedded_relevant.append(aid)
            if aid not in ids: ids.append(aid)
        relevant=case.get("relevant",case.get("clicked_articles",case.get("clicked",case.get("positive_ids",case.get("positive",[])))))
        if isinstance(relevant,(str,int)): relevant=[relevant]
        relevant_ids=[]
        for item in relevant or []:
            if isinstance(item,dict): item=item.get("article",item.get("article_id",item.get("id")))
            if item is not None and str(item) in ids: relevant_ids.append(str(item))
        return ids,list(dict.fromkeys([*relevant_ids,*embedded_relevant]))

    def evaluation_cases(self):
        source=self.data.get("eval_impressions",self.data.get("evaluation",self.data.get("tests",self.data.get("impressions",[]))))
        cases=[]
        for raw in source:
            user=raw.get("user",raw.get("user_id"))
            if user not in self.data["users"]: raise ValueError(f"unknown evaluation user: {user}")
            candidates,relevant=self._case_candidates(raw)
            if candidates and relevant:
                candidate_context=raw.get("candidate_context",{})
                supplied_digest=raw.get("training_confirmation_slate_digest")
                if supplied_digest is not None:
                    from ..evaluation.training_gate import confirmation_slate_digest
                    computed_digest=confirmation_slate_digest(
                        user,candidates,relevant,candidate_context
                    )
                    if supplied_digest!=computed_digest:
                        raise ValueError(
                            "training confirmation slate changed before evaluation"
                        )
                cases.append({"id":raw.get("id",raw.get("impression_id")),"user":user,
                              "source_impression_id":raw.get("source_impression_id"),
                              "candidates":candidates,"relevant":relevant,
                              "candidate_context":candidate_context,
                              "training_confirmation_slate_digest":supplied_digest})
        return cases

    def default_candidates(self,user):
        configured=self.data.get("candidate_sets")
        if isinstance(configured,dict) and user in configured:
            candidates,_=self._case_candidates({"candidates":configured[user]})
            if candidates: return candidates
        if isinstance(configured,list):
            for case in configured:
                if case.get("user",case.get("user_id"))==user:
                    candidates,_=self._case_candidates(case)
                    if candidates: return candidates
        for case in self.evaluation_cases():
            if case["user"]==user: return case["candidates"]
        return list(self._articles)

    def default_candidate_context(self,user):
        source=self.data.get("eval_impressions",self.data.get("evaluation",self.data.get("tests",self.data.get("impressions",[]))))
        for case in source:
            if case.get("user",case.get("user_id"))==user:
                return case.get("candidate_context",{})
        return {}

    def _feed_pool(self,user):
        """Return unseen corpus IDs; live context is recomputed during scoring."""
        configured=self.data.get("candidate_sets")
        if isinstance(configured,dict) and user in configured:
            source=[{"user":user,"candidates":configured[user]}]
        elif isinstance(configured,list):
            source=[case for case in configured if case.get("user",case.get("user_id"))==user]
        else:
            source=self.data.get("eval_impressions",self.data.get("evaluation",self.data.get("tests",self.data.get("impressions",[]))))
            source=[case for case in source if case.get("user",case.get("user_id"))==user]
        profile=self.data.get("users",{}).get(user,{})
        profile_history=(profile.get("history",[]) if isinstance(profile,dict) else [])
        previously_seen={
            str(aid) for aid in profile_history if str(aid) in self._articles
        }
        for case in source:
            previously_seen.update(
                str(aid) for aid in case.get("history",[])
                if str(aid) in self._articles
            )
        pool=[]; seen=set()
        for case in source:
            candidates,_relevant=self._case_candidates(case)
            for aid in candidates:
                if aid in seen or aid in previously_seen: continue
                # Evaluation candidate_context belongs to the historical
                # impression replay. A live feed must derive context from the
                # current user/article state; carrying every historical
                # per-item context here explodes proof identities and makes a
                # cold first page wait on dozens of redundant queries.
                seen.add(aid); pool.append((aid,{}))
        # MIND usually exposes only one labeled impression per evaluation user.
        # Add the rest of the loaded news corpus so scrolling is a genuine feed;
        # the live scorer derives context afresh for every article.
        remaining=[aid for aid in self._articles if aid not in seen and aid not in previously_seen]
        stable=f'{self.config["random_seed"]}:{user}'.encode()
        local=random.Random(int.from_bytes(hashlib.sha256(stable).digest()[:8],"big"))
        local.shuffle(remaining)
        for aid in remaining:
            seen.add(aid); pool.append((aid,{}))
        return pool

    def _prune_feed_sessions(self,now):
        expired=[session for session,value in self._feed_sessions.items()
                 if now-value["last_access"]>3600]
        for session in expired: self._feed_sessions.pop(session,None)
        if len(self._feed_sessions)>128:
            oldest=sorted(self._feed_sessions,key=lambda key:self._feed_sessions[key]["last_access"])
            for session in oldest[:len(self._feed_sessions)-128]: self._feed_sessions.pop(session,None)

    def _feed_session_for_user(self,session,user):
        state=self._feed_sessions.get(session) if session else None
        if state is not None and state.get("user")!=user:
            raise ValueError("feed session belongs to another user")
        return state

    @staticmethod
    def _acknowledge_feed_deliveries(state,position):
        """Forget pages that a subsequent cursor proves were accepted."""
        state["deliveries"]=[
            delivery for delivery in state.get("deliveries",[])
            if int(delivery["end_position"])>int(position)
        ]

    @staticmethod
    def _record_feed_delivery(state,*,start_position,end_position,rows):
        """Retain a bounded tail so an aborted HTTP response can be undone."""
        if not rows:
            return
        deliveries=state.setdefault("deliveries",[])
        deliveries.append({
            "start_position":int(start_position),
            "end_position":int(end_position),
            "queue_revision":int(state["queue_revision"]),
            "rows":list(rows),
        })
        retained_rows=sum(len(delivery["rows"]) for delivery in deliveries)
        while len(deliveries)>1 and (
                len(deliveries)>FEED_DELIVERY_HISTORY_LIMIT
                or retained_rows>FEED_DELIVERY_ROW_LIMIT):
            retained_rows-=len(deliveries[0]["rows"])
            del deliveries[0]

    @staticmethod
    def _recover_unaccepted_feed_deliveries(
            state,*,accepted_position,accepted_revision):
        """Put server-delivered but client-unaccepted rows back in the queue.

        The browser accepts responses atomically and reports the last accepted
        position with feedback.  If an already-running GET crossed the server
        boundary after that position, its rows are recoverable from the
        bounded delivery tail and must participate in the feedback rerank.
        """
        if type(accepted_position) is not int or accepted_position<0:
            raise ValueError("feed_position must be a non-negative integer")
        if type(accepted_revision) is not int or accepted_revision<0:
            raise ValueError("queue_revision must be a non-negative integer")
        if accepted_revision!=state["queue_revision"]:
            raise ValueError("stale feedback queue revision")
        current_position=int(state["position"])
        if accepted_position>current_position:
            raise ValueError("feed_position is ahead of the feed session")

        deliveries=list(state.get("deliveries",[]))
        pending=[delivery for delivery in deliveries
                 if int(delivery["end_position"])>accepted_position]
        recovered=[]
        cursor=accepted_position
        for delivery in pending:
            start=int(delivery["start_position"])
            end=int(delivery["end_position"])
            if start!=cursor or int(delivery["queue_revision"])!=accepted_revision:
                raise ValueError("feed_position is outside retained delivery history")
            recovered.extend(delivery["rows"])
            cursor=end
        if cursor!=current_position:
            raise ValueError("feed_position is outside retained delivery history")

        if recovered:
            state["queue"]=recovered+state["queue"]
            for row in recovered:
                key=(str(row["impression"]),str(row["article"]["id"]))
                state.get("feedback_contexts",{}).pop(key,None)
                state.get("feedback_positions",{}).pop(key,None)
            state["position"]=accepted_position
        # The accepted prefix belongs to the old queue revision and cannot be
        # useful after feedback creates a new revision.
        state["deliveries"]=[]
        return {
            "accepted_position":accepted_position,
            "server_position_before_recovery":current_position,
            "recovered_rows":len(recovered),
        }

    @staticmethod
    def _live_feedback_signature(row):
        """Return stable, proof-backed online-feedback evidence for one row."""
        evidence=row.get("feedback_evidence",{}) or {}
        declared_ids={str(value) for value in evidence.get("rule_ids",[])}
        proof_rows=[]; proven_ids=set()
        for raw_proof in row.get("proofs",[]) or []:
            proof=str(raw_proof)
            ids=tuple(sorted(
                LIVE_NEGATIVE_RULE_IDS.intersection(
                    re.findall(r"\bfeedback_skip_[a-z]+\b",proof)
                )
            ))
            if not ids:
                continue
            proven_ids.update(ids)
            proof_rows.append((
                ids,
                tuple(round(value,12) for value in proof_tv(proof)),
                re.sub(r"\s+"," ",proof).strip(),
            ))
        # Metadata alone is not causal provenance. Keep only rule IDs actually
        # present in a returned PeTTa proof while retaining mismatches in the
        # diagnostic signature so they cannot compare equal silently.
        rule_ids=tuple(sorted(declared_ids & proven_ids))
        proof_rows=tuple(sorted(set(proof_rows)))
        proof_stv=evidence.get("proof_stv") or {}
        payload={
            "match":str(evidence.get("match","none")),
            "rule_ids":rule_ids,
            "declared_rule_ids":tuple(sorted(declared_ids)),
            "proven_rule_ids":tuple(sorted(proven_ids)),
            "proof_stv":(
                round(float(proof_stv.get("strength",0.0)),12),
                round(float(proof_stv.get("confidence",0.0)),12),
            ) if proof_stv else None,
            "proofs":proof_rows,
        }
        encoded=json.dumps(payload,sort_keys=True,separators=(",",":"))
        return {
            **payload,
            "proof_texts":[item[2] for item in proof_rows],
            "signature":hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]
                if rule_ids and proof_rows else None,
        }

    def feed_page(self,user,*,cursor=None,session=None,limit=None):
        """Return the next page from a window-proof-ranked feed session."""
        with self.lock:
            if user not in self.data["users"]: raise ValueError(f"unknown user: {user}")
            page_size=self.config["top_k"] if limit is None else int(limit)
            if page_size<1 or page_size>1000: raise ValueError("feed limit must be between 1 and 1000")
            cursor_position=None; cursor_revision=None
            if cursor:
                try:
                    cursor_session,raw_position,raw_revision=cursor.rsplit(":",2)
                    cursor_position=int(raw_position); cursor_revision=int(raw_revision)
                except (AttributeError,ValueError) as exc: raise ValueError("invalid feed cursor") from exc
                if session and session!=cursor_session: raise ValueError("feed cursor does not match session")
                session=cursor_session
            now=time.time(); self._prune_feed_sessions(now); reset=False
            # Validate ownership before applying stale-cursor reset semantics;
            # a token must never become usable merely because its rule version
            # is old.
            state=self._feed_session_for_user(session,user)
            if ((state is not None and state.get("version")!=self.version)
                    or (cursor and state is None)):
                # A mining/configuration change invalidates the old ordering.
                # Start a fresh proof-ranked stream and tell the UI to reset.
                # Validate every explicit session, not only cursor requests: the
                # HTTP API also accepts ``?session=...`` and must never drain a
                # queue that was proof-ranked by an older rule snapshot.
                session=None; cursor_position=None; cursor_revision=None; state=None; reset=True
            if session is None:
                session=uuid.uuid4().hex
                pool=self._feed_pool(user)
                self._feed_sessions[session]={"user":user,"items":pool,"source_position":0,
                                              "queue":[],"position":0,"last_access":now,
                                              "version":self.version,"feedback_contexts":{},
                                              "feedback_positions":{},"deliveries":[],
                                              "pair_replay":None,
                                              "queue_revision":0,
                                              "last_feedback_revision":None}
                state=self._feed_sessions[session]
            if state is None: raise ValueError("unknown or expired feed session")
            if cursor_position is not None and cursor_position!=state["position"]:
                raise ValueError("stale feed cursor")
            if cursor_revision is not None and cursor_revision!=state["queue_revision"]:
                raise ValueError("stale feed cursor revision")
            if cursor_position is not None:
                # Possession of this cursor proves the preceding response was
                # accepted by the client, so its rollback copy can be dropped.
                self._acknowledge_feed_deliveries(state,cursor_position)
            # Treat a bounded slice of the corpus as newly arrived inventory.
            # PeTTaChainer proof-ranks each arrival window before any item from
            # that window can reach the client, keeping first-page latency low.
            window_size=max(page_size,int(self.config["feed_window"]))
            while len(state["queue"])<page_size and state["source_position"]<len(state["items"]):
                source_start=state["source_position"]
                source_end=min(len(state["items"]),source_start+window_size)
                arriving=state["items"][source_start:source_end]
                candidates=[aid for aid,_context in arriving]
                contexts={aid:context for aid,context in arriving}
                rows=(self.score(
                    user,candidates,contexts,limit=0,include_context=True,
                    cache_result=False,
                ) if candidates else [])
                state["pair_replay"]=getattr(self,"_last_pair_replay",None)
                impression=f"live_{session}_{source_start}_{source_end}"
                queued=[]
                for row in rows:
                    aid=row["article"]["id"]
                    prepared_context=row.get("_prepared_context")
                    served_context=(dict(prepared_context)
                                    if prepared_context is not None else
                                    self.contextual_features(
                                        user,row["article"],contexts.get(aid)
                                    ))
                    relation=row.get("relational_evidence",{})
                    served_context.update(relation.get("scopes",{}))
                    served_context.update({
                        key:list(value) for key,value
                        in relation.get("proof_ids",{}).items()
                    })
                    public_row={key:value for key,value in row.items()
                                if key!="_prepared_context"}
                    queued.append({**public_row,"context":served_context,
                                   "impression":impression,
                                   "queue_revision":state["queue_revision"]})
                state["queue"].extend(queued)
                state["source_position"]=source_end
            delivery_start=state["position"]
            rows=state["queue"][:page_size]; del state["queue"][:len(rows)]
            delivery_end=delivery_start+len(rows)
            # Only rows that actually crossed the HTTP boundary are eligible
            # for feedback.  Previously all 40 proof-ranked rows were
            # registered before five were emitted, allowing a caller to react
            # to an item the browser had never received.
            for row in rows:
                aid=str(row["article"]["id"])
                key=(str(row["impression"]),aid)
                state["feedback_contexts"][key]=row["context"]
                state["feedback_positions"][key]=delivery_end
            state["position"]=delivery_end; state["last_access"]=now
            self._record_feed_delivery(
                state,start_position=delivery_start,
                end_position=delivery_end,rows=rows,
            )
            has_more=bool(state["queue"] or state["source_position"]<len(state["items"]))
            next_cursor=(f"{session}:{state['position']}:{state['queue_revision']}"
                         if has_more else None)
            return {"feed":rows,"session":session,"cursor":next_cursor,
                    "next_cursor":next_cursor,"has_more":has_more,
                    "position":state["position"],
                    "remaining":len(state["items"])-state["position"],"total_candidates":len(state["items"]),
                    "rule_version":self.version,"reset":reset,
                    "queue_revision":state["queue_revision"],
                    "last_feedback_revision":state["last_feedback_revision"]}

    def _rerank_unserved_queue(self,session_id,state,*,action,article):
        """Proof-rank one live session's already-arrived, unserved inventory."""
        before_rows=list(state["queue"])
        before=[str(row["article"]["id"]) for row in before_rows]
        before_by_article={
            str(row["article"]["id"]):(index+1,row)
            for index,row in enumerate(before_rows)
        }
        state["queue_revision"]+=1
        revision=state["queue_revision"]
        rerank_mode="full"
        recomputed_candidates=len(before)
        if before:
            can_reuse_pairwise=(
                action=="skip"
                and all(isinstance(row.get("pairwise_score"),(int,float))
                        for row in before_rows)
            )
            replay_source=state.get("pair_replay")
            replay_available=(self._pairwise_replay_plan(
                before_rows,state.get("pair_replay")
            ) if can_reuse_pairwise else None)
            can_reuse_pairwise=(can_reuse_pairwise
                                and replay_available is not None)
            changed=[]
            revised_contexts={}
            if can_reuse_pairwise:
                for aid in before:
                    row=before_by_article[aid][1]
                    revised_context=dict(row.get("context",{}))
                    old_match=str(
                        revised_context.get(
                            LIVE_NEGATIVE_FEATURE,
                            row.get("feedback_evidence",{}).get("match","none"),
                        )
                    )
                    new_match=self._negative_feedback_features(
                        state["user"],self.article(aid)
                    )[LIVE_NEGATIVE_FEATURE]
                    revised_context[LIVE_NEGATIVE_FEATURE]=new_match
                    revised_contexts[aid]=revised_context
                    if old_match!=new_match:
                        changed.append(aid)
            if can_reuse_pairwise:
                changed_rows=(self.score(
                    state["user"],changed,
                    contexts={aid:revised_contexts[aid] for aid in changed},
                    limit=0,
                    include_context=True,apply_pairwise=False,
                    cache_result=False,
                ) if changed else [])
                changed_by_article={
                    str(row["article"]["id"]):row for row in changed_rows
                }
                reranked=[]
                for aid in before:
                    previous=before_by_article[aid][1]
                    if aid not in changed_by_article:
                        reranked.append(dict(previous))
                        continue
                    updated=changed_by_article[aid]
                    reranked.append(updated)
                reranked.sort(key=lambda row:(
                    -row["score"],-row["stv"]["strength"],
                    -row["stv"]["confidence"],
                    -row["tie_break"]["topic_prior"],
                    -row["tie_break"]["format_prior"],
                    -row["tie_break"]["subcategory_prior"],
                    row["article"]["id"],
                ))
                plan,proof_map=self._pairwise_replay_plan(
                    reranked,replay_source
                )
                reranked=self._pairwise_rank(
                    reranked,(),plan=plan,proof_map=proof_map,
                    reasoner_timeout_sec=self._serving_reasoner_timeout(),
                )
                rerank_mode="incremental_point_reuse_pairwise"
                recomputed_candidates=len(changed)
            else:
                reranked=self.score(
                    state["user"],before,contexts={},limit=0,
                    include_context=True,cache_result=False,
                )
            impression=(f"live_{session_id}_{state['position']}_"
                        f"{state['source_position']}")
            state["queue"]=[]
            for row in reranked:
                aid=str(row["article"]["id"])
                prepared=row.get("_prepared_context")
                context=(dict(prepared) if prepared is not None else
                         dict(before_by_article[aid][1].get("context",{})))
                if not context:
                    context=self.contextual_features(
                        state["user"],row["article"],None
                    )
                relation=row.get("relational_evidence",{})
                context.update(relation.get("scopes",{}))
                context.update({
                    key:list(value) for key,value
                    in relation.get("proof_ids",{}).items()
                })
                public_row={key:value for key,value in row.items()
                            if key!="_prepared_context"}
                state["queue"].append({
                    **public_row,"context":context,"impression":impression,
                    "queue_revision":revision,
                })
        after=[str(row["article"]["id"]) for row in state["queue"]]
        state["version"]=self.version
        old_positions={aid:index+1 for index,aid in enumerate(before)}
        movements=[
            {"article":aid,"from":old_positions[aid],"to":index+1,
             "delta":old_positions[aid]-(index+1)}
            for index,aid in enumerate(after)
            if old_positions.get(aid)!=index+1
        ]
        affected=[]; observed=[]
        for after_rank,row in enumerate(state["queue"],1):
            aid=str(row["article"]["id"])
            before_rank,before_row=before_by_article[aid]
            before_evidence=self._live_feedback_signature(before_row)
            after_evidence=self._live_feedback_signature(row)
            rule_ids=list(after_evidence["rule_ids"])
            if not rule_ids:
                continue
            before_score=float(before_row.get("score",0.0))
            after_score=float(row.get("score",0.0))
            before_ranking=float(before_row.get("ranking_score",before_score))
            after_ranking=float(row.get("ranking_score",after_score))
            introduced=before_evidence["signature"] is None
            evidence_changed=(
                after_evidence["signature"] is not None
                and after_evidence["signature"]!=before_evidence["signature"]
            )
            score_decreased=after_score<before_score
            ranking_score_decreased=after_ranking<before_ranking
            diagnostic={
                "article":aid,
                "before_rank":before_rank,"after_rank":after_rank,
                "rank_delta":after_rank-before_rank,
                "before_score":before_score,"after_score":after_score,
                "score_delta":round(after_score-before_score,8),
                "before_ranking_score":before_ranking,
                "after_ranking_score":after_ranking,
                "ranking_score_delta":round(after_ranking-before_ranking,8),
                "score_method":row.get("score_method"),
                "before_match":before_evidence["match"],
                "match":after_evidence["match"],
                "before_rule_ids":list(before_evidence["rule_ids"]),
                "rule_ids":rule_ids,
                "before_proofs":before_evidence["proof_texts"],
                "proofs":after_evidence["proof_texts"],
                "before_feedback_signature":before_evidence["signature"],
                "after_feedback_signature":after_evidence["signature"],
                "feedback_evidence_introduced":introduced,
                "feedback_evidence_changed":evidence_changed,
                "own_score_decreased":score_decreased,
                "own_ranking_score_decreased":ranking_score_decreased,
                "causal_eligible":bool(
                    evidence_changed and (score_decreased or ranking_score_decreased)
                ),
            }
            observed.append(diagnostic)
            if evidence_changed:
                affected.append(diagnostic)
        # A rank can worsen merely because some *other* row moved. Causality is
        # certified only when this candidate gained/changed proof-backed
        # feedback evidence and its own point or final ranking score fell.
        causal_demotion=next((
            item for item in affected if item["causal_eligible"]
        ),None)
        has_more=bool(state["queue"] or state["source_position"]<len(state["items"]))
        next_cursor=(f"{session_id}:{state['position']}:{revision}"
                     if has_more else None)
        summary={
            "session":session_id,"revision":revision,"action":action,
            "feedback_article":str(article),"reranked_candidates":len(after),
            "rerank_mode":rerank_mode,
            "recomputed_candidates":recomputed_candidates,
            "reused_candidates":max(0,len(after)-recomputed_candidates),
            "pairwise_reused":rerank_mode=="incremental_point_reuse_pairwise",
            "changed_positions":len(movements),
            "top_before":before[0] if before else None,
            "top_after":after[0] if after else None,
            "movements":movements[:20],"negative_proof_candidates":affected[:20],
            "negative_proof_observations":observed[:20],
            "unchanged_negative_proof_candidates":sum(
                not item["feedback_evidence_changed"] for item in observed
            ),
            "causal_demotion":causal_demotion,"next_cursor":next_cursor,
            "reasoner":"PeTTaChainer",
        }
        state["last_feedback_revision"]=summary
        state["last_access"]=time.time()
        return summary

    @staticmethod
    def candidate_case(user,aid,attrs):
        signature=json.dumps(attrs,sort_keys=True,separators=(",",":"),ensure_ascii=False)
        # Serving rules only inspect these mined feature predicates. Reusing a
        # grounded context for identical feature triples is semantically exact
        # and keeps a large impression replay bounded.
        digest=hashlib.blake2s(signature.encode("utf-8"),digest_size=10).hexdigest()
        return f"candidate_{digest}"

    @staticmethod
    def _relational_proof_references(context):
        """Return normalized relation-proof references from one context.

        Proof IDs are audit dependencies rather than scoring features. Keep
        them out of fpMiner while still treating malformed or duplicate
        references as invalid provenance at the serving boundary.
        """
        if not isinstance(context,dict):
            return {}
        found={}
        for proof_field in RELATIONAL_PROOF_FIELDS.values():
            if proof_field not in context:
                continue
            references=context[proof_field]
            if not isinstance(references,(list,tuple)) or any(
                    not isinstance(reference,str) or not reference
                    for reference in references):
                raise ValueError(
                    "relational proof references must be nonempty string lists"
                )
            normalized=tuple(references)
            if len(normalized)!=len(set(normalized)):
                raise ValueError("relational proof references must be unique")
            found[proof_field]=normalized
        return found

    def _validate_relational_context_provenance(self,context,*,user,article):
        """Bind each relational scope to one coherent proved history snapshot."""
        references=self._relational_proof_references(context)
        immutable=self.data.get("relational_proof_ledger",{})
        if not isinstance(immutable,dict):
            immutable={}
        live=getattr(self,"_live_relational_proof_ledger",{})
        expected_families={
            REL_ENTITY_CONTINUITY_SCOPE:"wikidata_entity_continuity",
            REL_CONCEPT_CONTINUITY_SCOPE:"canonical_concept_continuity",
        }
        causal_histories=set()
        for scope_field,proof_field in RELATIONAL_PROOF_FIELDS.items():
            field_references=references.get(proof_field,())
            scope=context.get(scope_field)
            if scope is not None and scope not in {
                    "unknown","none","older","recent"}:
                raise ValueError("invalid relational scope")
            if scope in {"older","recent"} and not field_references:
                raise ValueError(
                    "a positive relational scope requires proof references"
                )
            if field_references and scope not in {"older","recent"}:
                raise ValueError(
                    "positive relational proof references require a positive scope"
                )
            records=[]
            for proof_id in field_references:
                immutable_record=immutable.get(proof_id)
                live_record=live.get(proof_id)
                if (immutable_record is not None and live_record is not None
                        and immutable_record!=live_record):
                    raise RuntimeError("relational proof ID collision")
                record=(live_record if live_record is not None
                        else immutable_record)
                if not isinstance(record,dict):
                    raise ValueError("relational proof reference is not available")
                if (str(record.get("user_id"))!=str(user)
                        or str(record.get("candidate_id"))!=str(article)):
                    raise ValueError(
                        "relational proof reference belongs to another context"
                    )
                if record.get("relation_family")!=expected_families[scope_field]:
                    raise ValueError(
                        "relational proof reference belongs to another family"
                    )
                records.append(record)
            if records:
                record_scopes={str(record.get("recency"))
                               for record in records}
                derived_scope=("recent" if "recent" in record_scopes
                               else "older")
                if not record_scopes.issubset({"older","recent"}) or scope!=derived_scope:
                    raise ValueError(
                        "relational scope disagrees with proof recency"
                    )
                for identity_field in (
                        "case_id","scope_id","causal_history_id"):
                    identities={record.get(identity_field) for record in records}
                    if len(identities)!=1 or None in identities:
                        raise ValueError(
                            "relational proof references mix incompatible contexts"
                        )
                causal_histories.add(records[0]["causal_history_id"])
        if len(causal_histories)>1:
            raise ValueError(
                "relational proof families use different causal histories"
            )
        return references

    def _referenced_live_relational_proof_ids(self):
        """Return proof IDs that must survive the next worker promotion."""
        references=set()
        events=self.data.get("events",[])
        if isinstance(events,list):
            start=max(0,int(getattr(self,"_offline_event_count",0)))
            for event in events[start:]:
                for values in self._relational_proof_references(event).values():
                    references.update(values)
        for state in getattr(self,"_feed_sessions",{}).values():
            if not isinstance(state,dict):
                continue
            contexts=state.get("feedback_contexts",{})
            if not isinstance(contexts,dict):
                continue
            for context in contexts.values():
                for values in self._relational_proof_references(context).values():
                    references.update(values)
        return references

    def _retained_live_relational_proofs(self,*additional_ledgers):
        """Snapshot live proof records referenced by events or served cards."""
        immutable=self.data.get("relational_proof_ledger",{})
        if not isinstance(immutable,dict):
            immutable={}
        ledgers=[getattr(self,"_live_relational_proof_ledger",{}),
                 *additional_ledgers]
        combined={}
        for ledger in ledgers:
            if not isinstance(ledger,dict):
                continue
            for proof_id,record in ledger.items():
                previous=combined.get(proof_id)
                if previous is not None and previous!=record:
                    raise RuntimeError("relational proof ID collision")
                combined[proof_id]=record
        retained={}
        missing=[]
        for proof_id in sorted(self._referenced_live_relational_proof_ids()):
            if proof_id in immutable:
                continue
            record=combined.get(proof_id)
            if record is None:
                missing.append(proof_id)
            else:
                retained[proof_id]=copy.deepcopy(record)
        if missing:
            raise RuntimeError(
                "live relational proof references would become dangling: "
                +", ".join(missing[:5])
            )
        return retained

    def _live_relational_features(
            self,user,candidates,required_features,*,timeout_sec=None):
        """Derive live candidate relations through the active PeTTa worker.

        Historical replay contexts already contain their causal, pre-outcome
        relational projection. This path is exclusively for a live profile: a
        positive interaction changes its ordered click history, creating new
        versioned relation scopes and therefore new proof queries. Every
        positive conclusion remains gated by a returned two-hop PeTTa proof.
        """
        selected=set(required_features).intersection({
            REL_ENTITY_CONTINUITY_SCOPE,
            REL_CONCEPT_CONTINUITY_SCOPE,
        })
        if (not selected
                or self.config.get("relational_evidence_mode")!="chained"):
            return {str(aid):{} for aid in candidates}
        timeout_sec=(self._serving_reasoner_timeout()
                     if timeout_sec is None else float(timeout_sec))
        profile=self.data["users"].get(user,{})
        raw_history=(profile.get("history")
                     if isinstance(profile,dict) and "history" in profile
                     else None)
        history_available=(
            isinstance(raw_history,(list,tuple))
            and all(isinstance(aid,str) and bool(aid) for aid in raw_history)
        )
        normalized_history=(tuple(raw_history) if history_available else ())
        history_truncated=len(normalized_history)>RELATIONAL_LIVE_HISTORY_LIMIT
        # Unknown IDs are evidence gaps. Preserve them so the builders emit
        # ``unknown`` instead of manufacturing a closed-world ``none`` from a
        # silently filtered history.
        history=normalized_history[-RELATIONAL_LIVE_HISTORY_LIMIT:]
        entity_observations={}; concept_observations={}
        plans=[]
        for aid in candidates:
            entity_plan,concept_plan=build_relational_plans(
                str(aid),history,self._articles,
                self._llm_article_annotations,user_id=str(user),
                entity_observation_cache=entity_observations,
                concept_observation_cache=concept_observations,
            )
            if REL_ENTITY_CONTINUITY_SCOPE in selected:
                plans.append((str(aid),REL_ENTITY_CONTINUITY_SCOPE,
                              entity_plan,reduce_relational_proofs))
            if REL_CONCEPT_CONTINUITY_SCOPE in selected:
                plans.append((str(aid),REL_CONCEPT_CONTINUITY_SCOPE,
                              concept_plan,reduce_concept_relational_proofs))

        missing=[]; statements=set()
        for aid,field,plan,reducer in plans:
            cache_key=(field,plan.case_id,history_truncated,history_available)
            if cache_key in self._relational_feature_cache:
                continue
            missing.append((aid,field,plan,reducer))
            if plan.requires_query:
                statements.update(plan.statements)
        cache_limit=int(self.config.get(
            "max_proof_cache_entries",DEFAULT_MAX_PROOF_CACHE_ENTRIES
        ))
        if len(self._relational_feature_cache)+len(missing)>cache_limit:
            raise RuntimeError(
                "relational feature cache limit exceeded; promote a fresh "
                "rule snapshot"
            )
        new_statements=sorted(
            statement for statement in statements
            if statement not in self._loaded_relational_statements
        )
        for offset in range(0,len(new_statements),1000):
            statement_batch=new_statements[offset:offset+1000]
            self.engine.add_atoms_no_check(
                statement_batch,timeout_sec=timeout_sec
            )
            # Publish each acknowledged mutation immediately. If a later batch
            # fails, the worker journal recovers these facts and a retry must
            # not insert named duplicates and inflate revision confidence.
            self._loaded_relational_statements.update(statement_batch)

        proof_results={}
        queryable=[item for item in missing if item[2].requires_query]
        query_roots=[
            (item,root) for item in queryable for root in item[2].proof_roots
        ]
        if any(not item[2].proof_roots for item in queryable):
            raise RuntimeError(
                "a queryable live relation has no specific proof roots"
            )
        query_batch=max(1,int(self.config.get("query_batch_size",512)))
        for offset in range(0,len(query_roots),query_batch):
            batch=query_roots[offset:offset+query_batch]
            results=self.engine.query_many(
                [root.query for _item,root in batch],
                steps=max(8,RELATIONAL_QUERY_STEPS_PER_ROOT*len(batch)),
                timeout_sec=timeout_sec,
            )
            if not isinstance(results,(list,tuple)) or len(results)!=len(batch):
                raise RuntimeError(
                    "PeTTaChainer returned an invalid live relational batch"
                )
            self._relational_query_calls+=1
            self._relational_query_roots+=len(batch)
            for ((_aid,_field,plan,_reducer),root),proofs in zip(batch,results):
                if not isinstance(proofs,(list,tuple)) or not proofs:
                    raise RuntimeError(
                        "a required live relational proof root returned no "
                        "PeTTa proof: "
                        f"{root.origin_id}/{root.matched_value_id}"
                    )
                proof_results.setdefault(plan.case_id,[]).extend(proofs)

        staged_cache={}; staged_ledger={}
        for _aid,field,plan,reducer in missing:
            facts,ledger=reducer(plan,proof_results.get(plan.case_id,()))
            facts=dict(facts)
            if ((history_truncated or not history_available)
                    and facts.get(field)=="none"):
                # No match inside the bounded window says nothing about the
                # omitted prefix. This is abstention, not negative knowledge.
                facts[field]="unknown"
            staged_cache[(
                field,plan.case_id,history_truncated,history_available,
            )]=facts
            for proof_id,record in ledger.items():
                previous=(self._live_relational_proof_ledger.get(proof_id)
                          or staged_ledger.get(proof_id))
                if previous is not None and previous!=record:
                    raise RuntimeError("stable live relational proof ID collision")
                staged_ledger[proof_id]=record
        self._relational_feature_cache.update(staged_cache)
        self._live_relational_proof_ledger.update(staged_ledger)

        projected={str(aid):{} for aid in candidates}
        for aid,field,plan,_reducer in plans:
            facts=self._relational_feature_cache[
                (field,plan.case_id,history_truncated,history_available)
            ]
            value=facts.get(field)
            proof_field=RELATIONAL_PROOF_FIELDS[field]
            projected[aid][proof_field]=list(facts.get(proof_field,()))
            # Unknown means source annotations were incomplete. It must
            # abstain, not become a provider-coverage ranking signal.
            if value!="unknown":
                projected[aid][field]=value
        return projected

    def _candidate_specs(self,user,candidates,contexts=None):
        contexts=contexts or {}
        # A proof can only inspect predicates selected by the active mining
        # profile.  Projecting grounded contexts to that profile is both
        # semantically exact and important for a large MIND replay: temporal
        # fields that are not in the rule vocabulary must not create a new
        # candidate identity (and therefore a new PeTTa query) for every
        # impression.
        active=set(FEATURE_PROFILES[self.config["feature_profile"]])
        # A persisted replay value is authoritative even when it is
        # ``unknown``. Derive only fields absent from the supplied historical
        # context; this prevents current live history leaking into replay.
        live_required={
            feature for feature in active
            if feature in {
                REL_ENTITY_CONTINUITY_SCOPE,
                REL_CONCEPT_CONTINUITY_SCOPE,
            }
            and any(feature not in (contexts.get(str(aid)) or {})
                    for aid in candidates)
        }
        live_relational=self._live_relational_features(
            user,candidates,live_required,
            timeout_sec=self._serving_reasoner_timeout(),
        )
        profile=self.data["users"].get(user,{})
        history=(profile.get("history",[]) if isinstance(profile,dict) else [])
        needs_live_history=any(
            not (contexts.get(str(aid)) or {}).get("history_size_bucket")
            for aid in candidates
        )
        history_workspace=(prepare_history_feature_workspace(
            history,self._articles,
            entity_vectors=self._article_entity_vectors,
            text_semantic_vectors=self._article_text_vectors,
        ) if isinstance(profile,dict) and "history" in profile
             and needs_live_history else None)
        specs=[]
        feature_started=time.perf_counter()
        def prepare(aid):
            supplied=contexts.get(aid) or {}
            raw_attrs=self.contextual_features(
                user,self.article(aid),contexts.get(aid),
                history_workspace=history_workspace,
            )
            for feature,value in live_relational.get(str(aid),{}).items():
                if feature not in supplied:
                    raw_attrs[feature]=value
            return aid,supplied,raw_attrs
        executor=getattr(self,"_candidate_feature_executor",None)
        if executor is not None and len(candidates)>1:
            prepared=list(executor.map(prepare,candidates))
        else:
            prepared=[prepare(aid) for aid in candidates]
        feature_seconds=time.perf_counter()-feature_started
        projection_seconds=0.0
        case_serialization_seconds=0.0
        for aid,supplied,raw_attrs in prepared:
            started=time.perf_counter()
            attrs=self._bounded_features(raw_attrs)
            attrs={key:value for key,value in attrs.items() if key in active}
            # Online negative feedback is a serving fact, not a mined feature
            # vocabulary value.  Keep it available under every experimental
            # profile so the generic PeTTa feedback policy cannot silently
            # disappear when an operator changes the mining profile.
            if LIVE_NEGATIVE_FEATURE in raw_attrs:
                attrs[LIVE_NEGATIVE_FEATURE]=raw_attrs[LIVE_NEGATIVE_FEATURE]
            projection_seconds+=time.perf_counter()-started
            started=time.perf_counter()
            relational_refs={}
            for _scope_field,proof_field in RELATIONAL_PROOF_FIELDS.items():
                references=raw_attrs.get(
                    proof_field,supplied.get(proof_field,())
                )
                if isinstance(references,(list,tuple)):
                    relational_refs[proof_field]=tuple(map(str,references))
            case_identity=dict(attrs)
            nonempty_refs={key:value for key,value in relational_refs.items()
                           if value}
            if nonempty_refs:
                # A proof-positive categorical value may be reusable, but its
                # source occurrence is not interchangeable. Bind the scoring
                # case to exact proof IDs so returned provenance cannot drift
                # across users or historical snapshots with the same scope.
                case_identity["__relational_proof_ids__"]=nonempty_refs
            case=self.candidate_case(user,aid,case_identity)
            previous=self._candidate_relational_proof_refs.get(case)
            if previous is not None and previous!=relational_refs:
                raise RuntimeError("candidate relation-provenance collision")
            self._candidate_relational_proof_refs[case]=relational_refs
            case_serialization_seconds+=time.perf_counter()-started
            specs.append((aid,case,attrs,raw_attrs))
        self._last_candidate_preparation_profile={
            "candidate_history_feature_calculation_seconds":feature_seconds,
            "candidate_feature_projection_seconds":projection_seconds,
            "candidate_case_serialization_seconds":case_serialization_seconds,
            "candidates":len(specs),
        }
        return specs

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
        timeout_sec=(self._serving_reasoner_timeout()
                     if timeout_sec is None else float(timeout_sec))
        facts=[]; missing=set()
        for aid,case,attrs,_raw_attrs in specs:
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
        return tuple(sorted(topology,key=lambda item:(item[0],item[1])))

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

    @staticmethod
    def _mining_workspace_plan(kind,selected_features,depth,case_policy):
        """Return the stable identity of one exact AtomSpace projection."""
        return json.dumps({
            "schema":MINING_WORKSPACE_SCHEMA_VERSION,
            "kind":str(kind),
            "features":list(selected_features),
            "depth":int(depth),
            "case_policy":str(case_policy),
        },sort_keys=True,separators=(",",":"),ensure_ascii=True)

    def _retained_mining_population(self):
        """Freeze the newest complete causal units used by every miner stage.

        A source impression is atomic.  Records without an impression are
        already closed singleton interactions and receive a stable event unit.
        The returned source indices remain unchanged when the window appends,
        which preserves exact delta case identities until a unit expires.
        """
        records=[]
        for event_index,event in enumerate(self.data.get("events",[])):
            impression=event.get("impression")
            unit=(f"impression:{impression}" if impression
                  else f"event:{event_index}")
            records.append((unit,(event_index,event)))
        return retain_complete_units(
            records,
            max_units=int(self.config["mining_retention_max_units"]),
            max_cases=int(self.config["mining_retention_max_cases"]),
        )

    def _prune_mining_workspace_cache(self):
        """Best-effort cleanup after a complete rule snapshot is usable."""
        try:
            with MINER_LOCK:
                removed=self._mining_workspaces.prune(
                    keep=self._active_mining_workspace_plans
                )
                support_removed=self._incremental_fpminer.prune(
                    keep=self._active_mining_workspace_plans
                )
            return {"removed":list(removed),
                    "support_removed":list(support_removed),"error":None}
        except Exception as exc:
            # Cache maintenance must never roll back a scorer that compiled
            # successfully. A failed clear is marked dirty by the cache and the
            # next synchronization will rebuild that space before trusting it.
            return {
                "removed":[],
                "support_removed":[],
                "error":f"{type(exc).__name__}: {exc}"[:1000],
            }

    def _mine_incremental_workspace(
        self, synced, *, plan, cases, features, depth, min_support,
        retention_units, retention_audit, workspace_kind,
    ):
        """Mine one synced space and release a rejected non-active stage."""
        try:
            return self._incremental_fpminer.mine(
                plan=plan,full_space=synced.space,cases=cases,
                features=features,depth=depth,min_support=min_support,
                evidence_k=float(self.config["ctv_evidence_k"]),
                retention_units=retention_units,
                retention_audit=retention_audit,
            )
        except BaseException as exc:
            # Active spaces remain aligned with the last usable scorer and can
            # safely retry a failed transactional support update. A new or
            # otherwise non-active stage has no usable support snapshot, so it
            # must not consume a workspace-cache slot after rejection.
            if plan not in self._active_mining_workspace_plans:
                try:
                    self._mining_workspaces.rollback(synced)
                except BaseException as rollback_exc:
                    if hasattr(exc,"add_note"):
                        exc.add_note(
                            f"failed to roll back rejected {workspace_kind} "
                            "mining workspace: "
                            f"{type(rollback_exc).__name__}: {rollback_exc}"
                        )
            raise

    def _mining_event_cases(self,indexed_events=None):
        """Sample point cases while retaining their append-stable source IDs."""
        indexed=(list(indexed_events) if indexed_events is not None else
                 list(enumerate(self.data["events"])))
        ratio=int(self.config["negative_ratio"])
        if ratio<=0 or not any(event.get("impression")
                               for _index,event in indexed):
            return [(f"point_case_{index}",event) for index,event in indexed]
        groups={}
        for index,event in indexed:
            key=event.get("impression") or f'live_{event.get("user","unknown")}'
            groups.setdefault(str(key),[]).append((index,event))
        selected=set()
        for key,items in groups.items():
            positives=[item for item in items if item[1].get("action") in POSITIVE]
            negatives=[item for item in items if item[1].get("action") not in POSITIVE]
            selected.update(index for index,_event in positives)
            if not positives:
                # Logged MIND impressions without a click remain excluded by
                # the historical click-conditioned sampler.  A newly observed
                # live impression is different: dropping its only skip would
                # make the online outcome visible to calibration but invisible
                # to fpMiner discovery.  Retain the first bounded negatives so
                # later appends cannot replace an earlier selected case and
                # force the incremental workspace to rebuild.
                is_online_live=(key.startswith("live_") and any(
                    index>=self._offline_event_count for index,_event in items
                ))
                if is_online_live and negatives:
                    online_negatives=[
                        item for item in negatives
                        if item[0]>=self._offline_event_count
                    ]
                    selected.update(
                        index for index,_event in online_negatives[:ratio]
                    )
                continue
            if not negatives: continue
            count=min(len(negatives),ratio*len(positives))
            stable=f'{self.config["random_seed"]}:{key}'.encode()
            local=random.Random(int.from_bytes(hashlib.sha256(stable).digest()[:8],"big"))
            selected.update(index for index,_event in local.sample(negatives,count))
        return [(f"point_case_{index}",event)
                for index,event in indexed if index in selected]

    def _online_negative_sampling_audit(self,training_cases,indexed_events=None):
        """Describe bounded live-negative retention in a mining snapshot."""
        retained_indices={
            int(case_id.rsplit("_",1)[-1]) for case_id,_event in training_cases
        }
        groups={}
        indexed=(list(indexed_events) if indexed_events is not None else
                 list(enumerate(self.data["events"])))
        for index,event in indexed:
            key=event.get("impression") or f'live_{event.get("user","unknown")}'
            groups.setdefault(str(key),[]).append((index,event))
        eligible=set()
        for key,items in groups.items():
            if (not key.startswith("live_")
                    or not any(index>=self._offline_event_count
                               for index,_event in items)
                    or any(event.get("action") in POSITIVE
                           for _index,event in items)):
                continue
            eligible.update(
                index for index,event in items
                if (index>=self._offline_event_count
                    and event.get("action") not in POSITIVE)
            )
        return {
            "policy":"first_observed_append_stable",
            "cap_per_all_negative_live_impression":int(
                self.config["negative_ratio"]
            ),
            "eligible":len(eligible),
            "retained":len(eligible & retained_indices),
        }

    def _mining_events(self):
        """Compatibility view of the sampled point events without case IDs."""
        return [event for _case_id,event in self._mining_event_cases()]

    @staticmethod
    def _relative_value(left, right, order):
        """Encode a comparison without exposing either training outcome."""
        left=str(left or "unknown"); right=str(right or "unknown")
        if left==right: return "equal"
        ranks={value:index for index,value in enumerate(order)}
        if left not in ranks and right not in ranks: return "incomparable"
        if left not in ranks: return "right_known"
        if right not in ranks: return "left_known"
        return "left" if ranks[left]>ranks[right] else "right"

    @staticmethod
    def _relative_numeric_value(left, right):
        """Bound a numeric comparison to six stable symbolic outcomes."""
        def finite(value):
            try:
                number=float(value)
            except (TypeError,ValueError):
                return None
            return number if math.isfinite(number) else None
        left_number=finite(left); right_number=finite(right)
        if left_number is None and right_number is None: return "unknown"
        if left_number is None: return "right_known"
        if right_number is None: return "left_known"
        if math.isclose(left_number,right_number,rel_tol=1e-12,abs_tol=1e-12):
            return "equal"
        return "left" if left_number>right_number else "right"

    @staticmethod
    def _numeric_missing(value):
        """Return whether numeric observation is absent or non-finite."""
        if value is None or isinstance(value,bool): return True
        try: number=float(value)
        except (TypeError,ValueError,OverflowError): return True
        return not math.isfinite(number)

    @staticmethod
    def _category_symbol(value):
        """Return a stable, parser-safe symbolic category plus its label.

        Ordinary MIND labels remain readable (for example ``news``). Values
        containing quoting/control syntax are represented by a readable slug
        and content hash so they cannot corrupt the MeTTa form parser or alias
        two distinct taxonomy nodes.
        """
        if not isinstance(value,str): return None,None
        label=unicodedata.normalize("NFKC"," ".join(value.split())).casefold()
        if label in {"","unknown","none","null"}: return None,None
        if re.fullmatch(r"[a-z0-9][a-z0-9._:/-]{0,127}",label):
            return label,label
        ascii_label=(unicodedata.normalize("NFKD",label)
                     .encode("ascii","ignore").decode("ascii"))
        slug=re.sub(r"[^a-z0-9._:-]+","_",ascii_label).strip("_.:-")[:48]
        digest=hashlib.sha256(label.encode("utf-8")).hexdigest()[:16]
        return f"{slug or 'category'}__{digest}",label

    def _pair_features(self,left_attrs,right_attrs,left_article,right_article,
                       needed=None):
        needed=set(needed) if needed is not None else None
        wants=lambda predicate: needed is None or predicate in needed
        plan=_pair_feature_execution_plan(
            None if needed is None else tuple(sorted(needed))
        )
        attrs={}
        left_scope=left_attrs.get("history_size_bucket","unknown")
        right_scope=right_attrs.get("history_size_bucket","unknown")
        # This is a context shared by both pair orientations, not an identity
        # or outcome. Unary support is neutral; only a mined conjunction can
        # establish that a preference depends on the amount of prior history.
        if wants("pair_history_scope"):
            attrs["pair_history_scope"]=(
                left_scope if left_scope==right_scope else "unknown"
            )
        for output,source in plan["lexical"]:
            attrs[output]=self._relative_numeric_value(
                left_attrs.get(source),right_attrs.get(source)
            )
        for output,source in plan["semantic_workspace"]:
            attrs[output]=self._relative_numeric_value(
                left_attrs.get(source),right_attrs.get(source)
            )
        for output,source in plan["ordered"]:
            if source in RELATIONAL_PROOF_FIELDS and (
                    left_attrs.get(source) not in {"none","older","recent"}
                    or right_attrs.get(source)
                       not in {"none","older","recent"}):
                # Annotation absence/truncation is not directional evidence.
                # In particular, never encode it as left_known/right_known.
                continue
            if source in left_attrs or source in right_attrs:
                attrs[output]=self._relative_value(
                    left_attrs.get(source),right_attrs.get(source),PAIR_ORDERS[source]
                )
        for output,source in plan["direct_numeric"]:
            attrs[output]=self._relative_numeric_value(
                left_attrs.get(source),right_attrs.get(source)
            )
        for output,source in plan["multi_interest"]:
            if source in LLM_NUMERIC_FEATURES and (
                    self._numeric_missing(left_attrs.get(source))
                    or self._numeric_missing(right_attrs.get(source))):
                # Budget-limited extraction coverage is not a preference.
                # Missing annotation on one side cannot become left_known or
                # right_known evidence for choosing the annotated candidate.
                attrs[output]="incomparable"
                continue
            attrs[output]=self._relative_numeric_value(
                left_attrs.get(source),right_attrs.get(source)
            )
        for source,predicate in plan["quantile"]:
            if source in LLM_NUMERIC_FEATURES and (
                    self._numeric_missing(left_attrs.get(source))
                    or self._numeric_missing(right_attrs.get(source))):
                # Annotation absence is not directional evidence. This is
                # stricter than the generic encoder's useful left_known /
                # right_known states because LLM coverage may be budget- or
                # provider-dependent and must never decide the ranking.
                attrs[predicate]="incomparable"
                continue
            encoder=self._numeric_pair_encoders.get(source)
            if encoder is not None:
                attrs[predicate]=encoder.encode_pair(
                    left_attrs.get(source),right_attrs.get(source)
                )
        # A mined symbolic multi-interest ensemble.  The feature construction
        # only states which side wins a majority of three independent views;
        # it does not prescribe that the predicate predicts a click.  The
        # MeTTa miner must discover and calibrate that implication from closed
        # same-impression outcomes before PeTTaChainer can use it.
        if wants("pair_stable_dominance"):
            stable_values={}
            for feature,source in PAIR_STABLE_DOMINANCE_SOURCES.items():
                value=attrs.get(feature)
                if value is None and (source in left_attrs or source in right_attrs):
                    value=self._relative_value(
                        left_attrs.get(source),right_attrs.get(source),
                        PAIR_ORDERS[source],
                    )
                stable_values[feature]=value
            left_votes=sum(value=="left" for value in stable_values.values())
            right_votes=sum(value=="right" for value in stable_values.values())
            attrs["pair_stable_dominance"]=(
                "left" if left_votes>right_votes else
                "right" if right_votes>left_votes else "equal"
            )
        left_topic=str(left_article.get("topic","unknown"))
        right_topic=str(right_article.get("topic","unknown"))
        left_subcategory=str(left_article.get("subcategory","unknown"))
        right_subcategory=str(right_article.get("subcategory","unknown"))
        if wants("pair_same_topic"):
            attrs["pair_same_topic"]=(
                "same" if left_topic==right_topic else "different"
            )
        if wants("pair_same_subcategory"):
            attrs["pair_same_subcategory"]=(
                "same" if left_subcategory==right_subcategory else "different"
            )
        wanted_categorical=(needed is None
                            or bool(needed & PAIR_CATEGORICAL_SIDE_PREDICATES))
        if not wanted_categorical:
            return attrs
        left_topic_value,left_topic_label=self._category_symbol(left_topic)
        right_topic_value,right_topic_label=self._category_symbol(right_topic)
        left_subcategory_value,left_subcategory_label=self._category_symbol(
            left_subcategory
        )
        right_subcategory_value,right_subcategory_label=self._category_symbol(
            right_subcategory
        )
        left_format_value,left_format_label=self._category_symbol(
            left_attrs.get("llm_format")
        )
        right_format_value,right_format_label=self._category_symbol(
            right_attrs.get("llm_format")
        )
        categorical_values={
            "topic":(left_topic_value,right_topic_value),
            # A subcategory label is meaningful only under its parent topic;
            # namespacing prevents generic labels from colliding across trees.
            "subcategory":(
                (f"topic={left_topic_value}|subcategory={left_subcategory_value}"
                 if left_topic_value and left_subcategory_value else None),
                (f"topic={right_topic_value}|subcategory={right_subcategory_value}"
                 if right_topic_value and right_subcategory_value else None),
            ),
            "llm_format":(left_format_value,right_format_value),
        }
        categorical_labels={
            left_topic_value:left_topic_label,
            right_topic_value:right_topic_label,
            left_format_value:left_format_label,
            right_format_value:right_format_label,
        }
        if left_topic_value and left_subcategory_value:
            categorical_labels[categorical_values["subcategory"][0]]=(
                f"topic={left_topic_label}|subcategory={left_subcategory_label}"
            )
        if right_topic_value and right_subcategory_value:
            categorical_labels[categorical_values["subcategory"][1]]=(
                f"topic={right_topic_label}|subcategory={right_subcategory_label}"
            )
        label_map=getattr(self,"_pair_categorical_labels",None)
        if label_map is not None:
            label_map.update({symbol:label for symbol,label in categorical_labels.items()
                              if symbol is not None and label is not None})
        for family,(left_predicate,right_predicate) in (
                PAIR_CATEGORICAL_SIDE_FAMILIES.items()):
            left_value,right_value=categorical_values[family]
            # Omit the complete family unless both candidates are grounded;
            # annotation/metadata availability can never become a preference.
            if left_value is None or right_value is None: continue
            if wants(left_predicate): attrs[left_predicate]=left_value
            if wants(right_predicate): attrs[right_predicate]=right_value
        return attrs

    def _bounded_pair_features(self,attrs):
        bounded={}
        categorical=PAIR_CATEGORICAL_SIDE_PREDICATES
        for _family,(left_predicate,right_predicate) in (
                PAIR_CATEGORICAL_SIDE_FAMILIES.items()):
            if left_predicate not in attrs and right_predicate not in attrs:
                continue
            left_allowed=self._pair_feature_vocabulary.get(left_predicate)
            right_allowed=self._pair_feature_vocabulary.get(right_predicate)
            # A side category is meaningful only as a fully observed pair.
            # If either side is absent/OOV, the whole family abstains.
            if (left_predicate in attrs and right_predicate in attrs
                    and left_allowed and right_allowed
                    and attrs[left_predicate] in left_allowed
                    and attrs[right_predicate] in right_allowed):
                bounded[left_predicate]=attrs[left_predicate]
                bounded[right_predicate]=attrs[right_predicate]
        for predicate,value in attrs.items():
            if predicate in categorical: continue
            allowed=self._pair_feature_vocabulary.get(predicate)
            if allowed:
                if value in allowed:
                    bounded[predicate]=value
                else:
                    bounded[predicate]="other"
        return bounded

    def _fit_pair_numeric_encoders(self,indexed_events=None):
        """Freeze dataset-scale-free numeric symbols from training facts only."""
        active_profile=set(PAIR_FEATURE_PROFILES[
            self.config["pair_feature_profile"]
        ])
        enabled={
            source:predicate for source,predicate in NUMERIC_PAIR_EVIDENCE.items()
            if predicate in active_profile
        }
        if not enabled:
            self._numeric_pair_encoders={}
            return
        groups={}
        indexed=(list(indexed_events) if indexed_events is not None else
                 list(enumerate(self.data.get("events",[]))))
        for _event_index,event in indexed:
            impression=event.get("impression")
            if impression: groups.setdefault(str(impression),[]).append(event)
        values={source:[] for source in enabled}
        pairs={source:[] for source in enabled}
        pair_weights={source:[] for source in enabled}

        def finite(value):
            if value is None or isinstance(value,bool): return None
            try: number=float(value)
            except (TypeError,ValueError,OverflowError): return None
            return number if math.isfinite(number) else None

        for events in groups.values():
            contextual=[(event,self.event_features(event)) for event in events]
            for _event,attrs in contextual:
                for source in values: values[source].append(attrs.get(source))
            # Magnitude bins are a label-free representation choice: use all
            # unordered candidates in the logged slate, never click-vs-skip
            # membership. Per-feature, each impression contributes total
            # mass one, preventing large slates from defining every cut.
            for source in pairs:
                candidates=[finite(attrs.get(source)) for _event,attrs in contextual]
                complete=[
                    (left,right)
                    for left_index,left in enumerate(candidates)
                    if left is not None
                    for right in candidates[left_index+1:]
                    if right is not None and left != right
                ]
                if not complete: continue
                weight=1.0/len(complete)
                pairs[source].extend(complete)
                pair_weights[source].extend([weight]*len(complete))
        bins=int(self.config["pair_numeric_bins"])
        self._numeric_pair_encoders={
            source:QuantileNumericEvidence.fit(
                source,values[source],pairs[source],
                training_pair_weights=pair_weights[source],bins=bins
            )
            for source in enabled
        }

    @staticmethod
    def _pair_mining_case_id(impression,left_event_index,right_event_index):
        """Name an oriented pair by its stable source events, not batch order."""
        signature=json.dumps({
            "schema":MINING_WORKSPACE_SCHEMA_VERSION,
            "impression":str(impression),
            "left_event_index":int(left_event_index),
            "right_event_index":int(right_event_index),
        },sort_keys=True,separators=(",",":"),ensure_ascii=True)
        digest=hashlib.sha256(signature.encode("utf-8")).hexdigest()[:32]
        return f"pair_case_{digest}"

    def _pair_training_cases(self,negative_ratio=None,indexed_events=None):
        """Create orientation-balanced clicked-vs-exposed pair cases.

        ``search_weight`` gives every closed impression total mass one.  This
        aligns target-aware discovery with the benchmark's macro-per-impression
        AUC even when impressions have very different slate sizes.  The legacy
        fpMiner path consumes raw cases.  Final CTV estimation always revisits
        the complete retained pair population; ``pair_ctv_mode`` determines
        whether those rows are counted directly or normalized by impression.
        """
        groups={}
        indexed=(list(indexed_events) if indexed_events is not None else
                 list(enumerate(self.data.get("events",[]))))
        for event_index,event in indexed:
            impression=event.get("impression")
            if impression:
                groups.setdefault(str(impression),[]).append((event_index,event))
        ratio=int(self.config["pair_negative_ratio"] if negative_ratio is None else negative_ratio)
        cases=[]; group_count=max(1,len(groups))
        for group_index,(impression,events) in enumerate(groups.items()):
            positives=[item for item in events if item[1].get("action") in POSITIVE]
            negatives=[item for item in events if item[1].get("action") not in POSITIVE]
            if not positives or not negatives: continue
            impression_cases=[]
            for positive_index,(positive_source,positive) in enumerate(positives):
                selected=negatives
                if ratio>0 and len(selected)>ratio:
                    stable=f'{self.config["random_seed"]}:{impression}:{positive_index}'.encode()
                    local=random.Random(int.from_bytes(hashlib.sha256(stable).digest()[:8],"big"))
                    selected=local.sample(selected,ratio)
                for negative_source,negative in selected:
                    # Emit both orientations. This makes the pair target exactly
                    # balanced, prevents a random left/right artifact, and gives
                    # every learned predicate an explicit anti-symmetric view.
                    for left_source,left,right_source,right,left_wins in (
                        (positive_source,positive,negative_source,negative,True),
                        (negative_source,negative,positive_source,positive,False),
                    ):
                        left_article=self.article(left["article"])
                        right_article=self.article(right["article"])
                        attrs=self._pair_features(
                            self.event_features(left),self.event_features(right),
                            left_article,right_article
                        )
                        impression_cases.append({
                            "case_id":self._pair_mining_case_id(
                                impression,left_source,right_source
                            ),
                            "impression":impression,"attrs":attrs,
                            "positive":left_wins,
                            "temporal_fold":min(2,3*group_index//group_count),
                        })
            search_weight=1.0/len(impression_cases)
            for case in impression_cases:
                case["search_weight"]=search_weight
            cases.extend(impression_cases)
        return cases

    @staticmethod
    def _pair_evidence_lineage(premises):
        """Canonical raw-evidence lineage for proof dependency grouping."""
        def canonical(predicate):
            seen=set()
            while predicate in PAIR_EVIDENCE_ALIASES and predicate not in seen:
                seen.add(predicate)
                predicate=PAIR_EVIDENCE_ALIASES[predicate]
            return predicate
        features={canonical(predicate) for predicate,_value in premises}
        return frozenset(
            "topic_interest" if feature in PAIR_REDUNDANT_INTEREST else feature
            for feature in features
        )

    @classmethod
    def _pair_dependency_lineage(cls,premises):
        """Directional evidence that may connect two proof dependencies.

        Orientation-invariant predicates can condition a useful rule, but they
        cannot identify which candidate should win. Excluding those gates here
        prevents unrelated text and taxonomy rules from becoming one dependency
        merely because both apply to the same history scope. The complete
        evidence lineage remains available to the miner above.
        """
        return cls._pair_evidence_lineage(premises).difference(
            CONDITIONAL_LLM_CONTEXT_PREDICATES
        )

    @classmethod
    def _pair_rules_share_dependency(cls,left_rule,right_rule):
        """Whether two selected rules must share one PeTTa proof dependency."""
        left_lineage=cls._pair_dependency_lineage(left_rule["premises"])
        right_lineage=cls._pair_dependency_lineage(right_rule["premises"])
        left_owner=left_rule.get("dependency_owner")
        right_owner=right_rule.get("dependency_owner")
        # An explicit owner is authoritative even when a conditional variant's
        # other premises have a distinct lineage.
        if ((left_owner and left_owner in right_lineage)
                or (right_owner and right_owner in left_lineage)
                or (left_owner and right_owner and left_owner==right_owner)):
            return True
        # Gate-only rules have no directional evidence and must not all collapse
        # into one dependency through empty-lineage equality.
        if not left_lineage or not right_lineage:
            return False
        if left_lineage==right_lineage:
            return True
        if not left_lineage.intersection(right_lineage):
            return False
        smaller=min(len(left_rule["coverage"]),len(right_rule["coverage"]))
        overlap=(len(left_rule["coverage"]&right_rule["coverage"])/smaller
                 if smaller else 1.0)
        return overlap>=0.9

    @staticmethod
    def _select_positive_residual_hyperedges(rules):
        """Compile only positive evidence increments over proper-subset rules.

        Pair rules are mined in the direction in which their premises predict a
        left-hand win.  A conjunction whose calibrated posterior is no better
        than its already-selected parents therefore has no additional positive
        evidence to contribute.  Keeping its negative residual would silently
        turn a positive mined rule into an anti-rule during proof aggregation.

        Rejected edges are deliberately not added to ``calibrated_edges``: a
        redundant or contradictory edge must not become a parent of a later
        conjunction either.
        """
        ordered=sorted(
            (dict(rule) for rule in rules),
            key=lambda rule:(rule.get("specificity",len(rule["premises"])),
                             rule["premises"]),
        )
        selected=[]; calibrated_edges=[]; rejected_nonpositive=0

        def clipped_logit(probability):
            bounded=max(1e-9,min(1.0-1e-9,float(probability)))
            return math.log(bounded/(1.0-bounded))

        for rule in ordered:
            premise_set=frozenset(rule["premises"])
            lower_edges=[previous for previous in calibrated_edges
                         if previous["premise_set"]<premise_set]
            lower_logit=math.fsum(previous["delta"] for previous in lower_edges)
            lower_probability=1.0/(1.0+math.exp(
                -max(-40.0,min(40.0,lower_logit))
            ))
            reliability=max(0.0,min(1.0,float(
                rule.get("selection_confidence",rule["confidence"])
            )))
            posterior=((1.0-reliability)*lower_probability
                       +reliability*float(rule["strength"]))
            delta=clipped_logit(posterior)-lower_logit
            if not math.isfinite(delta) or delta<=1e-12:
                rejected_nonpositive+=1
                continue
            proof_strength=1.0/(1.0+math.exp(
                -max(-40.0,min(40.0,delta))
            ))
            dependency_id=f"pair_mined_cluster_{len(selected)+1}"
            rule.update(
                id=dependency_id,dependency_id=dependency_id,
                variant_id=f"{dependency_id}_v1",
                residual_delta=delta,residual_parent_logit=lower_logit,
                residual_parent_count=len(lower_edges),
                proof_strength=proof_strength,proof_confidence=1.0,
            )
            selected.append(rule)
            calibrated_edges.append({"premise_set":premise_set,"delta":delta})
        return selected,{
            "candidates":len(ordered),
            "selected_positive":len(selected),
            "rejected_nonpositive":rejected_nonpositive,
        }

    def _mine_pairwise(self,indexed_events=None,retention=None):
        """Mine AUC-aligned preference rules, then prove them with PeTTa.

        All target-aware strategies keep the real MeTTa fpMiner unary pass as
        their discovery layer. ``target_aware`` expands all active bounded
        predicates. ``conditional_llm`` is narrower: it expands only real-
        miner LLM seeds with orientation-invariant context. The
        ``conditional_llm_seed_only`` ablation uses those unary LLM rules only
        as discovery/provenance seeds and compiles accepted conditional
        children over the established non-LLM backoff. Both conditional modes
        require a stable increment over every immediate parent before the
        ordinary population CTV calibration and PeTTa compilation below.
        """
        if retention is None:
            retention=self._retained_mining_population()
        if indexed_events is None:
            indexed_events=retention.records
        indexed_events=tuple(indexed_events)
        retention_audit=retention.audit.as_dict()
        started=time.perf_counter(); self._fit_pair_numeric_encoders(indexed_events)
        self._pair_categorical_labels={}
        strategy=self.config["miner_strategy"]
        weighted_search=strategy in {
            "target_aware","conditional_llm","conditional_llm_seed_only"
        }
        discovery_cases=self._pair_training_cases(indexed_events=indexed_events)
        if not discovery_cases:
            self._active_pair_rule_ids=set(); self.pair_rules=[]
            self._pair_rule_sources=[]
            self.last_pair_mining={
                "rules":0,"cases":0,
                "reason":"no closed positive/negative impressions",
                "retention":retention_audit,
                "workspace_mode":MINING_WORKSPACE_MODE,
                "workspace_sync":[],
                "full_structure_research":True,
            }
            return self.last_pair_mining
        # Negative sampling is only a search-time optimisation for fpMiner.
        # Recalibrate every discovered structure over every within-impression
        # click-vs-skip pair, otherwise the sampled opponents make CTVs and
        # temporal-stability decisions depend on an arbitrary mining budget.
        population_cases=(discovery_cases if int(self.config["pair_negative_ratio"])==0
                          else self._pair_training_cases(
                              negative_ratio=0,indexed_events=indexed_events
                          ))
        discovery_search_weight=math.fsum(
            case["search_weight"] for case in discovery_cases
        )
        # The two miners consume different support units. fpMiner counts raw
        # oriented rows in its MeTTa scratch space; target-aware search consumes
        # case weights whose total is one per usable impression. Never pass a
        # weighted threshold off as though fpMiner itself understood weights.
        fpminer_min_support=max(
            int(self.config["pair_min_support"]),
            math.ceil(len(discovery_cases)*0.001),
        )
        target_min_support=max(
            int(self.config["pair_min_support"]),
            math.ceil(discovery_search_weight*0.001),
        )
        vocabulary_min_support=(target_min_support
                                if weighted_search
                                else fpminer_min_support)
        calibration_min_support=max(
            int(self.config["pair_min_support"]),math.ceil(len(population_cases)*0.001)
        )
        value_counts={feature:Counter() for feature in PAIR_FEATURES}
        for case in discovery_cases:
            for feature in PAIR_FEATURES:
                if feature in case["attrs"]:
                    value_counts[feature][case["attrs"][feature]] += (
                        case["search_weight"] if weighted_search else 1
                    )
        max_values=max(2,int(self.config["max_feature_values"])); vocabulary={}
        for feature,counts in value_counts.items():
            if feature in PAIR_CATEGORICAL_SIDE_PREDICATES:
                continue
            ranked=[value for value,count in counts.most_common()
                    if value is not None and count>=vocabulary_min_support]
            if not ranked: continue
            keep=set(ranked[:max_values])
            if any(value not in keep for value in counts): keep.add("other")
            vocabulary[feature]=keep
        # Left/right categorical roles use one pooled, training-only
        # vocabulary. Mirrored cases therefore cannot keep one orientation
        # while mapping its reverse to a different symbol.
        for _family,(left_predicate,right_predicate) in (
                PAIR_CATEGORICAL_SIDE_FAMILIES.items()):
            if (left_predicate not in value_counts
                    or right_predicate not in value_counts):
                continue
            pooled=value_counts[left_predicate]+value_counts[right_predicate]
            ranked=[
                value for value,count in pooled.most_common()
                if value is not None and count>=2*vocabulary_min_support
            ]
            if not ranked: continue
            keep=set(ranked[:max_values])
            vocabulary[left_predicate]=set(keep)
            vocabulary[right_predicate]=set(keep)
        self._pair_feature_vocabulary=vocabulary
        bounded_discovery=[{**case,"attrs":self._bounded_pair_features(case["attrs"])}
                           for case in discovery_cases]
        bounded_population=[{**case,"attrs":self._bounded_pair_features(case["attrs"])}
                            for case in population_cases]
        predicate_comparability={}
        for feature in PAIR_FEATURES:
            comparable=sum(
                str(case["attrs"].get(feature,"")).lower() in {"left","right"}
                or str(case["attrs"].get(feature,"")).lower().startswith(("left_q","right_q"))
                for case in bounded_population
            )
            if comparable:
                predicate_comparability[feature]=comparable/len(bounded_population)

        workspace_sync=[]; workspace_plans=set()
        def sync_space(selected_features,depth,cases,case_policy):
            workspace_cases={}
            for case in cases:
                case_id=case["case_id"]
                facts=[]
                for predicate in selected_features:
                    if predicate in case["attrs"]:
                        value=case["attrs"][predicate]
                        if (predicate in LLM_PAIR_PREDICATES
                                and str(value).lower() in {
                                    "unknown","incomparable",
                                    "left_known","right_known",
                                }):
                            # Annotation availability is not a preference and
                            # cannot become a real-fpMiner discovery seed.
                            continue
                        facts.append(
                            f'({predicate} {case_id} '
                            f'{json.dumps(case["attrs"][predicate])})'
                        )
                outcome="click" if case["positive"] else "skip"
                facts.append(f'(engagement {case_id} {json.dumps(outcome)})')
                if case_id in workspace_cases:
                    raise RuntimeError(f"duplicate stable pair case ID: {case_id}")
                workspace_cases[case_id]=tuple(facts)
            plan_key=self._mining_workspace_plan(
                "pair",selected_features,depth,case_policy
            )
            synced=self._mining_workspaces.sync(plan_key,workspace_cases)
            workspace_plans.add(plan_key); workspace_sync.append(synced.as_dict())
            return synced,plan_key,workspace_cases

        active=[feature for feature in PAIR_FEATURE_PROFILES[
            self.config["pair_feature_profile"]
        ] if feature in vocabulary]
        # The real MeTTa miner always owns unary discovery.  Only the legacy
        # strategy enumerates the curated deeper feature combinations.
        unary_active=tuple(
            feature for feature in active
            if feature not in PAIR_CATEGORICAL_SIDE_PREDICATES
        )
        plan=[(unary_active,2,bounded_discovery,"sampled_discovery")]
        if not unary_active: plan=[]
        scoped_categorical_plans=[]
        if (int(self.config["pair_conjunctions"])>=3
                and "pair_history_scope" in active):
            scoped_categorical_plans=[
                (predicate,"pair_history_scope")
                for predicate in sorted(PAIR_CATEGORICAL_SIDE_PREDICATES)
                if predicate in active
            ]
            # Accuracy-first categorical discovery sees the exhaustive closed
            # pair population. Opponent sampling may accelerate unrelated
            # semantic discovery, but it must not erase a global cold-start
            # category before impression-macro validation can examine it.
            plan.extend((features,3,bounded_population,"full_population")
                        for features in scoped_categorical_plans)
        if (strategy=="fixed_combinations"
                and int(self.config["pair_conjunctions"])>=3):
            plan.extend((pair,3,bounded_discovery,"sampled_discovery") for pair in PAIR_INTERACTIONS
                        if all(feature in active for feature in pair))
        raw=[]; fpminer_incremental=[]
        for selected_features,depth,cases,case_policy in plan:
            synced,plan_key,workspace_cases=sync_space(
                selected_features,depth,cases,case_policy
            )
            mined=self._mine_incremental_workspace(
                synced,plan=plan_key,cases=workspace_cases,
                features=selected_features,depth=depth,
                min_support=fpminer_min_support,
                retention_units=retention.unit_ids,
                retention_audit=retention_audit,workspace_kind="pair",
            )
            raw.extend(mined.output)
            fpminer_incremental.append(mined.audit.as_dict())
        rules=parse_rules([str(value) for value in raw],features=PAIR_FEATURES)
        categorical_emitted=sum(
            any(predicate in PAIR_CATEGORICAL_SIDE_PREDICATES
                for predicate,_value in rule["premises"])
            for rule in rules
        )
        filtered_rules=[]; categorical_eligible=[]
        for rule in rules:
            side_premises=[
                (predicate,value) for predicate,value in rule["premises"]
                if predicate in PAIR_CATEGORICAL_SIDE_PREDICATES
            ]
            if not side_premises:
                filtered_rules.append(rule); continue
            scopes=[value for predicate,value in rule["premises"]
                    if predicate=="pair_history_scope"]
            # Categorical priors are allowed only as actual fpMiner-emitted,
            # scope-conditioned rules. Side unaries and cross-family joins
            # are discovery artifacts, not serving rules.
            if (len(rule["premises"])!=2 or len(side_premises)!=1
                    or len(scopes)!=1
                    or str(scopes[0]).lower() in {"unknown","incomparable"}):
                continue
            predicate,_value=side_premises[0]
            family=next(
                name for name,predicates in PAIR_CATEGORICAL_SIDE_FAMILIES.items()
                if predicate in predicates
            )
            rule.update(
                categorical_fact_family=family,
                scoped_categorical_prior=True,
                evidence_relationship="dependent_variant_not_independent_vote",
                dependency_owner=(
                    "pair_text_semantic_top3_mean"
                    if family=="llm_format" else
                    "pair_subcategory_affinity"
                    if family=="subcategory" else "topic_interest"
                ),
            )
            categorical_eligible.append(rule)
            filtered_rules.append(rule)
        rules=filtered_rules
        categorical_search={
            "actual_miner":"recommendation/miner/fpMiner.metta",
            "plans":[list(plan) for plan in scoped_categorical_plans],
            "plan_population":"full exhaustive closed pair population",
            "miner_depth":3,
            "emitted_with_side_predicate":categorical_emitted,
            "eligible_scoped_rules":len(categorical_eligible),
            "eligible_scoped_rule_records":[{
                "premises":[list(item) for item in rule["premises"]],
                "source":rule["source"],
                "discovery_ctv":rule.get("discovery_ctv"),
                "support":rule.get("support"),
            } for rule in categorical_eligible],
            "rejected_side_rules":categorical_emitted-len(categorical_eligible),
            "scope_policy":"all known causal history-size buckets",
            "vocabulary":"shared train-only left/right; serving OOV abstains",
            "host_generated_rules":0,
        }
        target_search=None
        if strategy in {"conditional_llm","conditional_llm_seed_only"}:
            semantic=tuple(feature for feature in active
                           if feature in LLM_PAIR_PREDICATES)
            context=tuple(feature for feature in active
                          if feature in CONDITIONAL_LLM_CONTEXT_PREDICATES)
            semantic_seeds=[
                FpMinerUnary(
                    rule_id=rule["id"],predicate=rule["premises"][0][0],
                    value=rule["premises"][0][1],target=rule["target"],
                    source=rule["source"],
                )
                for rule in rules
                if len(rule["premises"])==1
                and rule["premises"][0][0] in semantic
            ]
            if semantic and context and semantic_seeds:
                # Use the complete training population for the conditional
                # test. Equal-impression mass prevents large slates from
                # dominating support, while fpMiner itself still receives the
                # raw-row support unit above for its mandatory unary seeds.
                conditional_result=mine_conditional_llm_patterns(
                    ({feature:case["attrs"][feature] for feature in (*semantic,*context)
                      if feature in case["attrs"]
                      and str(case["attrs"][feature]).lower() not in {
                          "unknown","incomparable","left_known","right_known"
                      }}
                     for case in bounded_population),
                    (case["positive"] for case in bounded_population),
                    weights=(case["search_weight"] for case in bounded_population),
                    folds=(case["temporal_fold"] for case in bounded_population),
                    fpminer_unaries=semantic_seeds,positive_target="click",
                    semantic_predicates=semantic,context_predicates=context,
                    predicate_lineages={
                        feature:"+".join(sorted(self._pair_evidence_lineage(((feature,"value"),))))
                        for feature in (*semantic,*context)
                    },
                    config=ConditionalMiningConfig(
                        min_support=target_min_support,
                        fold_min_support=max(1.0,target_min_support/3.0),
                        max_depth=max(2,int(self.config["pair_conjunctions"])-1),
                        top_k=max(256,int(self.config["pair_max_rules"])*24),
                        min_usable_folds=2,
                        min_effect=float(self.config["pair_min_effect"]),
                        min_incremental_effect=0.0,
                        max_values_per_predicate=max_values,
                    ),
                )
                existing={(rule["premises"],rule["target"]) for rule in rules}
                for pattern in conditional_result.patterns:
                    premises=tuple((atom.predicate,atom.value)
                                   for atom in pattern.premises)
                    key=(premises,"click")
                    if key in existing:
                        continue
                    rules.append({
                        "premises":premises,"target":"click",
                        "support":pattern.counts.tp,
                        "strength":pattern.precision,"confidence":0.0,
                        "conditional_incremental_effect":pattern.incremental_effect,
                        "conditional_stable_incremental_effect":
                            pattern.stable_incremental_effect,
                        "conditional_incremental_wracc":pattern.incremental_wracc,
                        "conditional_robust_incremental_wracc":
                            pattern.robust_incremental_wracc,
                        "conditional_evidence_lineages":pattern.evidence_lineages,
                        "conditional_lineage_signature":pattern.lineage_signature,
                        "conditional_variant_signature":pattern.variant_signature,
                        "conditional_fpminer_seed_rule_ids":
                            pattern.fpminer_seed_rule_ids,
                        # Conditional semantic structures remain variants of
                        # the text evidence that seeded them; the invariant
                        # gate does not mint an independent vote.
                        "dependency_owner":"pair_text_semantic_top3_mean",
                        "target_weighted_support":pattern.weighted.support,
                        "target_weighted_contingency":pattern.weighted.as_dict(),
                        "target_count_contingency":pattern.counts.as_dict(),
                        "target_fold_statistics":[
                            fold.as_dict() for fold in pattern.fold_statistics
                        ],
                        "source":"recommendation/mining/conditional_llm_mining.py",
                    })
                    existing.add(key)
                target_search={
                    "kind":strategy,
                    "requires_real_fpminer_seed":True,
                    "semantic_predicates":list(semantic),
                    "context_predicates":list(context),
                    "fpminer_semantic_seeds":len(semantic_seeds),
                    "config":dict(conditional_result.config.__dict__),
                    "audit":conditional_result.audit.as_dict(),
                    "candidate_patterns":len(conditional_result.patterns),
                    "deeper_candidate_patterns":len(conditional_result.patterns),
                }
            else:
                target_search={
                    "kind":strategy,
                    "requires_real_fpminer_seed":True,
                    "semantic_predicates":list(semantic),
                    "context_predicates":list(context),
                    "fpminer_semantic_seeds":len(semantic_seeds),
                    "candidate_patterns":0,"deeper_candidate_patterns":0,
                    "reason":"no active semantic/context vocabulary or real fpMiner semantic seed",
                }
            if strategy=="conditional_llm_seed_only":
                semantic_seed_ids={seed.rule_id for seed in semantic_seeds}
                removed=[
                    rule for rule in rules
                    if rule.get("id") in semantic_seed_ids
                    and len(rule.get("premises",()))==1
                    and rule["premises"][0][0] in LLM_PAIR_PREDICATES
                ]
                removed_ids={id(rule) for rule in removed}
                rules=[rule for rule in rules if id(rule) not in removed_ids]
                target_search.update({
                    "semantic_seed_policy":"discovery_only",
                    "discovery_only_seed_rules":[{
                        "rule_id":rule["id"],
                        "premises":[list(item) for item in rule["premises"]],
                        "source":rule["source"],
                        "support":rule["support"],
                        "discovery_ctv":rule["discovery_ctv"],
                    } for rule in removed],
                    "discovery_only_seed_count":len(removed),
                    "candidate_conditional_children":sum(
                        rule.get("source","").endswith(
                            "conditional_llm_mining.py"
                        ) for rule in rules
                    ),
                    "backoff_policy":"non-LLM mined rules remain active",
                })
        elif strategy=="target_aware":
            target_result=mine_target_patterns(
                ({feature:case["attrs"][feature] for feature in active
                  if feature in case["attrs"]} for case in bounded_discovery),
                (case["positive"] for case in bounded_discovery),
                weights=(case["search_weight"] for case in bounded_discovery),
                folds=(case["temporal_fold"] for case in bounded_discovery),
                config=TargetMinerConfig(
                    min_support=target_min_support,
                    max_depth=max(1,int(self.config["pair_conjunctions"])-1),
                    top_k=max(256,int(self.config["pair_max_rules"])*24),
                    objective="positive",
                    min_wracc=0.0,
                    exhaustive=False,
                    fold_min_support=max(1.0,target_min_support/3.0),
                ),
            )
            existing={(rule["premises"],rule["target"]) for rule in rules}
            for pattern in target_result.patterns:
                if pattern.depth<2:
                    continue
                premises=tuple((atom.predicate,atom.value)
                               for atom in pattern.premises)
                key=(premises,"click")
                if key in existing:
                    continue
                rules.append({
                    "premises":premises,"target":"click",
                    "support":pattern.counts.tp,
                    "strength":pattern.precision or 0.0,"confidence":0.0,
                    "target_wracc":pattern.wracc,
                    "target_weighted_support":pattern.weighted.support,
                    "target_weighted_contingency":pattern.weighted.as_dict(),
                    "target_count_contingency":pattern.counts.as_dict(),
                    "target_fold_statistics":[
                        fold.as_dict() for fold in pattern.fold_statistics
                    ],
                    "source":"recommendation/mining/target_miner.py",
                })
                existing.add(key)
            target_search={
                "config":dict(target_result.config.__dict__),
                "audit":target_result.audit.as_dict(),
                "candidate_patterns":len(target_result.patterns),
                "deeper_candidate_patterns":sum(
                    pattern.depth>=2 for pattern in target_result.patterns
                ),
            }
        n=len(bounded_population); wins=sum(case["positive"] for case in bounded_population)
        base_rate=wins/n if n else 0.5
        impression_macro=self.config["pair_ctv_mode"]=="impression_macro"
        unavailable_evidence={"incomparable","unknown","left_known","right_known"}
        def rule_applicable(case,premises):
            """True only when every premise family is observed for this pair."""
            return all(
                case["attrs"].get(predicate) is not None
                and str(case["attrs"].get(predicate)).lower()
                    not in unavailable_evidence
                for predicate,_value in premises
            )
        calibrated=[]
        for rule in rules:
            matched={index for index,case in enumerate(bounded_population)
                     if all(case["attrs"].get(predicate)==value
                            for predicate,value in rule["premises"])}
            antecedent_total=len(matched)
            joint=sum(bounded_population[index]["positive"] for index in matched)
            outside=n-antecedent_total; outside_joint=wins-joint
            raw_pair_strength=(joint/antecedent_total
                               if antecedent_total else 0.0)
            strength=raw_pair_strength
            raw_selection_k=float(self.config["pair_rule_selection_k"])
            selection_count_confidence=(
                antecedent_total/(antecedent_total+raw_selection_k)
                if antecedent_total else 0.0
            )
            # PeTTa's tv_formulas.metta decodes every confidence with K=800.
            # Encode actual evidence using that same invariant at the boundary.
            count_confidence=(
                antecedent_total/(antecedent_total+CTV_EVIDENCE_K_DEFAULT)
                if antecedent_total else 0.0
            )
            # In an orientation-balanced pair workspace a directional premise
            # can match at most one of the two orientations.  Its comparable
            # coverage is therefore 2n_r/n.  Treat that portability/reliability
            # as host-side selection reliability: a rule that only decides a
            # small exceptional slice must not displace an equally accurate
            # rule that orders most production candidates. Coverage must not be
            # folded into the CTV confidence that PeTTa decodes as evidence.
            directional_predicates=[
                predicate for predicate,_value in rule["premises"]
                if predicate in predicate_comparability
            ]
            if directional_predicates:
                jointly_comparable=sum(
                    all(
                        str(case["attrs"].get(predicate,"")).lower()
                        in {"left","right"}
                        or str(case["attrs"].get(predicate,"")).lower().startswith(
                            ("left_q","right_q")
                        )
                        for predicate in directional_predicates
                    )
                    for case in bounded_population
                )
                activation_coverage=jointly_comparable/n if n else 0.0
            else:
                activation_coverage=(
                                 (min(1.0,2.0*antecedent_total/n) if n else 0.0))
            confidence=count_confidence
            selection_confidence=(
                selection_count_confidence*activation_coverage
            )
            negative_strength=outside_joint/outside if outside else base_rate
            negative_confidence=(outside/(outside+CTV_EVIDENCE_K_DEFAULT)
                                 if outside else 0.0)
            calibration=None
            petta_calibration=None
            calibrated_base_rate=base_rate
            is_conditional_child=rule.get("source","").endswith(
                "conditional_llm_mining.py"
            )
            is_scoped_categorical=rule.get("scoped_categorical_prior") is True
            use_effective_conditional=(
                self.config["pair_ctv_mode"]=="conditional_effective_backoff"
                and is_conditional_child
            )
            if (self.config["pair_ctv_mode"] in {
                    "impression_macro","raw_strength_effective_confidence"}
                    or use_effective_conditional or is_scoped_categorical):
                calibration=calibrate_ctv((
                    CTVObservation(
                        impression_id=case["impression"],
                        matched=index in matched,
                        target=bool(case["positive"]),
                        applicable=rule_applicable(case,rule["premises"]),
                        # ``population_cases`` is exhaustive even when the
                        # discovery pass sampled opponents.  The stored weight
                        # is label-independent and sums to one per impression.
                        weight=float(case["search_weight"]),
                    )
                    for index,case in enumerate(bounded_population)
                ), evidence_k=float(self.config["pair_rule_selection_k"]),
                   rule_kind="pair_preference")
                petta_calibration=reencode_ctv_confidence(
                    calibration,evidence_k=CTV_EVIDENCE_K_DEFAULT
                )
                if (self.config["pair_ctv_mode"]=="impression_macro"
                        or is_scoped_categorical):
                    strength=calibration.positive.strength
                    negative_strength=calibration.negative.strength
                    calibrated_base_rate=calibration.applicable_target_base_rate
                # This mode deliberately preserves the raw-pair conditional
                # rate used by the established scorer. Only its epistemic
                # sample unit changes from quadratic pair rows to
                # Kish-effective independent impression mass. Applicability
                # remains an audit dimension and is not multiplied into CTV
                # confidence a second time.
                selection_confidence=calibration.positive.confidence
                confidence=petta_calibration.positive.confidence
                if not use_effective_conditional:
                    negative_confidence=petta_calibration.negative.confidence
                    activation_coverage=calibration.activation.weighted_fraction
            effect=strength-calibrated_base_rate; specificity=len(rule["premises"])
            fold_effects=[]
            fold_supports=[]
            fold_effective_impressions=[]
            for fold in range(3):
                fold_population=[index for index,case in enumerate(bounded_population)
                                 if case["temporal_fold"]==fold]
                fold_matches=matched.intersection(fold_population)
                if not fold_population: continue
                if (self.config["pair_ctv_mode"]=="impression_macro"
                        or is_scoped_categorical):
                    fold_calibration=calibrate_ctv((
                        CTVObservation(
                            impression_id=bounded_population[index]["impression"],
                            matched=index in fold_matches,
                            target=bool(bounded_population[index]["positive"]),
                            applicable=rule_applicable(
                                bounded_population[index],rule["premises"]
                            ),
                            weight=float(
                                bounded_population[index]["search_weight"]
                            ),
                        )
                        for index in fold_population
                    ), evidence_k=float(self.config["pair_rule_selection_k"]),
                       rule_kind="pair_preference_fold")
                    fold_support=fold_calibration.positive.weighted_support
                    fold_effective=fold_calibration.positive.effective_impressions
                    macro_fold_min=(target_min_support/3.0
                                          if (impression_macro
                                              or is_scoped_categorical)
                                          else calibration_min_support/3.0)
                    if fold_support<max(1.0,macro_fold_min):
                        continue
                    fold_base=fold_calibration.applicable_target_base_rate
                    fold_strength=fold_calibration.positive.strength
                else:
                    if len(fold_matches)<max(4,calibration_min_support//3):
                        continue
                    fold_support=float(len(fold_matches))
                    fold_effective=fold_support
                    fold_base=(sum(bounded_population[index]["positive"] for index in fold_population)
                               /len(fold_population))
                    fold_strength=(sum(bounded_population[index]["positive"] for index in fold_matches)
                                   /len(fold_matches))
                fold_effects.append(fold_strength-fold_base)
                fold_supports.append(fold_support)
                fold_effective_impressions.append(fold_effective)
            required_folds=3 if is_scoped_categorical else 2
            stable_effect=(min(fold_effects)
                           if len(fold_effects)>=required_folds else -1.0)
            uses_impression_support=(impression_macro or is_scoped_categorical)
            calibrated_support=(
                calibration.positive.weighted_support
                if uses_impression_support and calibration is not None
                else antecedent_total
            )
            required_support=(target_min_support if uses_impression_support
                              else calibration_min_support)
            mined_support=rule["support"]
            ctv_calibration_audit=None
            selection_calibration_audit=None
            if calibration is not None and petta_calibration is not None:
                selection_calibration_audit=calibration.as_dict()
                selection_calibration_audit["confidence_role"]=(
                    "host_rule_selection_only_not_petta_stv"
                )
                ctv_calibration_audit=petta_calibration.as_dict()
                # Some experimental modes preserve raw conditional strengths
                # or a raw unmatched branch. Record the exact CTV compiled for
                # PeTTa, not merely the population object it was derived from.
                ctv_calibration_audit["positive"]["strength"]=strength
                ctv_calibration_audit["positive"]["confidence"]=confidence
                ctv_calibration_audit["negative"]["strength"]=negative_strength
                ctv_calibration_audit["negative"]["confidence"]=negative_confidence
                ctv_calibration_audit["confidence_role"]=(
                    "petta_stv_evidence_encoded_with_fixed_k800"
                )
            rule.update(
                strength=strength,confidence=confidence,
                selection_confidence=selection_confidence,
                raw_pair_strength=raw_pair_strength,
                count_confidence=count_confidence,
                selection_count_confidence=selection_count_confidence,
                activation_coverage=activation_coverage,support=joint,
                mined_support=mined_support,
                antecedent_support=antecedent_total,joint_support=joint,
                negative_strength=negative_strength,negative_confidence=negative_confidence,
                lift=(strength/calibrated_base_rate if calibrated_base_rate else 0.0),effect=effect,
                ctv_calibration=ctv_calibration_audit,
                selection_calibration=selection_calibration_audit,
                petta_evidence_k=CTV_EVIDENCE_K_DEFAULT,
                pair_rule_selection_k=float(
                    self.config["pair_rule_selection_k"]
                ),
                ctv_estimation_mode=(
                    "impression_macro_kish"
                    if (impression_macro or is_scoped_categorical) else
                    "raw_strength_kish_confidence"
                    if (self.config["pair_ctv_mode"]==
                        "raw_strength_effective_confidence"
                        or use_effective_conditional) else
                    "raw_oriented_pairs"
                ),
                confidence_basis=(
                    "petta_kish_effective_impressions_k800_scoped_categorical"
                    if is_scoped_categorical else
                    "petta_kish_effective_impressions_k800_conditional_backoff"
                    if use_effective_conditional else
                    "petta_kish_effective_impressions_k800"
                    if self.config["pair_ctv_mode"] in {
                        "impression_macro","raw_strength_effective_confidence"
                    } else "petta_raw_pair_count_k800"
                ),
                selection_confidence_basis=(
                    "kish_effective_impressions_with_pair_rule_selection_k"
                    if calibration is not None else
                    "raw_pair_count_with_pair_rule_selection_k_times_activation_coverage"
                ),
                calibration_base_rate=calibrated_base_rate,
                temporal_fold_effects=[round(value,8) for value in fold_effects],
                temporal_fold_weighted_supports=[round(value,8)
                                                 for value in fold_supports],
                temporal_fold_kish_effective_impressions=[round(value,8)
                                                          for value in fold_effective_impressions],
                required_temporal_folds=required_folds,
                calibrated_support=calibrated_support,
                ctv_support=calibrated_support,
                calibrated_support_unit=(
                    "equal_impression_weighted_activation_mass"
                    if uses_impression_support else "raw_oriented_pair_case"
                ),
                ctv_support_unit=(
                    "equal_impression_weighted_activation_mass"
                    if uses_impression_support else "raw_oriented_pair_case"
                ),
                required_calibrated_support=required_support,
                stable_effect=stable_effect,
                # Balanced orientations make 2p-1 an anti-symmetric learned
                # vote margin.  It is later counted only when PeTTaChainer
                # proves this mined rule for the grounded pair.
                vote_weight=max(0.0,2.0*strength-1.0),
                specificity=specificity,coverage=matched,
                quality=max(0.0,stable_effect)*selection_confidence
                        *math.log1p(calibrated_support)
                        *(1+0.2*(specificity-1)),
            )
            if (calibrated_support>=required_support and effect>0
                    and stable_effect>=float(self.config["pair_min_effect"])):
                calibrated.append(rule)
        calibrated.sort(key=lambda rule:(-rule["quality"],-rule["specificity"],
                                         -rule["antecedent_support"],rule["premises"]))
        selected=[]
        rule_cap=int(self.config["pair_max_rules"])
        def retain_candidates(candidates,limit=None,*,within_family=False):
            retained=0
            for rule in candidates:
                redundant=False
                for previous in selected:
                    if (within_family
                            and (previous["specificity"]==1)
                            != (rule["specificity"]==1)):
                        continue
                    union_cases=rule["coverage"]|previous["coverage"]
                    intersection_cases=rule["coverage"]&previous["coverage"]
                    if (impression_macro
                            or rule.get("scoped_categorical_prior") is True
                            or previous.get("scoped_categorical_prior") is True):
                        # Categorical priors are calibrated per impression;
                        # redundancy must use the same sample unit. Otherwise
                        # one large slate can erase a distinct portable rule.
                        union=math.fsum(
                            bounded_population[index]["search_weight"]
                            for index in union_cases
                        )
                        intersection=math.fsum(
                            bounded_population[index]["search_weight"]
                            for index in intersection_cases
                        )
                    else:
                        union=len(union_cases)
                        intersection=len(intersection_cases)
                    overlap=(intersection/union if union else 1.0)
                    if overlap>=0.98 and abs(rule["strength"]-previous["strength"])<0.02:
                        redundant=True; break
                if not redundant:
                    selected.append(rule); retained+=1
                if len(selected)>=rule_cap or (limit is not None and retained>=limit):
                    return True
            return False
        if weighted_search:
            # A shared cap must not let either half of the hybrid starve the
            # other. Reserve deterministic quotas for the real fpMiner unary
            # backoff and target-expanded structures, then backfill unused slots
            # by the common quality order. During the reserved pass, redundancy
            # is evaluated within each family so at least one deeper rule can be
            # exercised even when it closely specializes a unary rule.
            unary=[rule for rule in calibrated if rule["specificity"]==1]
            deeper=[rule for rule in calibrated if rule["specificity"]>1]
            if unary and deeper:
                if rule_cap>=2:
                    unary_quota=(rule_cap+1)//2
                    deeper_quota=rule_cap-unary_quota
                    retain_candidates(iter(unary),unary_quota,within_family=True)
                    retain_candidates(iter(deeper),deeper_quota,within_family=True)
                    if len(selected)<rule_cap:
                        retain_candidates(iter(calibrated))
                else:
                    # A one-rule artifact cannot contain both families; retain
                    # the auditable real-miner backoff instead of silently
                    # presenting one host-expanded rule as the whole hybrid.
                    retain_candidates(iter(unary))
            else:
                retain_candidates(iter(calibrated))
        else:
            retain_candidates(iter(calibrated))
        # Preserve useful specific variants, but make strongly nested evidence
        # dependent in PeTTa instead of confidence-inflating it. The residual
        # challenger keeps each conjunction as a separate evidence hyperedge
        # and compiles only its log-odds increment beyond selected parents.
        parents=list(range(len(selected)))
        def find(index):
            while parents[index]!=index:
                parents[index]=parents[parents[index]]; index=parents[index]
            return index
        def union(left,right):
            left_root,right_root=find(left),find(right)
            if left_root!=right_root: parents[right_root]=left_root
        roots={}
        variants=Counter()
        residual_audit=None
        if self.config["pair_dependency_mode"]=="residual_hypergraph":
            selected,residual_audit=self._select_positive_residual_hyperedges(selected)
            roots={rule_index:rule_index for rule_index in range(len(selected))}
        else:
            for left in range(len(selected)):
                for right in range(left+1,len(selected)):
                    # A conditional LLM rule is an incrementally validated
                    # variant of its real-miner semantic seed, not a new
                    # witness. Bind it to the declared text owner without
                    # allowing its invariant context gate to bridge unrelated
                    # structured dependencies through union-find.
                    if self._pair_rules_share_dependency(
                            selected[left],selected[right]):
                        union(left,right)
            for rule_index,rule in enumerate(selected):
                root=find(rule_index)
                cluster_index=roots.setdefault(root,len(roots)+1)
                dependency_id=f"pair_mined_cluster_{cluster_index}"
                variants[dependency_id]+=1
                rule.update(id=dependency_id,dependency_id=dependency_id,
                            variant_id=f"{dependency_id}_v{variants[dependency_id]}")
        if target_search is not None and strategy=="conditional_llm_seed_only":
            compiled_children=[
                rule for rule in selected
                if rule.get("source","").endswith("conditional_llm_mining.py")
            ]
            target_search.update({
                "compiled_conditional_children":len(compiled_children),
                "compiled_conditional_premises":[
                    [list(item) for item in rule["premises"]]
                    for rule in compiled_children
                ],
            })
        calibrated_categorical=[
            rule for rule in calibrated
            if rule.get("scoped_categorical_prior") is True
        ]
        selected_categorical=[
            rule for rule in selected
            if rule.get("scoped_categorical_prior") is True
        ]
        categorical_search.update({
            "calibration":"mandatory equal-impression CTV regardless of global pair_ctv_mode",
            "temporal_gate":"all three source-order folds",
            "support_gate":(
                "equal-impression weighted activation mass >= target_min_support; "
                "Kish effective impressions determine confidence only"
            ),
            "redundancy":"equal-impression weighted activation Jaccard",
            "calibrated_scoped_rules":len(calibrated_categorical),
            "selected_scoped_rules":len(selected_categorical),
            "selected_scoped_premises":[
                [list(item) for item in rule["premises"]]
                for rule in selected_categorical
            ],
            "dependency_policy":"dependent alternative inside existing evidence owner; never another vote",
            "category_symbol_labels":{
                symbol:self._pair_categorical_labels[symbol]
                for symbol in sorted({
                    value for rule in categorical_eligible
                    for predicate,value in rule["premises"]
                    if predicate in PAIR_CATEGORICAL_SIDE_PREDICATES
                })
                if symbol in self._pair_categorical_labels
            },
        })
        used_features={predicate for rule in selected for predicate,_value in rule["premises"]}
        for _family,(left_predicate,right_predicate) in (
                PAIR_CATEGORICAL_SIDE_FAMILIES.items()):
            if {left_predicate,right_predicate}.intersection(used_features):
                # A one-sided rule still needs both observed values to enforce
                # whole-family OOV/missing abstention at serving time.
                used_features.update((left_predicate,right_predicate))
        self._pair_feature_vocabulary={feature:values for feature,values in vocabulary.items()
                                       if feature in used_features}
        sources=[]
        # Correlated variants are alternatives, not independent observations
        # to revise into one conclusion.  Give each variant an isolated PeTTa
        # proof channel in proof-margin mode, while retaining its logical
        # dependency as a separate argument.  The scorer will still take at
        # most one inferred margin per dependency.
        isolate_variants=self.config["pair_aggregation"]=="proof_margin"
        for rule in selected:
            rule.setdefault("source","recommendation/miner/fpMiner.metta")
            terms=[f'({predicate.title()} $pair {json.dumps(value)})'
                   for predicate,value in rule["premises"]]
            premise=terms[0] if len(terms)==1 else f"(And {' '.join(terms)})"
            proof_strength=rule.get("proof_strength",rule["strength"])
            proof_confidence=rule.get("proof_confidence",rule["confidence"])
            negative_strength=(0.5 if "proof_strength" in rule
                               else rule["negative_strength"])
            negative_confidence=(0.0 if "proof_strength" in rule
                                 else rule["negative_confidence"])
            ctv=(f'(CTV (STV {proof_strength} {proof_confidence}) '
                 f'(STV {negative_strength} {negative_confidence}))')
            proof_channel=(rule["variant_id"] if isolate_variants
                           else rule["dependency_id"])
            rule["proof_channel_id"]=proof_channel
            dependency=json.dumps(rule["dependency_id"])
            channel=json.dumps(proof_channel)
            conclusion=(f'(MinedPairPreference $pair {dependency} {channel})'
                        if isolate_variants else
                        f'(MinedPairPreference $pair {dependency})')
            variant_source=(
                f'(: {rule["variant_id"]} (Implication {premise} '
                f'{conclusion}) {ctv})'
            )
            sources.append(variant_source)
            if isolate_variants:
                # This compiler-issued contract is checked before applying the
                # alpha-normalized proof optimization.  Hand-written or future
                # nonlocal rules therefore fail closed instead of being
                # silently treated as extensional channel templates.
                rule["proof_factorization"]={
                    "schema":"isolated_extensional_pair_channel_v1",
                    "case_variable":"$pair",
                    "premise_tv":[1.0,1.0],
                    "single_channel_producer":True,
                    "dependency_id":rule["dependency_id"],
                    "proof_channel_id":proof_channel,
                    "premises":[list(item) for item in rule["premises"]],
                    "rule_source_sha256":hashlib.sha256(
                        variant_source.encode("utf-8")
                    ).hexdigest(),
                }
            rule.pop("coverage",None)
        merge_ids=set()
        if selected:
            proof_roots=(
                sorted({(rule["dependency_id"],rule["proof_channel_id"])
                        for rule in selected})
                if isolate_variants else
                [(dependency_id,dependency_id) for dependency_id in sorted(
                    {rule["dependency_id"] for rule in selected}
                )]
            )
            for root_index,(dependency_id,proof_channel) in enumerate(proof_roots,1):
                dependency=json.dumps(dependency_id)
                channel=json.dumps(proof_channel)
                pair_signal=(f'(PairSignal $pair {dependency} {channel})'
                             if isolate_variants else
                             f'(PairSignal $pair {dependency})')
                mined_signal=(f'(MinedPairPreference $pair {dependency} {channel})'
                              if isolate_variants else
                              f'(MinedPairPreference $pair {dependency})')
                merge_id=f"pair_merge_rule_{root_index}"
                merge_ids.add(merge_id)
                decision_id=f"pair_decision_rule_{root_index}"
                merge_ids.add(decision_id)
                sources.extend((
                    f'(: {decision_id} (Implication '
                    f'{mined_signal} {pair_signal}) '
                    '(CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
                    # This is a one-way decision adapter, not a claim that
                    # every family is necessary for PairWin. Bayesian inversion
                    # would feed other families back through the shared target
                    # and invent evidence in an otherwise absent channel.
                    f'(: (no_inverse {merge_id}) (Implication {pair_signal} '
                    '(PairWin $pair)) (CTV (STV 1.0 1.0) (STV 0.0 1.0)))',
                ))
        self._pair_rule_sources=sources
        self._active_pair_rule_ids={
            identifier for rule in selected
            for identifier in (rule["dependency_id"],rule["variant_id"])
        }
        if selected: self._active_pair_rule_ids.update(merge_ids)
        self.pair_rules=selected
        self.last_pair_mining={
            "rules":len(selected),"cases":len(discovery_cases),"source_cases":n,
            "wins":wins,"losses":n-wins,
            "pair_target_semantics":{
                "training_unit":"one closed observed training impression",
                "eligible_impression":(
                    "contains at least one observed clicked and one observed "
                    "nonclicked candidate"
                ),
                "pair_construction":(
                    "every observed clicked-versus-nonclicked candidate pair; "
                    "both left/right orientations"
                ),
                "positive_target":(
                    "left candidate is the observed clicked candidate and right "
                    "candidate is the observed nonclicked candidate"
                ),
                "interpretation":(
                    "conditional within-impression pair-ordering target; not a "
                    "universal item-relevance claim"
                ),
            },
            "evidence_clusters":len(roots),
            "proof_channels":len(proof_roots) if selected else 0,
            "proof_channel_mode":(
                "variant_isolated" if isolate_variants else "shared_dependency"
            ),
            "proof_factorization":{
                "enabled":bool(selected and isolate_variants),
                "mode":"alpha_normalized_isolated_channels",
                "safety_contract":"isolated_extensional_pair_channel_v1",
                "maximum_templates":len(proof_roots) if selected else 0,
                "host_applicability":"exact categorical premise join",
                "reasoner_role":"two-hop channel proof and inferred STV",
            },
            "base_rate":round(base_rate,8),
            "min_support":(target_min_support if weighted_search
                           else fpminer_min_support),
            "min_support_unit":("equal_impression_mass_target_search_only"
                                if weighted_search
                                else "raw_oriented_pair_case"),
            "fpminer_min_support":fpminer_min_support,
            "fpminer_support_unit":"raw_oriented_pair_case",
            "target_min_support":(target_min_support
                                  if weighted_search else None),
            "target_support_unit":("equal_impression_mass"
                                   if weighted_search else None),
            "calibration_min_support":calibration_min_support,
            "calibration_min_support_unit":"raw_oriented_pair_case",
            "selected_ctv_min_support":(
                target_min_support if impression_macro else calibration_min_support
            ),
            "selected_ctv_min_support_unit":(
                "equal_impression_weighted_activation_mass"
                if impression_macro else "raw_oriented_pair_case"
            ),
            "selected_ctv_support_policy":{
                "ordinary_rules":(
                    "equal_impression_weighted_activation_mass"
                    if impression_macro else "raw_oriented_pair_case"
                ),
                "scoped_categorical_rules":(
                    "equal_impression_weighted_activation_mass"
                ),
            },
            "miner_calls":len(plan),
            "fpminer_query_calls":sum(
                item["miner_calls"] for item in fpminer_incremental
            ),
            "fpminer_incremental":fpminer_incremental,
            "workspace_mode":MINING_WORKSPACE_MODE,
            "workspace_sync":workspace_sync,
            "workspace_plans":len(workspace_plans),
            "retention":retention_audit,
            "full_structure_research":(
                strategy!="fixed_combinations" or any(
                    item["full_structure_research"]
                    for item in fpminer_incremental
                )
            ),
            # No held-out outcomes participate in this training-population
            # estimate.
            "full_population_ctv_estimation":True,
            "discovery_sample":{
                "purpose":"rule_structure_discovery_only",
                "oriented_pair_cases":len(bounded_discovery),
                "complete_impressions":len({
                    case["impression"] for case in bounded_discovery
                }),
                "negative_opponents_per_positive":int(
                    self.config["pair_negative_ratio"]
                ),
                "sampling_policy":(
                    "all negative opponents"
                    if int(self.config["pair_negative_ratio"])==0 else
                    "deterministic capped negative opponents per positive"
                ),
                "held_out_evaluation_labels_used":False,
                "discovery_statistics_reused_for_final_ctv_estimation":False,
            },
            "ctv_estimation_population":{
                "purpose":"CTV estimation, temporal gates and rule selection",
                "oriented_pair_cases":len(bounded_population),
                "complete_impressions":len({
                    case["impression"] for case in bounded_population
                }),
                "opponent_sampling":"none",
                "source":(
                    "all clicked-versus-nonclicked oriented pairs from complete "
                    "retained closed training impressions"
                ),
                "held_out_evaluation_labels_used":False,
                "ordinary_rule_weighting":(
                    "one total mass per impression with Kish effective support"
                    if impression_macro else "raw oriented-pair rows"
                ),
                "scoped_categorical_weighting":(
                    "one total mass per impression with Kish effective support"
                ),
            },
            "miner_strategy":strategy,
            "ctv_evidence_k":float(self.config["ctv_evidence_k"]),
            "pair_ctv_mode":self.config["pair_ctv_mode"],
            "petta_ctv_evidence_k":CTV_EVIDENCE_K_DEFAULT,
            "pair_rule_selection_k":float(
                self.config["pair_rule_selection_k"]
            ),
            "pair_ctv_evidence_k":float(self.config["pair_ctv_evidence_k"]),
            "pair_ctv_evidence_k_status":(
                "deprecated alias of pair_rule_selection_k; never used to "
                "encode PeTTa STV confidence"
            ),
            "confidence_contract":(
                "PeTTa STV confidence = evidence/(evidence+800); configurable "
                "selection K affects host-side rule selection only"
            ),
            "conditional_effective_backoff":{
                "enabled":self.config["pair_ctv_mode"]==
                    "conditional_effective_backoff",
                "scope":(
                    "conditional_llm_mining.py children only; scoped categorical "
                    "priors always use mandatory impression-macro calibration"
                ),
                "base_rule_policy":"byte-equivalent raw-pair CTV",
                "fusion":"replace weaker variant within shared text dependency",
                "independent_vote_added":False,
            },
            "pair_dependency_mode":self.config["pair_dependency_mode"],
            "residual_hypergraph":residual_audit,
            "target_search":target_search,
            "categorical_search":categorical_search,
            "search_weight_unit":("equal_impression_mass_target_search_only"
                                  if weighted_search
                                  else "raw_oriented_pair_case"),
            "search_weight_total":round(
                discovery_search_weight if weighted_search
                else len(discovery_cases),8
            ),
            "feature_profile":self.config["pair_feature_profile"],
            "rules_by_premises":dict(sorted(Counter(rule["specificity"] for rule in selected).items())),
            "feature_vocabulary":{feature:len(values) for feature,values
                                  in self._pair_feature_vocabulary.items()},
            "numeric_evidence":{
                source:encoder.to_metadata()
                for source,encoder in self._numeric_pair_encoders.items()
            },
            "numeric_evidence_fitting":{
                "split":"training_only",
                "pair_population":"all_unordered_same_impression_candidates",
                "target_labels_used":False,
                "weight_unit":"equal_impression_mass_per_feature",
                "delta_direction":"restored_only_after_magnitude_bin_transform",
            },
            "seconds":round(time.perf_counter()-started,3),
        }
        return self.last_pair_mining

    @staticmethod
    def _serving_model_digest(payload):
        encoded=json.dumps(
            payload,sort_keys=True,separators=(",",":"),ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def serving_model(self):
        """Return a content-addressed immutable model for scorer startup.

        The artifact contains learned symbolic vocabularies, calibrated rules,
        numeric encoders and the exact MeTTa compiler output. It intentionally
        excludes user/session state, proof caches and training events so every
        serving replica starts clean from one auditable model version.
        """
        payload={
            "schema":SERVING_MODEL_SCHEMA,
            "source_rule_version":int(self.version),
            "symbolic_only":bool(self.symbolic_only),
            "config":copy.deepcopy(self.config),
            "feature_vocabulary":{
                predicate:sorted(values)
                for predicate,values in self._feature_vocabulary.items()
            },
            "pair_feature_vocabulary":{
                predicate:sorted(values)
                for predicate,values in self._pair_feature_vocabulary.items()
            },
            "pair_categorical_labels":dict(
                sorted(self._pair_categorical_labels.items())
            ),
            "numeric_pair_encoders":{
                source:encoder.to_metadata()
                for source,encoder in self._numeric_pair_encoders.items()
            },
            "point_rules":copy.deepcopy(self.mined_rules),
            "pair_rules":copy.deepcopy(self.pair_rules),
            "compiled_point_rules":list(self._point_rule_sources),
            "compiled_point_channels":list(self._point_channel_sources),
            "compiled_pair_rules":list(self._pair_rule_sources),
            "active_point_rule_ids":sorted(self._active_rule_ids),
            "active_pair_rule_ids":sorted(self._active_pair_rule_ids),
            "click_base_rate":float(self._click_base_rate),
            "tie_break_stats":copy.deepcopy(self._tie_break_stats),
        }
        return {
            **payload,
            "model_sha256":self._serving_model_digest(payload),
        }

    def _load_serving_model(self,model):
        """Validate and install a frozen model without rerunning fpMiner."""
        if not isinstance(model,dict) or model.get("schema")!=SERVING_MODEL_SCHEMA:
            raise ValueError("unsupported serving model schema")
        payload={key:value for key,value in model.items()
                 if key!="model_sha256"}
        expected=self._serving_model_digest(payload)
        if not hmac.compare_digest(
                str(model.get("model_sha256","")),expected):
            raise ValueError("serving model digest does not match its content")
        if bool(model.get("symbolic_only"))!=self.symbolic_only:
            raise ValueError("serving model evidence mode does not match scorer")

        def vocabulary(field):
            raw=model.get(field)
            if not isinstance(raw,dict):
                raise ValueError(f"serving model has no valid {field}")
            parsed={}
            for predicate,values in raw.items():
                if (not isinstance(predicate,str)
                        or not isinstance(values,list)
                        or any(not isinstance(value,str) for value in values)):
                    raise ValueError(f"serving model has invalid {field}")
                parsed[predicate]=set(values)
            return parsed

        point_rules=copy.deepcopy(model.get("point_rules"))
        pair_rules=copy.deepcopy(model.get("pair_rules"))
        if not isinstance(point_rules,list) or not isinstance(pair_rules,list):
            raise ValueError("serving model rules must be lists")
        for rule in (*point_rules,*pair_rules):
            if not isinstance(rule,dict) or not isinstance(
                    rule.get("premises"),list):
                raise ValueError("serving model contains an invalid rule")
            rule["premises"]=[tuple(item) for item in rule["premises"]]
        source_fields=(
            "compiled_point_rules","compiled_point_channels",
            "compiled_pair_rules",
        )
        sources={field:model.get(field) for field in source_fields}
        if any(not isinstance(value,list)
               or any(not isinstance(item,str) for item in value)
               for value in sources.values()):
            raise ValueError("serving model compiled rules must be string lists")

        self._feature_vocabulary=vocabulary("feature_vocabulary")
        self._pair_feature_vocabulary=vocabulary("pair_feature_vocabulary")
        labels=model.get("pair_categorical_labels")
        if (not isinstance(labels,dict)
                or any(not isinstance(key,str) or not isinstance(value,str)
                       for key,value in labels.items())):
            raise ValueError("serving model categorical labels are invalid")
        self._pair_categorical_labels=dict(labels)
        encoders=model.get("numeric_pair_encoders")
        if not isinstance(encoders,dict):
            raise ValueError("serving model numeric encoders are invalid")
        self._numeric_pair_encoders={
            source:QuantileNumericEvidence.from_metadata(metadata)
            for source,metadata in encoders.items()
        }
        self.mined_rules=point_rules
        self.mined_output=[]
        self.pair_rules=pair_rules
        self._point_rule_sources=list(sources["compiled_point_rules"])
        self._point_channel_sources=list(sources["compiled_point_channels"])
        self._pair_rule_sources=list(sources["compiled_pair_rules"])
        self._active_rule_ids=set(model.get("active_point_rule_ids") or ())
        self._active_pair_rule_ids=set(
            model.get("active_pair_rule_ids") or ()
        )
        self._click_base_rate=float(model.get("click_base_rate"))
        if not 0.0<=self._click_base_rate<=1.0:
            raise ValueError("serving model click base rate is invalid")
        tie_break_stats=model.get("tie_break_stats")
        if not isinstance(tie_break_stats,dict):
            raise ValueError("serving model tie-break statistics are invalid")
        self._tie_break_stats=copy.deepcopy(tie_break_stats)

        # Fail before accepting traffic if compiler metadata and rule objects
        # disagree. PeTTa then receives exactly the validated source snapshot.
        self._point_channel_topology()
        self._compile_pair_rule_matcher()
        self.engine.replace([
            *self._point_rule_sources,*self._point_channel_sources,
            *self._pair_rule_sources,*LIVE_NEGATIVE_RULE_SOURCES,
            *(RELATIONAL_STRUCTURAL_RULES
              if self.config.get("relational_evidence_mode")=="chained"
              else ()),
        ])
        self.version=max(1,int(model.get("source_rule_version",1)))
        self.pending_events=0
        self._last_mined_event_sequence=self._event_sequence
        self.last_mined_at=time.time()
        self.last_pair_mining={
            "execution":"serving_model_load",
            "rules":len(self.pair_rules),
            "model_sha256":expected,
        }
        self.last_mining={
            "execution":"serving_model_load",
            "rules":len(self.mined_rules),
            "pair_rules":len(self.pair_rules),
            "model_sha256":expected,
            "version":self.version,
        }
        self._prewarm_serving_channels()
        self._startup_mode="serving_model"
        self._serving_model_sha256=expected

    def _prewarm_serving_channels(self):
        """Resolve model-invariant proof templates before readiness.

        Isolated point/pair channel proofs depend on their compiled rule and
        certain premise facts, not on a user's case identifier. Loading one
        alpha-normalized template per rule removes the first request's query
        penalty. Case-level caches are then cleared so no synthetic warm-up
        identity can be served as user evidence.
        """
        started=time.perf_counter()
        point_specs=[]
        for index,rule in enumerate(self.mined_rules):
            attrs=dict(rule["premises"])
            point_specs.append((
                f"warm_point_article_{index}",f"warm_point_case_{index}",
                attrs,attrs,
            ))
        if point_specs:
            self._proofs_for_specs(point_specs)

        pair_specs=[]
        if (self.config.get("ranking_mode")=="pairwise"
                and self.pair_rules):
            pair_specs=[
                (f"warm_pair_case_{index}",dict(rule["premises"]))
                for index,rule in enumerate(self.pair_rules)
            ]
            self._ensure_pair_specs(pair_specs)
            self._proofs_for_pair_specs(pair_specs)

        audit={
            "status":"complete",
            "seconds":round(time.perf_counter()-started,6),
            "point_channels":len(self._point_channel_proof_cache),
            "pair_channels":len(self._pair_channel_proof_cache),
            "point_reasoner_query_calls":self._point_query_calls,
            "pair_reasoner_query_calls":self._pair_query_calls,
        }
        self._proof_cache.clear()
        self._pair_proof_cache.clear()
        self._pair_case_attrs.clear()
        self._pair_margin_cache.clear()
        self._point_query_calls=0
        self._point_query_roots=0
        self._point_pruned_query_roots=0
        self._point_channel_activations=0
        self._point_reused_channel_activations=0
        self._pair_query_calls=0
        self._pair_query_roots=0
        self._pair_pruned_query_roots=0
        self._pair_channel_activations=0
        self._pair_reused_channel_activations=0
        self._startup_prewarm=audit

    def _rebuild_tie_break_stats(self,indexed_events=None):
        """Build safe, training-only priors used only for exact proof ties.

        The symbolic score remains the decision signal.  MIND's bounded
        predicates intentionally collapse many articles to the same proof, so
        a small empirical editorial prior makes the order of an exact tie
        useful without allowing metadata to override a proof with a different
        score.  Beta(1, 9) smoothing matches the dataset CTR bucket prior.
        """
        totals={"topic":Counter(),"format":Counter(),"subcategory":Counter()}
        positives={"topic":Counter(),"format":Counter(),"subcategory":Counter()}
        indexed=(list(indexed_events) if indexed_events is not None else
                 list(enumerate(self.data.get("events",[]))))
        for _event_index,event in indexed:
            try:
                article=self.article(event["article"])
            except (KeyError,ValueError):
                continue
            for key in totals:
                value=str(article.get(key,"unknown"))
                totals[key][value]+=1
                positives[key][value]+=event.get("action") in POSITIVE
        self._tie_break_stats={
            key:{value:(positives[key][value]+1)/(count+10.0)
                 for value,count in totals[key].items() if count}
            for key in totals
        }

    def _tie_break(self, article):
        return {
            "topic_prior":self._tie_break_stats.get("topic",{}).get(
                str(article.get("topic",article.get("category","unknown"))),0.0),
            "format_prior":self._tie_break_stats.get("format",{}).get(
                str(article.get("format","unknown")),0.0),
            "subcategory_prior":self._tie_break_stats.get("subcategory",{}).get(
                str(article.get("subcategory","unknown")),0.0),
        }

    @staticmethod
    def _proof_ranking_signature(row):
        """Ranking evidence derived exclusively from PeTTa-backed signals."""
        stv=row.get("stv",{})
        return (
            float(row.get("ranking_score",row.get("score",0.0))),
            float(row.get("pairwise_score",0.0) or 0.0),
            float(row.get("score",0.0)),
            float(stv.get("strength",0.0)),
            float(stv.get("confidence",0.0)),
        )

    @staticmethod
    def _ranking_signature(row):
        """Comparable signal used by serving and replay AUC.

        The first three components are proof-derived. The final three are
        consulted only when those proof values are equal, matching the sort
        order in :meth:`_rank` without turning a calibrated score into noisy
        pseudo-precision.
        """
        tie=row.get("tie_break",{})
        stv=row.get("stv",{})
        return (*Lab._proof_ranking_signature(row),
            float(tie.get("topic_prior",0.0)),
            float(tie.get("format_prior",0.0)),
            float(tie.get("subcategory_prior",0.0)),
        )

    @staticmethod
    def _pointwise_signature(row):
        tie=row.get("tie_break",{}); stv=row.get("stv",{})
        return (float(row.get("score",0.0)),float(stv.get("strength",0.0)),
                float(stv.get("confidence",0.0)),float(tie.get("topic_prior",0.0)),
                float(tie.get("format_prior",0.0)),
                float(tie.get("subcategory_prior",0.0)))

    def _capture_background_mining(self):
        """Capture one causal training snapshot without copying static corpora."""
        with self.lock:
            if self._closed:
                raise RuntimeError("recommendation Lab is closed")
            retained_relational_proofs=self._retained_live_relational_proofs()
            snapshot_data=dict(self.data)
            # These are the only source structures mutated by live feedback.
            # Articles, vectors, adapters and frozen workspaces are immutable and
            # intentionally shared to keep the short capture phase bounded.
            snapshot_data["events"]=copy.deepcopy(self.data.get("events",[]))
            snapshot_data["users"]=copy.deepcopy(self.data.get("users",{}))
            snapshot_config=self.config.copy()
            sequence=self._event_sequence; base_version=self.version

            staged=copy.copy(self)
            staged.instance_id=f"{self.instance_id}-staged-{sequence}"
            staged.lock=threading.RLock()
            staged.data=snapshot_data
            staged.config=snapshot_config
            staged.engine=IsolatedPeTTaChainer()
            # The long-lived PeTTa runtime (and, when enabled, its persistent
            # workspace cache) remains shared. MINER_LOCK serializes every use.
            staged.petta=self.petta
            staged._closed=False; staged._background_mining=None
            staged._staged_background_build=True
            staged._event_sequence=sequence
            staged._last_mined_event_sequence=self._last_mined_event_sequence
            staged.pending_events=0
            staged.runs=[]; staged.feed_cache=OrderedDict()
            staged._feed_rank_cache_hits=0; staged._feed_rank_cache_misses=0
            staged._feed_sessions={}
            staged._online_events=[]
            staged._popularity=Counter(
                event["article"] for event in snapshot_data["events"]
                if event.get("action") in POSITIVE
            )
            staged._tie_break_stats={}
            staged._loaded_candidates=set(); staged._proof_cache={}
            staged._loaded_point_channels=set()
            staged._point_channel_proof_cache={}
            staged._point_channel_templates={}
            staged._point_query_calls=0; staged._point_query_roots=0
            staged._point_pruned_query_roots=0
            staged._point_channel_activations=0
            staged._point_reused_channel_activations=0
            staged._last_point_completeness={}
            staged._last_point_cache_stats={}
            staged._loaded_pairs=set(); staged._pair_proof_cache={}
            staged._loaded_pair_channels=set()
            staged._pair_channel_proof_cache={}
            staged._pair_channel_templates={}; staged._pair_proof_origins={}
            staged._pair_case_attrs={}; staged._pair_margin_cache={}
            staged._pair_query_calls=0; staged._pair_query_roots=0
            staged._pair_pruned_query_roots=0
            staged._pair_channel_activations=0
            staged._pair_reused_channel_activations=0
            staged._relational_feature_cache={}
            staged._loaded_relational_statements=set()
            # Event proof references are immutable training provenance. Keep
            # their records in the staged snapshot even though derived-feature
            # and worker caches always start cold.
            staged._live_relational_proof_ledger=retained_relational_proofs
            staged._candidate_relational_proof_refs={}
            staged._relational_query_calls=0
            staged._relational_query_roots=0
            return MiningSnapshot(sequence,base_version,staged)

    def _build_background_mining(self,snapshot):
        """Fully rebuild rules on a staged object while the old scorer serves."""
        staged=snapshot.payload
        try:
            staged.mine()
            return staged
        except BaseException:
            staged.engine.close()
            staged.engine=None
            raise

    @staticmethod
    def _dispose_background_mining(staged):
        engine=getattr(staged,"engine",None)
        if engine is not None:
            engine.close()
            staged.engine=None

    def _promote_background_mining(self,snapshot,staged):
        """Adopt one coherent staged rule/worker snapshot if it is still current."""
        old_engine=None
        with self.lock:
            if (self._closed or self.version!=snapshot.base_version
                    or self.config!=staged.config):
                return {"promoted":False,"reason":"stale_source"}
            if staged.engine is None or staged.engine.pid is None:
                raise RuntimeError("staged PeTTaChainer worker is not ready")
            expected_version=snapshot.base_version+1
            if staged.version!=expected_version:
                raise RuntimeError("staged mining version is inconsistent")
            # The build may overlap newly served cards or accepted feedback.
            # Preserve only proof records that remain reachable from the
            # current event/served-context ledgers before retiring old caches.
            retained_relational_proofs=self._retained_live_relational_proofs(
                getattr(staged,"_live_relational_proof_ledger",{})
            )

            old_engine=self.engine
            self.engine=staged.engine; staged.engine=None
            for field in BACKGROUND_MODEL_FIELDS:
                setattr(self,field,getattr(staged,field))
            self.version=expected_version
            self._last_mined_event_sequence=snapshot.event_sequence
            self.pending_events=max(
                0,self._event_sequence-self._last_mined_event_sequence
            )
            self.last_mined_at=time.time()
            workspace_prune={
                **self._prune_mining_workspace_cache(),"deferred":False,
            }
            self.last_mining={
                **(staged.last_mining or {}),
                "version":self.version,
                "execution":"background",
                "build_mode":BACKGROUND_MINING_BUILD_MODE,
                "event_sequence":snapshot.event_sequence,
                "workspace_prune":workspace_prune,
            }
            self._loaded_candidates.clear(); self._loaded_pairs.clear()
            self._loaded_point_channels.clear()
            self._point_channel_proof_cache.clear()
            self._point_channel_templates.clear()
            self._point_query_calls=0; self._point_query_roots=0
            self._point_pruned_query_roots=0
            self._point_channel_activations=0
            self._point_reused_channel_activations=0
            self._last_point_completeness={}
            self._last_point_cache_stats={}
            self._loaded_pair_channels.clear()
            self._proof_cache.clear(); self._pair_proof_cache.clear()
            self._pair_channel_proof_cache.clear()
            self._pair_channel_templates.clear(); self._pair_case_attrs.clear()
            self._pair_margin_cache.clear(); self._pair_proof_origins.clear()
            self._relational_feature_cache.clear()
            self._loaded_relational_statements.clear()
            self._live_relational_proof_ledger=retained_relational_proofs
            self._candidate_relational_proof_refs.clear()
            self._relational_query_calls=0
            self._relational_query_roots=0
            # Preserve the bounded served-impression ledger so feedback from a
            # card rendered immediately before promotion remains valid. Its
            # unserved queue is upgraded through the new scorer on that event;
            # an ordinary next-page request receives normal version-reset
            # semantics instead.
            self.feed_cache.clear()
            outcome={
                "promoted":True,"version":self.version,
                "event_sequence":snapshot.event_sequence,
                "pending_events":self.pending_events,
            }
        # No request can still hold the previous engine after the Lab lock was
        # acquired above. Retiring it outside that lock keeps new feeds available.
        try:
            old_engine.close()
        except BaseException:
            pass
        return outcome

    def _background_remine_due(self):
        with self.lock:
            return (not self._closed
                    and self.pending_events>=int(self.config["mine_interval"]))

    def close(self):
        """Cancel future promotions and release the active proof worker."""
        with self.lock:
            self._closed=True
            coordinator=self._background_mining
            engine=self.engine
            candidate_executor=getattr(
                self,"_candidate_feature_executor",None
            )
            self._candidate_feature_executor=None
        def release_workspaces():
            if coordinator is not None:
                coordinator.wait()
            try:
                with MINER_LOCK:
                    self._mining_workspaces.prune()
                    self._incremental_fpminer.prune()
            except BaseException:
                # Process shutdown/dataset replacement must still release the
                # scorer. A dirty cache cannot be reused after close.
                pass
        if coordinator is not None:
            stopped=coordinator.close(wait=False)
        else:
            stopped=True
        if stopped:
            release_workspaces()
        else:
            threading.Thread(
                target=release_workspaces,
                name="recommendation-mining-workspace-cleanup",
                daemon=True,
            ).start()
        engine.close()
        if candidate_executor is not None:
            candidate_executor.shutdown(wait=True,cancel_futures=True)

    def mine(self):
        if getattr(self,"serving_only",False):
            raise ValueError(
                "serving-only scorers cannot mine; publish a new frozen model"
            )
        # Both PeTTa miner instances share the named recommendation scratch
        # space. Keep a metadata snapshot as well: if mining or clean-worker
        # compilation fails, the still-live old scorer and its model metadata
        # must remain one coherent version.
        with MINER_LOCK:
            snapshot={
                "feature_vocabulary":self._feature_vocabulary,
                "pair_feature_vocabulary":self._pair_feature_vocabulary,
                "pair_categorical_labels":self._pair_categorical_labels,
                "numeric_pair_encoders":self._numeric_pair_encoders,
                "mined_rules":self.mined_rules,
                "mined_output":self.mined_output,
                "point_rule_sources":self._point_rule_sources,
                "point_channel_sources":self._point_channel_sources,
                "pair_rules":self.pair_rules,
                "pair_rule_sources":self._pair_rule_sources,
                "active_rule_ids":self._active_rule_ids,
                "active_pair_rule_ids":self._active_pair_rule_ids,
                "active_mining_workspace_plans":self._active_mining_workspace_plans,
                "last_pair_mining":self.last_pair_mining,
                "click_base_rate":self._click_base_rate,
                "tie_break_stats":self._tie_break_stats,
            }
            try:
                return self._mine_once()
            except BaseException:
                self._feature_vocabulary=snapshot["feature_vocabulary"]
                self._pair_feature_vocabulary=snapshot["pair_feature_vocabulary"]
                self._pair_categorical_labels=snapshot["pair_categorical_labels"]
                self._numeric_pair_encoders=snapshot["numeric_pair_encoders"]
                self.mined_rules=snapshot["mined_rules"]
                self.mined_output=snapshot["mined_output"]
                self._point_rule_sources=snapshot["point_rule_sources"]
                self._point_channel_sources=snapshot["point_channel_sources"]
                self.pair_rules=snapshot["pair_rules"]
                self._pair_rule_sources=snapshot["pair_rule_sources"]
                self._active_rule_ids=snapshot["active_rule_ids"]
                self._active_pair_rule_ids=snapshot["active_pair_rule_ids"]
                self._active_mining_workspace_plans=snapshot[
                    "active_mining_workspace_plans"
                ]
                self.last_pair_mining=snapshot["last_pair_mining"]
                self._click_base_rate=snapshot["click_base_rate"]
                self._tie_break_stats=snapshot["tie_break_stats"]
                raise

    def _mine_once(self):
        started=time.perf_counter()
        strategy=self.config["miner_strategy"]
        retention=self._retained_mining_population()
        indexed_events=retention.records
        retention_audit=retention.audit.as_dict()
        self._rebuild_tie_break_stats(indexed_events)
        training_cases=self._mining_event_cases(indexed_events)
        if not training_cases: raise RuntimeError("negative sampling left no mining cases")
        training_events=[event for _case_id,event in training_cases]
        raw_rows=[(case_id,event,self.event_features(event))
                  for case_id,event in training_cases]
        selected_profile=FEATURE_PROFILES[self.config["feature_profile"]]
        value_counts={feature:Counter(attrs.get(feature)
                                      for _case_id,_event,attrs in raw_rows
                                      if feature in attrs)
                      for feature in selected_profile}
        max_values=max(2,int(self.config["max_feature_values"]))
        vocabulary={}
        for feature,counts in value_counts.items():
            ranked=[value for value,count in counts.most_common()
                    if value is not None and count>=int(self.config["min_support"])]
            if not ranked: continue
            keep=set(ranked[:max_values])
            if any(value not in keep for value in counts): keep.add("other")
            vocabulary[feature]=keep
        self._feature_vocabulary=vocabulary
        feature_rows=[(case_id,event,self._bounded_features(attrs))
                      for case_id,event,attrs in raw_rows]
        workspace_sync=[]; workspace_plans=set()
        def sync_space(selected_features,depth):
            workspace_cases={}
            for case,event,attrs in feature_rows:
                facts=[]
                outcome="click" if event["action"] in POSITIVE else "skip"
                for predicate in selected_features:
                    if predicate in attrs:
                        facts.append(
                            f'({predicate} {case} {json.dumps(attrs[predicate])})'
                        )
                facts.append(f'(engagement {case} {json.dumps(outcome)})')
                workspace_cases[case]=tuple(facts)
            plan_key=self._mining_workspace_plan(
                "point",selected_features,depth,"sampled_discovery"
            )
            synced=self._mining_workspaces.sync(plan_key,workspace_cases)
            workspace_plans.add(plan_key); workspace_sync.append(synced.as_dict())
            return synced,plan_key,workspace_cases

        active_features=[feature for feature in selected_profile if feature in vocabulary]
        plan=[(tuple(active_features),2)] if active_features else []
        if strategy=="fixed_combinations" and int(self.config["conjunctions"])>=3:
            plan.extend((pair,3) for pair in INTERACTION_PAIRS if all(feature in vocabulary for feature in pair))
        if strategy=="fixed_combinations" and int(self.config["conjunctions"])>=4:
            plan.extend((triple,4) for triple in INTERACTION_TRIPLES if all(feature in vocabulary for feature in triple))
        raw=[]; fpminer_incremental=[]
        for selected_features,depth in plan:
            synced,plan_key,workspace_cases=sync_space(selected_features,depth)
            mined=self._mine_incremental_workspace(
                synced,plan=plan_key,cases=workspace_cases,
                features=selected_features,depth=depth,
                min_support=int(self.config["min_support"]),
                retention_units=retention.unit_ids,
                retention_audit=retention_audit,workspace_kind="point",
            )
            raw.extend(mined.output)
            fpminer_incremental.append(mined.audit.as_dict())
        rules=parse_rules([str(v) for v in raw])
        target_search=None
        if strategy=="target_aware":
            target_result=mine_target_patterns(
                ({feature:attrs[feature] for feature in active_features
                  if feature in attrs}
                 for _case_id,_event,attrs in feature_rows),
                (event["action"] in POSITIVE
                 for _case_id,event,_attrs in feature_rows),
                config=TargetMinerConfig(
                    min_support=int(self.config["min_support"]),
                    max_depth=max(1,int(self.config["conjunctions"])-1),
                    top_k=max(256,int(self.config["max_rules"])*24),
                    objective="absolute",
                    min_wracc=0.0,
                    exhaustive=False,
                ),
            )
            existing={(rule["premises"],rule["target"]) for rule in rules}
            for pattern in target_result.patterns:
                if pattern.depth<2:
                    continue
                premises=tuple((atom.predicate,atom.value)
                               for atom in pattern.premises)
                key=(premises,"click")
                if key in existing:
                    continue
                rules.append({
                    "premises":premises,"target":"click",
                    "support":pattern.counts.tp,
                    "strength":pattern.precision or 0.0,"confidence":0.0,
                    "target_wracc":pattern.wracc,
                    "target_weighted_support":pattern.weighted.support,
                    "target_weighted_contingency":pattern.weighted.as_dict(),
                    "target_count_contingency":pattern.counts.as_dict(),
                    "target_fold_statistics":[
                        fold.as_dict() for fold in pattern.fold_statistics
                    ],
                    "source":"recommendation/mining/target_miner.py",
                })
                existing.add(key)
            target_search={
                "config":dict(target_result.config.__dict__),
                "audit":target_result.audit.as_dict(),
                "candidate_patterns":len(target_result.patterns),
                "deeper_candidate_patterns":sum(
                    pattern.depth>=2 for pattern in target_result.patterns
                ),
            }
        if not rules: raise RuntimeError("MeTTa miner produced no usable rules")
        self.mined_output=[str(v) for v in raw]
        # Negative sampling is a search-time optimization only.  Re-estimate
        # every mined rule on the complete event population before compiling
        # its CTV, otherwise the sampled click prevalence would bias both the
        # rule strength and PeTTaChainer ranking.
        population_rows=[(event,self._bounded_features(self.event_features(event)))
                         for _event_index,event in indexed_events]
        n=len(population_rows)
        target_total=sum(event["action"] in POSITIVE for event,_attrs in population_rows)
        sample_target_total=sum(event["action"] in POSITIVE for event in training_events)
        base_rate=target_total/n if n else 0.0
        for rule in rules:
            antecedent_total=joint=0
            for event,attrs in population_rows:
                matches=all(attrs.get(p)==v for p,v in rule["premises"])
                antecedent_total+=matches; joint+=matches and event["action"] in POSITIVE
            outside=n-antecedent_total; outside_joint=target_total-joint
            strength=joint/antecedent_total if antecedent_total else 0.0
            evidence_k=float(self.config["ctv_evidence_k"])
            confidence=(antecedent_total/(antecedent_total+evidence_k)
                        if antecedent_total else 0.0)
            negative_strength=outside_joint/outside if outside else 0.0
            negative_confidence=(outside/(outside+evidence_k)
                                 if outside else 0.0)
            specificity=len(rule["premises"]); mined_support=rule["support"]
            rule.update(strength=strength,confidence=confidence,support=joint,
                        mined_support=mined_support,
                        negative_strength=negative_strength,negative_confidence=negative_confidence,
                        antecedent_support=antecedent_total,joint_support=joint,
                        lift=(strength/base_rate if base_rate else 0.0),specificity=specificity,
                        # Both positive- and negative-lift click rules carry
                        # ranking information.  Their posterior is later
                        # shrunk toward this same empirical base rate.
                        quality=confidence*abs(strength-base_rate)
                                *(1+0.25*(specificity-1))*math.log1p(antecedent_total))
        def rule_order(rule):
            if self.config["rule_rank"]=="quality":
                return (-rule["quality"],-rule["specificity"],-rule["support"],rule["premises"])
            return (-rule["support"],-rule["strength"],rule["premises"])
        if self.config["rule_rank"]=="quality":
            rules.sort(key=rule_order)
        else:
            rules.sort(key=rule_order)
        max_rules=int(self.config["max_rules"])
        if len(rules)>max_rules:
            # When the budget can represent them, keep a backoff layer from
            # every mined depth before filling by global quality order.
            depths=sorted({rule["specificity"] for rule in rules})
            quota,extra=divmod(max_rules,len(depths))
            selected=[]; selected_keys=set()
            for index,depth in enumerate(depths):
                take=quota+(index<extra)
                for rule in (candidate for candidate in rules if candidate["specificity"]==depth):
                    if take<=0: break
                    key=(rule["premises"],rule["target"])
                    selected.append(rule); selected_keys.add(key); take-=1
            for rule in rules:
                if len(selected)>=max_rules: break
                key=(rule["premises"],rule["target"])
                if key not in selected_keys:
                    selected.append(rule); selected_keys.add(key)
            rules=sorted(selected,key=rule_order)
        rule_sources=[]; point_channel_sources=[]; point_compiled_ids=set()
        for index,rule in enumerate(rules,1):
            rule["id"]=f"mined_{index}"
            terms=[f'({p.title()} $case {json.dumps(v)})' for p,v in rule["premises"]]
            premise=terms[0] if len(terms)==1 else f"(And {' '.join(terms)})"
            ctv=f'(CTV (STV {rule["strength"]} {rule["confidence"]}) (STV {rule["negative_strength"]} {rule["negative_confidence"]}))'
            shared_source=(
                f'(: {rule["id"]} (Implication {premise} '
                f'(Engagement $case {json.dumps(rule["target"])})) {ctv})'
            )
            rule_sources.append(shared_source)

            # Weighted aggregation needs every applicable mined rule, whereas
            # one bounded query against the shared Engagement target returns
            # only the revision representative reached before its search budget
            # expires. Compile an additional isolated channel for each rule.
            # Max/hybrid continue to query the unchanged shared target above.
            channel=f'point_{rule["id"]}'
            variant_id=f"point_variant_{index}"
            decision_id=f"point_decision_rule_{index}"
            encoded_rule=json.dumps(rule["id"])
            encoded_channel=json.dumps(channel)
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
            point_channel_sources.extend((variant_source,decision_source))
            point_compiled_ids.update((rule["id"],variant_id,decision_id))
            rule["point_variant_id"]=variant_id
            rule["point_decision_id"]=decision_id
            rule["point_proof_channel_id"]=channel
            rule["point_proof_factorization"]={
                "schema":"isolated_extensional_point_channel_v1",
                "case_variable":"$case",
                "premise_tv":[1.0,1.0],
                "single_channel_producer":True,
                "rule_id":rule["id"],
                "variant_id":variant_id,
                "decision_id":decision_id,
                "proof_channel_id":channel,
                "premises":[list(item) for item in rule["premises"]],
                "rule_source_sha256":hashlib.sha256(
                    variant_source.encode("utf-8")
                ).hexdigest(),
                "decision_source_sha256":hashlib.sha256(
                    decision_source.encode("utf-8")
                ).hexdigest(),
            }
        pair_mining=self._mine_pairwise(
            indexed_events=indexed_events,retention=retention
        )
        pair_workspace_plans={
            item["plan"] for item in pair_mining.get("workspace_sync",[])
        }
        active_workspace_plans=frozenset(
            workspace_plans | pair_workspace_plans
        )
        retained_relational_proofs=self._retained_live_relational_proofs()
        # Retraction by proof name is process-global in the MeTTa runtime and a
        # second KB still shares all compiled rule indexes. Install one complete
        # point+pair snapshot in a fresh process instead. The old worker stays
        # live until this compilation succeeds, so a failed remine cannot leave
        # the scorer half-retracted.
        self.engine.replace([
            *rule_sources,*point_channel_sources,
            *self._pair_rule_sources,*LIVE_NEGATIVE_RULE_SOURCES,
            *(RELATIONAL_STRUCTURAL_RULES
              if self.config.get("relational_evidence_mode")=="chained"
              else ()),
        ])
        self._point_rule_sources=rule_sources
        self._point_channel_sources=point_channel_sources
        self._active_rule_ids=point_compiled_ids; self.mined_rules=rules
        self._active_mining_workspace_plans=active_workspace_plans
        self._click_base_rate=base_rate
        self._loaded_candidates.clear(); self._loaded_pairs.clear()
        self._loaded_point_channels.clear()
        self._point_channel_proof_cache.clear()
        self._point_channel_templates.clear()
        self._point_query_calls=0; self._point_query_roots=0
        self._point_pruned_query_roots=0
        self._point_channel_activations=0
        self._point_reused_channel_activations=0
        self._last_point_completeness={}
        self._last_point_cache_stats={}
        self._loaded_pair_channels.clear(); self._pair_channel_proof_cache.clear()
        self._pair_channel_templates.clear()
        self._pair_case_attrs.clear(); self._pair_proof_cache.clear(); self._pair_margin_cache.clear()
        self._pair_proof_origins.clear()
        self._relational_feature_cache.clear()
        self._loaded_relational_statements.clear()
        self._live_relational_proof_ledger=retained_relational_proofs
        self._candidate_relational_proof_refs.clear()
        self._relational_query_calls=0
        self._relational_query_roots=0
        self.pending_events=0; self.version+=1; self.feed_cache.clear(); self._proof_cache.clear(); self.last_mined_at=time.time()
        if self._background_mining is not None:
            # Explicit/configuration mines cover every event visible to this Lab.
            # A concurrently staged background artifact will fail its version
            # check and be disposed rather than overwriting this newer snapshot.
            self._last_mined_event_sequence=self._event_sequence
        workspace_prune={"removed":[],"error":None,"deferred":True}
        if not self._staged_background_build:
            workspace_prune={
                **self._prune_mining_workspace_cache(),"deferred":False,
            }
        result={"rules":len(rules),"cases":len(training_events),"source_cases":n,
                "positives":sample_target_total,"negatives":len(training_events)-sample_target_total,
                "source_positives":target_total,"source_negatives":n-target_total,
                "online_negative_sampling":self._online_negative_sampling_audit(
                    training_cases,indexed_events
                ),
                "retention":retention_audit,
                "sample_base_rate":round(sample_target_total/len(training_events),8),
                "base_rate":round(base_rate,8),"depths":list(range(2,int(self.config["conjunctions"])+1)),
                "feature_vocabulary":{feature:len(values) for feature,values in vocabulary.items()},
                "rules_by_premises":dict(sorted(Counter(rule["specificity"] for rule in rules).items())),
                "point_proof_channels":len(rules),
                "point_proof_factorization":{
                    "enabled":True,
                    "serving_mode":"weighted_aggregation",
                    "shared_target_modes":["max","hybrid"],
                    "mode":"alpha_normalized_isolated_channels",
                    "safety_contract":"isolated_extensional_point_channel_v1",
                    "maximum_templates":len(rules),
                    "host_applicability":"exact categorical premise join",
                    "reasoner_role":"two-hop channel proof and inferred STV",
                },
                "miner_calls":len(plan),"miner_strategy":strategy,
                "fpminer_query_calls":sum(
                    item["miner_calls"] for item in fpminer_incremental
                ),
                "fpminer_incremental":fpminer_incremental,
                "workspace_mode":MINING_WORKSPACE_MODE,
                "workspace_sync":workspace_sync,
                "workspace_plans":len(active_workspace_plans),
                "workspace_prune":workspace_prune,
                "full_structure_research":(
                    strategy!="fixed_combinations" or any(
                        item["full_structure_research"]
                        for item in fpminer_incremental
                    )
                ),
                "full_population_ctv_estimation":True,
                "ctv_evidence_k":float(self.config["ctv_evidence_k"]),
                "fpminer_min_support":int(self.config["min_support"]),
                "fpminer_support_unit":"raw_point_case",
                "target_min_support":(int(self.config["min_support"])
                                      if strategy=="target_aware" else None),
                "target_support_unit":("raw_point_case"
                                       if strategy=="target_aware" else None),
                "target_search":target_search,"pairwise":pair_mining,
                "seconds":round(time.perf_counter()-started,3),"version":self.version}
        self.last_mining=result
        return result

    @staticmethod
    def pair_candidate_case(attrs):
        signature=json.dumps(attrs,sort_keys=True,separators=(",",":"),ensure_ascii=False)
        digest=hashlib.blake2s(signature.encode("utf-8"),digest_size=10).hexdigest()
        return f"pair_candidate_{digest}"

    def _compile_pair_rule_matcher(self):
        """Compile an exact inverted join for the active pair-rule premises.

        Pair-rule applicability used to rescan every premise of every rule for
        both orientations of every candidate comparison.  The compiled index
        maps each observed ``(predicate, value)`` token to the rules containing
        it. A rule activates only when all of its indexed premises matched, so
        this changes traversal cost without changing applicability or order.

        The immutable signature also notices tests, staged promotion, or other
        callers that replace or mutate ``pair_rules`` directly.
        """
        premises=tuple(
            tuple(tuple(item) for item in rule["premises"])
            for rule in self.pair_rules
        )
        if premises==getattr(self,"_pair_matcher_signature",None):
            return
        inverted={}
        for rule_index,rule_premises in enumerate(premises):
            for token in rule_premises:
                inverted.setdefault(token,[]).append(rule_index)
        self._pair_matcher_signature=premises
        self._pair_matcher_inverted={
            token:tuple(rule_indices)
            for token,rule_indices in inverted.items()
        }
        self._pair_matcher_required=tuple(map(len,premises))

    def _matching_pair_rule_premises(self,attrs):
        """Return active premises in original rule order via the compiled join."""
        match_counts=[0]*len(self._pair_matcher_signature)
        inverted=self._pair_matcher_inverted
        for token in attrs.items():
            for rule_index in inverted.get(token,()):
                match_counts[rule_index]+=1
        signature=self._pair_matcher_signature
        return tuple(
            signature[rule_index]
            for rule_index,required in enumerate(self._pair_matcher_required)
            if match_counts[rule_index]==required
        )

    @staticmethod
    def _reverse_pair_features(attrs):
        """Reverse one pair projection without recomputing candidate evidence.

        Every directional pair encoder is antisymmetric. Categorical side
        predicates swap names; directional values swap their ``left`` and
        ``right`` prefixes; shared/equal/incomparable values remain unchanged.
        This operates on the unbounded projection so the normal vocabulary
        filter is still applied independently to the reversed orientation.
        """
        reversed_attrs={}
        for predicate,value in attrs.items():
            reverse_predicate=PAIR_SIDE_PREDICATE_SWAP.get(predicate,predicate)
            if predicate in PAIR_CATEGORICAL_SIDE_PREDICATES:
                # Category symbols are opaque: a real category may legally be
                # named "left", "right_known" or "left_q1". Only the side
                # predicate swaps for categorical facts.
                reverse_value=value
            elif value=="left": reverse_value="right"
            elif value=="right": reverse_value="left"
            elif value=="left_known": reverse_value="right_known"
            elif value=="right_known": reverse_value="left_known"
            elif isinstance(value,str) and value.startswith("left_q"):
                reverse_value="right_"+value[len("left_"):]
            elif isinstance(value,str) and value.startswith("right_q"):
                reverse_value="left_"+value[len("right_"):]
            else: reverse_value=value
            reversed_attrs[reverse_predicate]=reverse_value
        return reversed_attrs

    def _oriented_pair_specs(self,left,right,case_cache=None):
        """Build both orientations from one candidate-feature comparison."""
        left_article,left_attrs=left
        right_article,right_attrs=right
        started=time.perf_counter()
        forward_raw=self._pair_features(
            left_attrs,right_attrs,left_article,right_article,
            needed=self._pair_feature_vocabulary,
        )
        feature_seconds=time.perf_counter()-started
        started=time.perf_counter()
        reverse_raw=self._reverse_pair_features(forward_raw)
        reverse_seconds=time.perf_counter()-started
        started=time.perf_counter()
        forward_attrs=self._bounded_pair_features(forward_raw)
        reverse_attrs=self._bounded_pair_features(reverse_raw)
        bounding_seconds=time.perf_counter()-started
        started=time.perf_counter()
        forward_activation=self._matching_pair_rule_premises(forward_attrs)
        reverse_activation=self._matching_pair_rule_premises(reverse_attrs)
        activation_seconds=time.perf_counter()-started
        case_cache={} if case_cache is None else case_cache
        missing=object(); cache_seconds=0.0; serialization_seconds=0.0
        def activation_case(activation):
            nonlocal cache_seconds,serialization_seconds
            started=time.perf_counter()
            # The compiled matcher already returns a canonical immutable tuple
            # in model rule order. Re-normalizing it for every orientation was
            # pure allocation in the serving hot path.
            key=activation
            case=case_cache.get(key,missing)
            cache_seconds+=time.perf_counter()-started
            if case is missing:
                started=time.perf_counter()
                case=self.pair_candidate_case({"activation":activation})
                serialization_seconds+=time.perf_counter()-started
                case_cache[key]=case
            return case
        forward_case=activation_case(forward_activation)
        reverse_case=activation_case(reverse_activation)
        return (
            (forward_case,forward_attrs),(reverse_case,reverse_attrs),
            {
                "pair_feature_derivation_seconds":feature_seconds,
                "reverse_orientation_derivation_seconds":reverse_seconds,
                "pair_feature_bounding_seconds":bounding_seconds,
                "pair_activation_join_seconds":activation_seconds,
                "pair_activation_case_cache_seconds":cache_seconds,
                "pair_case_serialization_seconds":serialization_seconds,
            },
        )

    @staticmethod
    def pair_channel_case(dependency,channel,premises):
        """Return the alpha-normalized proof case for one isolated rule.

        Pair rules are universally quantified over ``$pair`` and their input
        facts are certain, extensional statements.  Consequently, a channel's
        proof and inferred STV are invariant under a collision-resistant renaming
        of the pair atom.  This stable identity lets proof-margin mode ask
        PeTTa once per mined channel instead of once per observed joint rule
        activation vector.
        """
        signature=json.dumps({
            "dependency":str(dependency),"channel":str(channel),
            "premises":[list(item) for item in premises],
        },sort_keys=True,separators=(",",":"),ensure_ascii=False)
        digest=hashlib.blake2s(signature.encode("utf-8"),digest_size=20).hexdigest()
        return f"pair_channel_{digest}"

    def _pair_spec(self,left,right):
        self._compile_pair_rule_matcher()
        left_article,left_attrs=left
        right_article,right_attrs=right
        attrs=self._bounded_pair_features(
            self._pair_features(
                left_attrs,right_attrs,left_article,right_article,
                needed=self._pair_feature_vocabulary,
            )
        )
        # Active rules can only distinguish their own premise match vector.
        # Reusing one grounded context for an identical activation vector is
        # semantically exact and bounds serving identities by 2^rule-count,
        # rather than by every irrelevant combination of raw feature values.
        activation=self._matching_pair_rule_premises(attrs)
        signature={"activation":activation}
        return self.pair_candidate_case(signature),attrs

    def _ensure_pair_specs(self,specs,*,timeout_sec=None):
        materialization_started=time.perf_counter()
        timeout_sec=(self._serving_reasoner_timeout()
                     if timeout_sec is None else float(timeout_sec))
        # Isolated proof-margin channels are grounded lazily below as one
        # alpha-normalized template per rule.  Keep the candidate activation
        # attributes for the exact applicability check, but do not materialize
        # the exponentially many joint activation contexts in PeTTa.
        cache_limit=int(self.config.get(
            "max_proof_cache_entries",DEFAULT_MAX_PROOF_CACHE_ENTRIES
        ))
        cache_started=time.perf_counter()
        new_cases={case for case,_attrs in specs
                   if case not in self._pair_case_attrs}
        if len(self._pair_case_attrs)+len(new_cases)>cache_limit:
            raise RuntimeError(
                "pair case cache limit exceeded; promote a fresh rule snapshot"
            )
        if self.config["pair_aggregation"]=="proof_margin":
            for case,attrs in specs:
                # Cases are keyed by their selected-rule activation vector.
                # Raw unused feature values may differ without changing any
                # possible proof, so the first representative is sufficient.
                self._pair_case_attrs.setdefault(case,attrs)
            cache_seconds=time.perf_counter()-cache_started
            self._last_pair_materialization_profile={
                "pair_case_cache_index_seconds":cache_seconds,
                "pair_atom_serialization_seconds":0.0,
                "pair_atomspace_insertion_seconds":0.0,
                "candidate_pair_atoms_inserted":0,
                "factorized_channels":True,
                "total_seconds":time.perf_counter()-materialization_started,
            }
            return
        cache_seconds=time.perf_counter()-cache_started
        serialization_started=time.perf_counter()
        facts=[]; missing=set()
        for case,attrs in specs:
            self._pair_case_attrs.setdefault(case,attrs)
            if case in self._loaded_pairs or case in missing: continue
            missing.add(case)
            for predicate,value in attrs.items():
                facts.append(
                    f'(: fact_{case}_{predicate} ({predicate.title()} {case} {json.dumps(value)}) '
                    '(STV 1.0 1.0))'
                )
        serialization_seconds=time.perf_counter()-serialization_started
        insertion_started=time.perf_counter()
        for offset in range(0,len(facts),1000):
            self.engine.add_atoms_no_check(
                facts[offset:offset+1000],timeout_sec=timeout_sec
            )
        self._loaded_pairs.update(missing)
        insertion_seconds=time.perf_counter()-insertion_started
        self._last_pair_materialization_profile={
            "pair_case_cache_index_seconds":cache_seconds,
            "pair_atom_serialization_seconds":serialization_seconds,
            "pair_atomspace_insertion_seconds":insertion_seconds,
            "candidate_pair_atoms_inserted":len(facts),
            "factorized_channels":False,
            "total_seconds":time.perf_counter()-materialization_started,
        }

    def _proofs_for_pair_specs(self,specs,*,timeout_sec=None):
        runtime_started=time.perf_counter()
        timeout_sec=(self._serving_reasoner_timeout()
                     if timeout_sec is None else float(timeout_sec))
        batch=max(1,int(self.config["query_batch_size"])); calls=0
        aggregation=self.config["pair_aggregation"]
        proof_roots=tuple(sorted({(
            rule.get("dependency_id",rule["id"]),
            rule.get("proof_channel_id",rule.get("dependency_id",rule["id"])),
        ) for rule in self.pair_rules}))
        inference_key=(self.version,int(self.config["pair_chain_steps"]),batch,
                       aggregation,proof_roots)
        unique=list(dict.fromkeys(case for case,_attrs in specs))
        missing=[case for case in unique if (*inference_key,case) not in self._pair_proof_cache]
        cache_limit=int(self.config.get(
            "max_proof_cache_entries",DEFAULT_MAX_PROOF_CACHE_ENTRIES
        ))
        if len(self._pair_proof_cache)+len(missing)>cache_limit:
            raise RuntimeError(
                "pair proof cache limit exceeded; promote a fresh rule snapshot"
            )
        pair_stats={
            "case_root_requests":len(unique),
            "case_root_hits":len(unique)-len(missing),
            "case_root_misses":len(missing),
            "channel_root_requests":0,
            "channel_root_hits":0,
            "channel_root_misses":0,
        }
        topology_seconds=0.0; activation_join_seconds=0.0
        template_serialization_seconds=0.0; atomspace_insertion_seconds=0.0
        query_seconds=0.0; inserted_template_atoms=0
        if aggregation=="proof_margin":
            topology_started=time.perf_counter()
            supplied_attrs={case:attrs for case,attrs in specs}
            rules_by_root={}
            for rule in self.pair_rules:
                root=(
                    rule.get("dependency_id",rule["id"]),
                    rule.get("proof_channel_id",rule.get(
                        "dependency_id",rule["id"]
                    )),
                )
                rules_by_root.setdefault(root,[]).append(rule)
            # Exact factorization is deliberately fail-closed.  Every queried
            # root must have one producer, an isolated variant channel, and a
            # conjunction of keyed same-case extensional facts.  Shared-target
            # posterior mode follows the unfactorized branch below.
            topology={}
            compiled_source_hashes={hashlib.sha256(source.encode("utf-8")).hexdigest()
                                    for source in self._pair_rule_sources}
            for root,producers in rules_by_root.items():
                dependency,channel=root
                if len(producers)!=1:
                    raise RuntimeError(
                        f"proof channel {root!r} has {len(producers)} producers"
                    )
                rule=producers[0]
                premises=tuple(rule.get("premises",()))
                predicates=[predicate for predicate,_value in premises]
                contract=rule.get("proof_factorization",{})
                if (not premises or len(predicates)!=len(set(predicates))
                        or channel==dependency
                        or channel!=rule.get("variant_id")
                        or any(not isinstance(predicate,str)
                               or not re.fullmatch(r"[a-z][a-z0-9_]*",predicate)
                               or not isinstance(value,str)
                               for predicate,value in premises)
                        or contract.get("schema")!="isolated_extensional_pair_channel_v1"
                        or contract.get("case_variable")!="$pair"
                        or contract.get("premise_tv")!=[1.0,1.0]
                        or contract.get("single_channel_producer") is not True
                        or contract.get("dependency_id")!=dependency
                        or contract.get("proof_channel_id")!=channel
                        or contract.get("premises")!=[
                            list(item) for item in premises
                        ]
                        or contract.get("rule_source_sha256")
                           not in compiled_source_hashes):
                    raise RuntimeError(
                        f"proof channel {root!r} is not safely factorable"
                    )
                topology[root]=premises
            topology_seconds=time.perf_counter()-topology_started
            activation_started=time.perf_counter()
            active_by_case={case:[] for case in missing}
            stored_attrs=getattr(self,"_pair_case_attrs",{})
            for case in missing:
                attrs=stored_attrs.get(case,supplied_attrs.get(case,{}))
                for dependency,channel in proof_roots:
                    premises=topology[(dependency,channel)]
                    if all(attrs.get(predicate)==value
                           for predicate,value in premises):
                        active_by_case[case].append((dependency,channel,premises))
            activation_join_seconds=time.perf_counter()-activation_started
            active_uses=sum(map(len,active_by_case.values()))
            active_templates=list(dict.fromkeys(
                (dependency,channel,premises)
                for case in missing
                for dependency,channel,premises in active_by_case[case]
            ))
            channel_key=lambda item:(*inference_key,item[0],item[1],item[2])
            channel_cache=getattr(self,"_pair_channel_proof_cache",None)
            if channel_cache is None:
                channel_cache={}; self._pair_channel_proof_cache=channel_cache
            query_templates=[item for item in active_templates
                             if channel_key(item) not in channel_cache]
            if len(channel_cache)+len(query_templates)>cache_limit:
                raise RuntimeError(
                    "pair proof-channel cache limit exceeded; promote a fresh "
                    "rule snapshot"
                )
            pair_stats.update(
                channel_root_requests=len(active_templates),
                channel_root_hits=len(active_templates)-len(query_templates),
                channel_root_misses=len(query_templates),
            )
            serialization_started=time.perf_counter(); facts=[]
            loaded_channels=getattr(self,"_loaded_pair_channels",None)
            if loaded_channels is None:
                loaded_channels=set(); self._loaded_pair_channels=loaded_channels
            template_audit=getattr(self,"_pair_channel_templates",None)
            if template_audit is None:
                template_audit={}; self._pair_channel_templates=template_audit
            known_template_identities={
                item["case"]:(
                    item["dependency_id"],item["proof_channel_id"],
                    tuple(tuple(fact) for fact in item["facts"]),
                )
                for item in template_audit.values()
            }
            pending_loaded=set(); pending_audit={}
            for dependency,channel,premises in query_templates:
                template=self.pair_channel_case(dependency,channel,premises)
                loaded_key=(self.version,dependency,channel,premises)
                identity=(dependency,channel,premises)
                previous_identity=known_template_identities.setdefault(template,identity)
                if previous_identity!=identity:
                    raise RuntimeError(
                        f"pair proof template hash collision at {template}"
                    )
                pending_audit[loaded_key]={
                    "case":template,"dependency_id":dependency,
                    "proof_channel_id":channel,
                    "facts":[[predicate,value] for predicate,value in premises],
                }
                if loaded_key in loaded_channels: continue
                for index,(predicate,value) in enumerate(premises,1):
                    facts.append(
                        f'(: fact_{template}_{index}_{predicate} '
                        f'({predicate.title()} {template} {json.dumps(value)}) '
                        '(STV 1.0 1.0))'
                    )
                pending_loaded.add(loaded_key)
            template_serialization_seconds=(
                time.perf_counter()-serialization_started
            )
            insertion_started=time.perf_counter()
            for offset in range(0,len(facts),1000):
                self.engine.add_atoms_no_check(
                    facts[offset:offset+1000],timeout_sec=timeout_sec
                )
            atomspace_insertion_seconds=time.perf_counter()-insertion_started
            inserted_template_atoms=len(facts)
            staged_channel_results={}
            query_started=time.perf_counter()
            for offset in range(0,len(query_templates),batch):
                items=query_templates[offset:offset+batch]
                queries=[
                    f'(: $proof (PairSignal '
                    f'{self.pair_channel_case(dependency,channel,premises)} '
                    f'{json.dumps(dependency)} {json.dumps(channel)}) $tv)'
                    for dependency,channel,premises in items
                ]
                # Each independently mined vote is a real two-hop proof:
                # predicate -> MinedPairPreference -> PairSignal.
                steps=max(10,int(self.config["pair_chain_steps"]))*len(items)
                results=self.engine.query_many(
                    queries,steps=steps,timeout_sec=timeout_sec
                ); calls+=1
                for item,proofs in zip(items,results):
                    staged_channel_results[channel_key(item)]=proofs
            query_seconds=time.perf_counter()-query_started
            candidate_channel_results={
                **channel_cache,**staged_channel_results,
            }
            unproved=[
                (dependency,channel)
                for dependency,channel,premises in active_templates
                if not candidate_channel_results.get(
                    channel_key((dependency,channel,premises))
                )
            ]
            if unproved:
                raise RuntimeError(
                    "active isolated pair channels returned no PeTTa proof: "
                    +", ".join(
                        f"{dependency}/{channel}"
                        for dependency,channel in unproved[:8]
                    )
                )
            # Publish cache and audit state only after every PeTTa mutation and
            # query has succeeded. A retired/failed worker can never leave a
            # false empty proof or a falsely "loaded" fact template behind.
            channel_cache.update(staged_channel_results)
            loaded_channels.update(pending_loaded)
            template_audit.update(pending_audit)
            self._pair_query_roots=getattr(self,"_pair_query_roots",0)+len(query_templates)
            self._pair_pruned_query_roots=(
                getattr(self,"_pair_pruned_query_roots",0)
                +len(missing)*len(proof_roots)-active_uses
            )
            self._pair_channel_activations=(
                getattr(self,"_pair_channel_activations",0)+active_uses
            )
            self._pair_reused_channel_activations=(
                getattr(self,"_pair_reused_channel_activations",0)
                +max(0,active_uses-len(query_templates))
            )
            origins=getattr(self,"_pair_proof_origins",None)
            if origins is None:
                origins={}; self._pair_proof_origins=origins
            for case,active in active_by_case.items():
                candidate_proofs=[]
                for dependency,channel,premises in active:
                    proofs=channel_cache[
                        channel_key((dependency,channel,premises))
                    ]
                    candidate_proofs.extend(proofs)
                    for proof in proofs:
                        origins[(self.version,case,proof)]=dependency
                self._pair_proof_cache[(*inference_key,case)]=candidate_proofs
        else:
            query_started=time.perf_counter()
            for offset in range(0,len(missing),batch):
                cases=missing[offset:offset+batch]
                queries=[f'(: $proof (PairWin {case}) $tv)' for case in cases]
                # Posterior mode adds a third merge hop over PairSignal.
                steps=max(12,int(self.config["pair_chain_steps"]))*len(cases)
                results=self.engine.query_many(
                    queries,steps=steps,timeout_sec=timeout_sec
                ); calls+=1
                for case,proofs in zip(cases,results):
                    self._pair_proof_cache[(*inference_key,case)]=proofs
            query_seconds=time.perf_counter()-query_started
        self._pair_query_calls+=calls
        pair_stats["hits"]=(pair_stats["case_root_hits"]
                            +pair_stats["channel_root_hits"])
        pair_stats["misses"]=(pair_stats["case_root_misses"]
                              +pair_stats["channel_root_misses"])
        pair_stats["requests"]=pair_stats["hits"]+pair_stats["misses"]
        self._last_pair_cache_stats=pair_stats
        self._last_pair_reasoning_profile={
            "proof_topology_validation_seconds":topology_seconds,
            "proof_activation_join_seconds":activation_join_seconds,
            "proof_template_serialization_seconds":template_serialization_seconds,
            "proof_atomspace_insertion_seconds":atomspace_insertion_seconds,
            "proof_query_seconds":query_seconds,
            "proof_template_atoms_inserted":inserted_template_atoms,
            "total_seconds":time.perf_counter()-runtime_started,
        }
        return {case:self._pair_proof_cache[(*inference_key,case)] for case in unique},calls

    @staticmethod
    def _pair_posterior(proofs,prior=0.5):
        if not proofs: return prior
        # query_many returns PeTTaChainer's canonical root proofs.  Strength is
        # the preference probability; confidence shrinks it to the neutral
        # pair prior rather than being multiplied into a pseudo-probability.
        strength,confidence=max((proof_tv(proof) for proof in proofs),
                                key=lambda tv:(tv[1],abs(tv[0]-prior)))
        return confidence*strength+(1.0-confidence)*prior

    def _proof_dependency_margins(self,case,proofs,*,cache=None):
        """Return PeTTa confidence-adjusted margins keyed by dependency.

        A PairSignal STV is a probability around the balanced pair prior 0.5.
        Its base linear decision margin is therefore ``c * (2s - 1)``.  The
        configured signed power may temper that linear/log-odds margin after
        PeTTa inference. Reading the STV from the returned root proof preserves
        PeTTa's CTV propagation and revision semantics instead of reducing the
        reasoner to a Boolean gate.
        """
        if not proofs: return {}
        margin_cache=(self._pair_margin_cache if cache is None else cache)
        margin_transform=self.config["pair_margin_transform"]
        margin_power=float(self.config["pair_margin_power"])
        cache_key=("dependencies",self.version,margin_transform,margin_power,
                   case,tuple(proofs))
        if cache_key in margin_cache: return margin_cache[cache_key]
        cache_limit=int(self.config.get(
            "max_proof_cache_entries",DEFAULT_MAX_PROOF_CACHE_ENTRIES
        ))
        if len(margin_cache)>=cache_limit:
            raise RuntimeError(
                "pair margin cache limit exceeded; promote a fresh rule snapshot"
            )
        margins={}
        origins=getattr(self,"_pair_proof_origins",{})
        for proof in proofs:
            origin=origins.get((self.version,case,proof))
            direct_origin=None
            if origin is None and proof.startswith("(direct-reconstruction "):
                direct_match=re.match(
                    r'^\(direct-reconstruction \(PairSignal [A-Za-z_][A-Za-z0-9_]* '
                    r'("(?:[^"\\]|\\.)*") ',proof
                )
                if direct_match is None:
                    raise RuntimeError(
                        "malformed direct pair reconstruction record"
                    )
                direct_origin=json.loads(direct_match.group(1))
            dependencies=({origin} if origin is not None else
                          {direct_origin} if direct_origin is not None else
                          set(re.findall(r"\bpair_mined_cluster_\d+\b",proof)))
            strength,confidence=proof_tv(proof)
            posterior=0.5+confidence*(strength-0.5)
            if margin_transform=="log_odds":
                # Pairwise AUC is driven by relative evidence. Log odds retains
                # PeTTa's confidence shrinkage while preventing a strong,
                # well-supported proof from being flattened into the same
                # nearly-linear vote as weak evidence.
                bounded=max(1e-9,min(1.0-1e-9,posterior))
                proof_margin=math.log(bounded/(1.0-bounded))
            else:
                proof_margin=2.0*posterior-1.0
            # A positive power preserves the proof direction and ordering of
            # variants inside one dependency.  Powers below one conservatively
            # temper calibration-scale differences between independent
            # evidence families; the default 1.0 is exactly the old margin.
            if proof_margin:
                proof_margin=math.copysign(
                    abs(proof_margin)**margin_power,proof_margin
                )
            for dependency_id in dependencies:
                # A dependency cluster can expose several correlated proof
                # variants. PeTTa supplies their inferred STVs; retain only the
                # dominant canonical alternative for that evidence source.
                previous=margins.get(dependency_id)
                if previous is None or abs(proof_margin)>abs(previous):
                    margins[dependency_id]=proof_margin
        margin_cache[cache_key]=margins
        return margins

    def _proof_vote_margin(self,case,proofs):
        """Sum PeTTaChainer margins once per independent dependency."""
        return sum(self._proof_dependency_margins(case,proofs).values())

    @staticmethod
    def _pair_rule_family(rule):
        """Return a dataset-independent evidence-family label for one rule."""
        # A conjunction containing text-semantic evidence still belongs to the
        # text family.  Keeping this binary is important: a mixed conjunction
        # must not accidentally create a third family and receive another
        # equal-weight vote in ``balanced_rank``.
        if (rule.get("categorical_fact_family")=="llm_format"
                or rule.get("dependency_owner")=="pair_text_semantic_top3_mean"):
            return "text_semantic"
        return (
            "text_semantic"
            if any(
                str(predicate).startswith(("pair_text_semantic_","pair_llm_"))
                or predicate=="pair_rel_concept_continuity_scope"
                   for predicate,_value in rule.get("premises",()))
            else "structured_symbolic"
        )

    def _relational_evaluation_activation_audit(
            self,point_specs,pair_specs):
        """Count relation evidence and rule uses in the held-out workload.

        This runs only after the normal proof calls have completed. The
        selected weighted/proof-margin paths fail closed whenever an active
        isolated channel has no PeTTa proof, so their activation counts are
        proof-backed uses rather than a mined-rule inventory.
        """
        relation_fields=(
            REL_ENTITY_CONTINUITY_SCOPE,
            REL_CONCEPT_CONTINUITY_SCOPE,
        )
        relation_proof_fields={
            REL_ENTITY_CONTINUITY_SCOPE:REL_ENTITY_CONTINUITY_PROOF_IDS,
            REL_CONCEPT_CONTINUITY_SCOPE:REL_CONCEPT_CONTINUITY_PROOF_IDS,
        }
        scope_counts={field:Counter() for field in relation_fields}
        proof_positive_candidates=0
        proof_refs_by_case=getattr(
            self,"_candidate_relational_proof_refs",{}
        )
        for _aid,case,_attrs,raw_attrs in point_specs:
            positive=False
            for field in relation_fields:
                value=raw_attrs.get(field)
                if value is not None:
                    scope_counts[field][str(value)]+=1
                if value in {"recent","older"}:
                    references=(proof_refs_by_case.get(case,{}) or {}).get(
                        relation_proof_fields[field]
                    )
                    if not isinstance(references,(list,tuple)) or not references:
                        raise RuntimeError(
                            "proof-positive relation lacks provenance IDs in "
                            "the held-out candidate context"
                        )
                    positive=True
            proof_positive_candidates+=positive

        def summarize(rules,specs,*,pair):
            relation_predicates={
                (f"pair_{field}" if pair else field):field
                for field in relation_fields
            }
            activations=0; relational=0; proof_positive=0
            by_field_value=Counter(); rule_ids=set(); positive_ids=set()
            activated_instances=0; positive_instances=0
            for _case,attrs in specs:
                instance_relational=False; instance_positive=False
                for rule in rules:
                    premises=tuple(tuple(item)
                                   for item in rule.get("premises",()))
                    if not all(attrs.get(predicate)==value
                               for predicate,value in premises):
                        continue
                    activations+=1
                    selected=[
                        (relation_predicates[predicate],str(value))
                        for predicate,value in premises
                        if predicate in relation_predicates
                    ]
                    if not selected:
                        continue
                    relational+=1; instance_relational=True
                    identifier=str(rule.get(
                        "proof_channel_id" if pair else "id",
                        rule.get("id","unknown"),
                    ))
                    rule_ids.add(identifier)
                    positive=any(
                        value in ({"left","right"} if pair
                                  else {"recent","older"})
                        for _field,value in selected
                    )
                    for field,value in selected:
                        by_field_value[(field,value)]+=1
                    if positive:
                        proof_positive+=1; instance_positive=True
                        positive_ids.add(identifier)
                activated_instances+=instance_relational
                positive_instances+=instance_positive
            return {
                "instances":len(specs),
                "all_rule_activations":activations,
                "relational_rule_activations":relational,
                "proof_positive_relational_rule_activations":proof_positive,
                "instances_with_relational_rule_activation":activated_instances,
                "instances_with_proof_positive_relational_rule_activation":(
                    positive_instances
                ),
                "relational_rule_ids":sorted(rule_ids),
                "proof_positive_relational_rule_ids":sorted(positive_ids),
                "by_field_and_value":{
                    f"{field}={value}":count
                    for (field,value),count in sorted(by_field_value.items())
                },
            }

        point_inputs=[(case,attrs)
                      for _aid,case,attrs,_raw in point_specs]
        result={
            "candidate_contexts":len(point_specs),
            "proof_positive_candidate_contexts":proof_positive_candidates,
            "candidate_scope_values":{
                field:dict(sorted(values.items()))
                for field,values in scope_counts.items()
            },
            "point":summarize(
                self.mined_rules,point_inputs,pair=False
            ),
            "pair":summarize(
                self.pair_rules,list(pair_specs),pair=True
            ),
            "proof_positive_definition":{
                "point":"recent or older derived scope",
                "pair":"left or right comparison of two known scopes",
            },
            "counting_units":{
                "point":"candidate-rule activation",
                "pair":"oriented-comparison-rule activation",
            },
            "proof_gate_complete":bool(
                self.config.get("aggregation")=="weighted"
                and self._last_point_completeness.get("complete")
                and (
                    not pair_specs
                    or self.config.get("pair_aggregation")=="proof_margin"
                )
            ),
        }
        return result

    @staticmethod
    def _symbolic_rule_families(rule):
        """Fixed portable families; resolve ownership per dependency below."""
        families=set()
        for predicate,_value in rule.get("premises",()):
            if "lexical" in predicate or "title" in predicate:
                families.add("lexical")
            elif "transition" in predicate:
                families.add("transition")
            else:
                families.add("interest")
        return families or {"interest"}

    @staticmethod
    def _midrank_scores(values):
        """Map descending values to stable [0,1] midranks."""
        # Algebraically equal vote sums can differ at floating-point epsilon.
        # Do not turn that addition-order noise into a whole rank position.
        # This precision is finer than the public eight-decimal score contract.
        values=[round(float(value),12) for value in values]
        scores=[0.0]*len(values)
        denominator=max(1,len(values)-1)
        ordered=sorted(range(len(values)),key=lambda index:(-values[index],index))
        position=0
        while position<len(ordered):
            end=position+1
            value=values[ordered[position]]
            while end<len(ordered) and values[ordered[end]]==value:
                end+=1
            rank_score=1.0-((position+end-1)/2)/denominator
            for offset in range(position,end):
                scores[ordered[offset]]=rank_score
            position=end
        return scores

    @staticmethod
    def _logistic(value):
        """Map an arbitrary signed proof margin to a bounded probability."""
        # 0.5 + 0.5*tanh(x/2) is algebraically sigmoid(x), but remains stable
        # when a near-certain PeTTa posterior produces a large log-odds margin.
        return 0.5+0.5*math.tanh(float(value)/2.0)

    def _pairwise_plan(self,rows,specs):
        planning_started=time.perf_counter()
        self._compile_pair_rule_matcher()
        raw_by_article={aid:(self.article(aid),raw_attrs)
                        for aid,_case,_attrs,raw_attrs in specs}
        comparisons=[]; pair_specs=[]
        opponent_limit=int(self.config["pairwise_opponents"])
        comparison_limit=int(self.config["max_pair_comparisons"])
        graph_started=time.perf_counter()
        if opponent_limit<=0 or opponent_limit>=len(rows)-1:
            exhaustive_count=len(rows)*(len(rows)-1)//2
            if exhaustive_count>comparison_limit:
                raise ValueError(
                    "pairwise comparison budget exceeded: "
                    f"{exhaustive_count} unordered pairs for {len(rows)} "
                    f"candidates is greater than max_pair_comparisons="
                    f"{comparison_limit}; reduce the candidate slate or set "
                    "a finite pairwise_opponents value"
                )
            index_pairs=((left,right) for left in range(len(rows))
                         for right in range(left+1,len(rows)))
        else:
            # A cyclic comparison graph gives every candidate the same local
            # proof budget while retaining occasional top-vs-tail comparisons.
            pair_set=set()
            for left in range(len(rows)):
                for offset in range(1,opponent_limit+1):
                    right=(left+offset)%len(rows)
                    pair_set.add(tuple(sorted((left,right))))
                    if len(pair_set)>comparison_limit:
                        raise ValueError(
                            "bounded pairwise comparison budget exceeded: "
                            f"more than {comparison_limit} unordered pairs; "
                            "reduce the candidate slate or pairwise_opponents"
                        )
            index_pairs=iter(sorted(pair_set))
        graph_seconds=time.perf_counter()-graph_started
        feature_seconds=0.0; reverse_seconds=0.0; bounding_seconds=0.0
        activation_seconds=0.0; cache_seconds=0.0; serialization_seconds=0.0
        activation_case_cache={}
        for left_index,right_index in index_pairs:
            left_id=rows[left_index]["article"]["id"]
            right_id=rows[right_index]["article"]["id"]
            # Preserve instance-level test/instrumentation overrides. Normal
            # execution uses the exact one-pass antisymmetric constructor.
            if "_pair_spec" in self.__dict__:
                forward=self._pair_spec(raw_by_article[left_id],raw_by_article[right_id])
                reverse=self._pair_spec(raw_by_article[right_id],raw_by_article[left_id])
            else:
                forward,reverse,pair_profile=self._oriented_pair_specs(
                    raw_by_article[left_id],raw_by_article[right_id],
                    activation_case_cache,
                )
                feature_seconds+=pair_profile["pair_feature_derivation_seconds"]
                reverse_seconds+=pair_profile["reverse_orientation_derivation_seconds"]
                bounding_seconds+=pair_profile["pair_feature_bounding_seconds"]
                activation_seconds+=pair_profile["pair_activation_join_seconds"]
                cache_seconds+=pair_profile["pair_activation_case_cache_seconds"]
                serialization_seconds+=pair_profile["pair_case_serialization_seconds"]
            comparisons.append((left_index,right_index,forward[0],reverse[0]))
            pair_specs.extend((forward,reverse))
        total_seconds=time.perf_counter()-planning_started
        self._last_pair_plan_profile={
            "candidate_feature_reuse":True,
            "orientation_policy":"derive_reverse_from_exact_antisymmetry",
            "unordered_comparisons":len(comparisons),
            "oriented_pair_cases":len(pair_specs),
            "unique_activation_cases_in_slate":len(activation_case_cache),
            "comparison_graph_seconds":graph_seconds,
            "pair_feature_derivation_seconds":feature_seconds,
            "reverse_orientation_derivation_seconds":reverse_seconds,
            "pair_feature_bounding_seconds":bounding_seconds,
            "pair_activation_join_seconds":activation_seconds,
            "pair_activation_case_cache_seconds":cache_seconds,
            "pair_case_serialization_seconds":serialization_seconds,
            "pair_plan_assembly_seconds":max(0.0,total_seconds-graph_seconds
                -feature_seconds-reverse_seconds-bounding_seconds
                -activation_seconds-cache_seconds-serialization_seconds),
            "total_seconds":total_seconds,
        }
        return comparisons,pair_specs

    def _pairwise_rank(self,rows,specs,plan=None,proof_map=None,
                       reasoner_timeout_sec=None):
        if self.config["ranking_mode"]!="pairwise" or not self.pair_rules or len(rows)<2:
            self._last_pair_replay=None
            for row in rows:
                row.update(ranking_score=row["score"],pairwise_score=None,
                           pairwise_margin_score=None,
                           pairwise_proof_coverage=0.0,pairwise_directional_coverage=0.0)
            return rows
        comparisons,pair_specs=plan or self._pairwise_plan(rows,specs)
        if proof_map is None:
            self._ensure_pair_specs(
                pair_specs,timeout_sec=reasoner_timeout_sec
            )
            proof_map,_calls=self._proofs_for_pair_specs(
                pair_specs,timeout_sec=reasoner_timeout_sec
            )
        self._last_pair_replay={
            "version":self.version,
            "edges":tuple(
                (str(rows[left]["article"]["id"]),
                 str(rows[right]["article"]["id"]),forward,reverse)
                for left,right,forward,reverse in comparisons
            ),
            "proof_map":proof_map,
        }
        pair_totals=[0.0]*len(rows); comparison_counts=[0]*len(rows)
        covered=[0]*len(rows); directional_covered=[0]*len(rows)
        proof_margin=self.config["pair_aggregation"]=="proof_margin"
        family_fusion=self.config.get("pair_family_fusion","flat_margin")
        symbolic_families=family_fusion=="symbolic_balanced"
        magnitude_balanced=family_fusion=="balanced_margin"
        balanced_families=(
            proof_margin
            and family_fusion
                in {"balanced_rank","balanced_margin","symbolic_balanced"}
        )
        dependency_families={}
        for rule in self.pair_rules:
            dependency_id=rule.get("dependency_id",rule["id"])
            if symbolic_families:
                dependency_families.setdefault(dependency_id,set()).update(
                    self._symbolic_rule_families(rule))
                continue
            family=self._pair_rule_family(rule)
            # Several correlated variants may share one dependency.  If any
            # variant consumes text evidence, classify the indivisible PeTTa
            # dependency as text-semantic instead of depending on rule order.
            if (family=="text_semantic"
                    or dependency_id not in dependency_families):
                dependency_families[dependency_id]=family
        if not symbolic_families:
            dependency_families={key:{family} for key,family in dependency_families.items()}
        else:
            # One proof dependency can influence only one family midrank.
            # Splitting its margin across families would still duplicate its
            # vote after independent normalization. Mixed dependencies belong
            # to their most specific source under this fixed schema order.
            dependency_families={
                key:{next(family for family in ("lexical","transition","interest")
                          if family in families)}
                for key,families in dependency_families.items()
            }
        active_families=tuple(sorted(set().union(*dependency_families.values())))
        family_totals={family:[0.0]*len(rows) for family in active_families}
        for left_index,right_index,forward_case,reverse_case in comparisons:
            forward_proofs=proof_map[forward_case]; reverse_proofs=proof_map[reverse_case]
            if proof_margin:
                if balanced_families:
                    forward_margins=self._proof_dependency_margins(
                        forward_case,forward_proofs
                    )
                    reverse_margins=self._proof_dependency_margins(
                        reverse_case,reverse_proofs
                    )
                    comparison_family_totals={}
                    comparison_family_dependency_mass={}
                    margin=0.0
                    for dependency_id in sorted(set(forward_margins)|set(reverse_margins)):
                        if symbolic_families and dependency_id not in dependency_families:
                            raise RuntimeError("proof references a dependency outside the active symbolic model")
                        dependency_margin=(
                            forward_margins.get(dependency_id,0.0)
                            -reverse_margins.get(dependency_id,0.0)
                        )
                        margin+=dependency_margin
                        families=dependency_families.get(dependency_id,{"structured_symbolic"})
                        for family in families:
                            contribution=dependency_margin/len(families)
                            family_totals.setdefault(family,[0.0]*len(rows))
                            if magnitude_balanced:
                                # Average only the dependency evidence that
                                # actually participated in this comparison.
                                # Merely compiling an inactive rule must not
                                # dilute an active family's PeTTa margin.
                                comparison_family_totals[family]=(
                                    comparison_family_totals.get(family,0.0)
                                    +contribution
                                )
                                comparison_family_dependency_mass[family]=(
                                    comparison_family_dependency_mass.get(
                                        family,0.0
                                    )+1.0/len(families)
                                )
                            else:
                                family_totals[family][left_index]+=contribution
                                family_totals[family][right_index]-=contribution
                    if magnitude_balanced:
                        # First average correlated/alternative dependencies
                        # inside each active family, then give each family
                        # that actually supplied a proof one top-level term.
                        # An absent family is not entered in the denominator:
                        # abstention must neither invent nor dilute evidence.
                        active_family_margins=[]
                        for family,total in comparison_family_totals.items():
                            family_margin=total/comparison_family_dependency_mass[family]
                            active_family_margins.append(family_margin)
                            family_totals[family][left_index]+=family_margin
                            family_totals[family][right_index]-=family_margin
                        margin=(
                            sum(active_family_margins)/len(active_family_margins)
                            if active_family_margins else 0.0
                        )
                else:
                    margin=(self._proof_vote_margin(forward_case,forward_proofs)
                            -self._proof_vote_margin(reverse_case,reverse_proofs))
                pair_totals[left_index]+=margin; pair_totals[right_index]-=margin
            else:
                left=self._pair_posterior(forward_proofs); right=self._pair_posterior(reverse_proofs)
                # The two directions are complementary pieces of evidence, not
                # unnormalised class logits. Antisymmetrising around the pair
                # prior preserves neutral 0.5 when both directions agree.
                probability=0.5+0.5*(left-right)
                pair_totals[left_index]+=probability
                pair_totals[right_index]+=1.0-probability
            comparison_counts[left_index]+=1; comparison_counts[right_index]+=1
            if forward_proofs or reverse_proofs:
                covered[left_index]+=1; covered[right_index]+=1
            directional_covered[left_index]+=bool(forward_proofs)
            directional_covered[right_index]+=bool(reverse_proofs)
        rank_denominator=max(1,len(rows)-1); weight=float(self.config["pairwise_weight"])
        feedback_active=any(
            row.get("feedback_evidence",{}).get("rule_ids") for row in rows
        )
        # A configured pair weight of 1 normally makes point evidence an
        # audit-only signal.  Once PeTTa has proved live negative evidence,
        # give its point-proof order one bounded Borda vote beside the PeTTa
        # pair tournament.  The mined pair model retains at least three
        # quarters of the decision weight. This is rank fusion of two reasoner
        # outputs—not a host-language score penalty—and leaves all offline /
        # no-feedback rankings byte-for-byte on their configured path.
        applied_pair_weight=min(weight,0.75) if feedback_active else weight
        cluster_weights={}
        margin_power=float(self.config["pair_margin_power"])
        for rule in self.pair_rules:
            dependency_id=rule.get("dependency_id",rule["id"])
            posterior=0.5+float(rule.get("proof_confidence",rule.get("confidence",0.0)))*(
                float(rule.get("proof_strength",rule.get("strength",0.5)))-0.5
            )
            if self.config["pair_margin_transform"]=="log_odds":
                bounded=max(1e-9,min(1.0-1e-9,posterior))
                rule_margin=abs(math.log(bounded/(1.0-bounded)))
            else:
                rule_margin=abs(2.0*posterior-1.0)
            rule_margin=rule_margin**margin_power
            cluster_weights[dependency_id]=max(
                cluster_weights.get(dependency_id,0.0),
                rule_margin
            )
        maximum_margin=max(1e-12,sum(cluster_weights.values()))
        raw_pair_values=[pair_totals[index]/max(1,comparison_counts[index])
                         for index in range(len(rows))]
        family_rank_scores={}
        family_pair_values={}
        family_margin_scores={}
        fused_family_margin=None
        if balanced_families and family_totals:
            for family,totals in family_totals.items():
                family_raw=[totals[index]/max(1,comparison_counts[index])
                            for index in range(len(rows))]
                if magnitude_balanced:
                    # Give each family one top-level term in the fusion while
                    # retaining the magnitude of PeTTa's confidence-adjusted
                    # proof margins. Dependency-count normalization was
                    # already performed per comparison over the union of its
                    # forward/reverse proofs. An absent family remains zero.
                    # Unlike balanced_rank, an uncertain 0.52/0.10 proof
                    # cannot cancel an opposing 0.90/0.95 proof merely because
                    # both families induced the opposite ordinal order.
                    family_margin_scores[family]=family_raw
                    family_pair_values[family]=[
                        self._logistic(value)
                        for value in family_margin_scores[family]
                    ]
                else:
                    family_max=max(1e-12,sum(
                        weight/len(dependency_families.get(dependency_id,{"structured_symbolic"}))
                        for dependency_id,weight in cluster_weights.items()
                        if family in dependency_families.get(dependency_id,{"structured_symbolic"})
                    ))
                    family_pair_values[family]=[
                        0.5+0.5*value/family_max for value in family_raw
                    ]
                family_rank_scores[family]=self._midrank_scores(family_raw)
            family_count=len(family_rank_scores)
            if magnitude_balanced:
                # ``raw_pair_values`` contains the mean of the already fused
                # per-comparison margins.  Re-averaging the diagnostic family
                # totals here would incorrectly count an absent family as a
                # zero-valued observation.
                fused_family_margin=list(raw_pair_values)
                pair_values=[self._logistic(value)
                             for value in fused_family_margin]
                # Rank only after calibrated magnitudes from all families have
                # been combined.  The final midrank is required by the public
                # rank-fusion contract; it no longer erases magnitude before
                # conflicting evidence families meet.
                pair_rank_scores=self._midrank_scores(fused_family_margin)
            else:
                pair_values=[sum(values[index] for values in family_pair_values.values())
                             /family_count for index in range(len(rows))]
                pair_rank_scores=[sum(values[index] for values in family_rank_scores.values())
                                  /family_count for index in range(len(rows))]
        else:
            pair_values=([0.5+0.5*value/maximum_margin for value in raw_pair_values]
                         if proof_margin else raw_pair_values)
            pair_rank_scores=self._midrank_scores(pair_values)
        # Midrank candidates whose point proofs are identical.  Editorial
        # priors and article IDs remain final deterministic tie-breakers, but
        # they no longer masquerade as a continuous point-proof signal inside
        # rank fusion.
        pointwise_rank_scores=[0.0]*len(rows)
        point_position=0
        while point_position<len(rows):
            point_end=point_position+1
            point_signature=(rows[point_position]["score"],
                             rows[point_position]["stv"]["strength"],
                             rows[point_position]["stv"]["confidence"])
            while point_end<len(rows) and (
                rows[point_end]["score"],rows[point_end]["stv"]["strength"],
                rows[point_end]["stv"]["confidence"]
            )==point_signature:
                point_end+=1
            average_position=(point_position+point_end-1)/2
            rank_score=1.0-average_position/rank_denominator
            for offset in range(point_position,point_end):
                pointwise_rank_scores[offset]=rank_score
            point_position=point_end
        for index,row in enumerate(rows):
            denominator=max(1,comparison_counts[index])
            pointwise_rank=pointwise_rank_scores[index]
            pairwise_score=pair_values[index]
            pairwise_rank_score=pair_rank_scores[index]
            pair_signal=(pairwise_rank_score if self.config["pairwise_fusion"]=="rank"
                         else pairwise_score)
            row.update(
                pairwise_score=round(pairwise_score,8),
                pairwise_margin_score=round(raw_pair_values[index],8),
                pairwise_rank_score=round(pairwise_rank_score,8),
                pointwise_rank_score=round(pointwise_rank,8),
                ranking_score=round(applied_pair_weight*pair_signal
                                    +(1.0-applied_pair_weight)*pointwise_rank,8),
                pairwise_proof_coverage=round(covered[index]/denominator,8),
                pairwise_directional_coverage=round(directional_covered[index]/denominator,8),
                pairwise_weight_applied=applied_pair_weight,
                live_feedback_rank_fusion=feedback_active,
            )
            if balanced_families:
                row["pairwise_family_rank_scores"]={
                    family:round(values[index],8)
                    for family,values in family_rank_scores.items()
                }
            if magnitude_balanced:
                row["pairwise_family_margin_scores"]={
                    family:round(values[index],8)
                    for family,values in family_margin_scores.items()
                }
                row["pairwise_fused_margin_score"]=round(
                    fused_family_margin[index],8
                )
        rows.sort(key=lambda row:(-row["ranking_score"],-row["pairwise_score"],
                                  -row["score"],-row["stv"]["strength"],
                                  -row["stv"]["confidence"],
                                  -row["tie_break"]["topic_prior"],
                                  -row["tie_break"]["format_prior"],
                                  -row["tie_break"]["subcategory_prior"],
                                  row["article"]["id"]))
        return rows

    def _pairwise_replay_plan(self,rows,replay):
        """Build a sub-slate tournament from already proved pair edges."""
        if not replay or replay.get("version")!=self.version:
            return None
        positions={str(row["article"]["id"]):index
                   for index,row in enumerate(rows)}
        comparisons=[]
        for left_id,right_id,forward,reverse in replay.get("edges",()):
            if left_id not in positions or right_id not in positions:
                continue
            comparisons.append((positions[left_id],positions[right_id],
                                forward,reverse))
        expected=len(rows)*(len(rows)-1)//2
        if len(comparisons)!=expected:
            return None
        proof_map=replay.get("proof_map",{})
        required={case for _left,_right,forward,reverse in comparisons
                  for case in (forward,reverse)}
        if not required.issubset(proof_map):
            return None
        return (comparisons,()),proof_map

    def _rank(self,specs,groups,limit,apply_pairwise=True,
              reasoner_timeout_sec=None,include_context=False):
        popularity=self._popularity; rows=[]; prior=float(self._click_base_rate)
        for (aid,_case,_attrs,_raw_attrs),proofs in zip(specs,groups):
            scored=[(proof_tv(proof),proof) for proof in proofs]
            inference_tv,inference_proof=max(
                scored,key=lambda item:prior+item[0][1]*(item[0][0]-prior),
                                                default=((0.0,0.0),""))
            feedback_scored=[
                (proof_tv(proof),proof) for proof in proofs
                if LIVE_NEGATIVE_RULE_IDS.intersection(
                    re.findall(r"\bfeedback_skip_[a-z]+\b",proof)
                )
            ]
            feedback_rule_ids=sorted(set().union(*(
                LIVE_NEGATIVE_RULE_IDS.intersection(
                    re.findall(r"\bfeedback_skip_[a-z]+\b",proof)
                ) for _tv,proof in feedback_scored
            ))) if feedback_scored else []
            if self.config["aggregation"]=="max":
                tv,proof=inference_tv,inference_proof
                proof_rule_ids=set(re.findall(r"\bmined_\d+\b",proof))
                fired=[rule for rule in self.mined_rules if rule["id"] in proof_rule_ids]
                score=prior+tv[1]*(tv[0]-prior) if proofs else prior
                score_method="pettachainer_base_rate_posterior"
            else:
                proof_rule_ids=set().union(*(set(re.findall(r"\bmined_\d+\b",proof)) for _tv,proof in scored)) if scored else set()
                fired=[rule for rule in self.mined_rules if rule["id"] in proof_rule_ids]
                if fired:
                    weights=[max(1,int(rule.get("specificity",len(rule["premises"])))) for rule in fired]
                    evidence_confidence=sum(rule["confidence"]*weight for rule,weight in zip(fired,weights))/sum(weights)
                    evidence_score=sum(
                        (prior+rule["confidence"]*(rule["strength"]-prior))*weight
                        for rule,weight in zip(fired,weights)
                    )/sum(weights)
                elif scored:
                    tv_values=[value for value,_proof in scored]
                    evidence_confidence=sum(value[1] for value in tv_values)/len(tv_values)
                    evidence_score=sum(prior+value[1]*(value[0]-prior)
                                       for value in tv_values)/len(tv_values)
                else:
                    evidence_confidence=0.0; evidence_score=prior
                if self.config["aggregation"]=="hybrid":
                    # Use PeTTaChainer's merged inference STV as part of the
                    # decision while retaining calibrated CTV evidence from
                    # each rule present in the proof tree.  This is a fixed,
                    # auditable 80/20 semantic blend (not a fallback ranker).
                    canonical_score=(prior+inference_tv[1]*(inference_tv[0]-prior)
                                     if scored else prior)
                    confidence=0.8*evidence_confidence+0.2*inference_tv[1]
                    score=0.8*evidence_score+0.2*canonical_score
                    strength=(prior+(score-prior)/confidence
                              if confidence else prior)
                    tv=(max(0.0,min(1.0,strength)),confidence)
                    score_method="pettachainer_hybrid_base_rate_posterior"
                else:
                    confidence=evidence_confidence; score=evidence_score
                    strength=(prior+(score-prior)/confidence
                              if confidence else prior)
                    tv=(max(0.0,min(1.0,strength)),confidence)
                    score_method="proof_gated_base_rate_posterior"
            if feedback_scored:
                # Weighted point aggregation intentionally reconstructs mined
                # rule CTVs for auditability. Online feedback is different: a
                # dedicated PeTTa policy goal proves its zero-strength CTV.
                # Read that canonical proof so neither the rule strength nor
                # its effect is duplicated in Python as an opaque penalty.
                feedback_tv,feedback_proof=max(
                    feedback_scored,key=lambda item:(item[0][1],
                                                     abs(item[0][0]-prior))
                )
                tv=feedback_tv
                score=prior+tv[1]*(tv[0]-prior)
                score_method="pettachainer_live_feedback_revision"
            relational_refs=getattr(
                self,"_candidate_relational_proof_refs",{}
            ).get(_case,{})
            relational_scopes={
                field:_raw_attrs[field] for field in RELATIONAL_PROOF_FIELDS
                if _raw_attrs.get(field) in {"none","older","recent"}
            }
            row={"article":self.public_article(aid),"score":round(score,8),"stv":{"strength":tv[0],"confidence":tv[1]},
                         "inference_stv":{"strength":inference_tv[0],"confidence":inference_tv[1]},
                         "baseline":popularity[aid],"rules":fired,"proofs":proofs,"engine":"PeTTaChainer",
                         "aggregation":self.config["aggregation"],"score_method":score_method,
                         "tie_break":self._tie_break(self.article(aid)),
                         "relational_evidence":{
                             "scopes":relational_scopes,
                             "proof_ids":{
                                 key:list(value)
                                 for key,value in relational_refs.items()
                                 if value
                             },
                             "ledger":"immutable_snapshot_or_live_sidecar",
                         },
                         "feedback_evidence":{
                             "match":_raw_attrs.get(LIVE_NEGATIVE_FEATURE,"none"),
                             "rule_ids":feedback_rule_ids,
                             "proof_stv":{"strength":tv[0],"confidence":tv[1]}
                                 if feedback_rule_ids else None,
                         }}
            if include_context:
                row["_prepared_context"]=_raw_attrs
            rows.append(row)
        # Popularity is diagnostic metadata only. Ranking is proof-gated, with
        # the training-only editorial priors used only when proof values tie.
        # Article id remains the final deterministic tie-break.
        rows.sort(key=lambda r:(-r["score"],-r["stv"]["strength"],-r["stv"]["confidence"],
                                -r["tie_break"]["topic_prior"],
                                -r["tie_break"]["format_prior"],
                                -r["tie_break"]["subcategory_prior"],r["article"]["id"]))
        if apply_pairwise:
            rows=self._pairwise_rank(
                rows,specs,reasoner_timeout_sec=reasoner_timeout_sec
            )
        return rows if limit==0 else rows[:limit]

    def score(self,user,candidates=None,contexts=None,limit=None,
              include_context=False,apply_pairwise=True,cache_result=True):
        score_started=time.perf_counter()
        if user not in self.data["users"]: raise ValueError(f"unknown user: {user}")
        if candidates is None:
            candidates=self.default_candidates(user)
            if contexts is None: contexts=self.default_candidate_context(user)
        candidates=list(dict.fromkeys(str(aid) for aid in candidates))
        for aid in candidates: self.article(aid)
        contexts=contexts or {}; limit=self.config["top_k"] if limit is None else int(limit)
        # Proof references do not become miner predicates, but they are part
        # of a candidate's provenance identity. Without them, equal categorical
        # scopes from different history origins could reuse a cached row that
        # cites the wrong proof ledger records.
        proof_fields=tuple(RELATIONAL_PROOF_FIELDS.values())
        context_key_started=time.perf_counter(); context_key=[]
        for aid in candidates:
            candidate_context=contexts.get(aid) or {}
            references=self._relational_proof_references(candidate_context)
            context_key.append((
                aid,
                tuple((feature,candidate_context.get(feature))
                      for feature in CONTEXT_FEATURES),
                tuple((proof_field,references.get(proof_field,()))
                      for proof_field in proof_fields),
            ))
        context_key=tuple(context_key)
        context_key_seconds=time.perf_counter()-context_key_started
        key=(self.version,user,tuple(candidates),context_key,limit,
             bool(include_context),bool(apply_pairwise))
        if cache_result and key in self.feed_cache:
            self._feed_rank_cache_hits=(
                getattr(self,"_feed_rank_cache_hits",0)+1
            )
            cached=self.feed_cache.pop(key); self.feed_cache[key]=cached; return cached
        self._feed_rank_cache_misses=(
            getattr(self,"_feed_rank_cache_misses",0)+1
        )
        timeout_sec=self._serving_reasoner_timeout()
        candidate_started=time.perf_counter()
        specs=self._candidate_specs(user,candidates,contexts)
        candidate_seconds=time.perf_counter()-candidate_started
        point_started=time.perf_counter()
        self._ensure_candidate_specs(specs,timeout_sec=timeout_sec)
        groups,_calls=self._proofs_for_specs(specs,timeout_sec=timeout_sec)
        point_seconds=time.perf_counter()-point_started
        rank_started=time.perf_counter()
        ranked=self._rank(
            specs,groups,limit,reasoner_timeout_sec=timeout_sec,
            include_context=include_context,apply_pairwise=apply_pairwise,
        )
        rank_seconds=time.perf_counter()-rank_started
        if cache_result:
            self.feed_cache[key]=ranked
            while len(self.feed_cache)>256:
                self.feed_cache.popitem(last=False)
        self._last_live_score_profile={
            "candidates":len(candidates),
            "apply_pairwise":bool(apply_pairwise),
            "context_key_seconds":round(context_key_seconds,6),
            "candidate_preparation_seconds":round(candidate_seconds,6),
            "point_reasoning_seconds":round(point_seconds,6),
            "pair_ranking_seconds":round(rank_seconds,6),
            "total_seconds":round(time.perf_counter()-score_started,6),
            "point_reasoner_query_calls":_calls,
            "candidate_plan":{
                key:(round(value,6) if isinstance(value,float) else value)
                for key,value in getattr(
                    self,"_last_candidate_preparation_profile",{}
                ).items()
            },
            "pair_plan":({
                key:(round(value,6) if isinstance(value,float) else value)
                for key,value in getattr(
                    self,"_last_pair_plan_profile",{}
                ).items()
            } if apply_pairwise else {}),
        }
        return ranked

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
                    "conditional_llm_seed_only"}:
                raise ValueError(
                    "miner_strategy must be fixed_combinations, target_aware "
                    "conditional_llm or conditional_llm_seed_only"
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
        if (effective_strategy in {"conditional_llm","conditional_llm_seed_only"}
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
        if (effective_strategy in {"conditional_llm","conditional_llm_seed_only"}
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
                and effective_strategy!="conditional_llm_seed_only"):
            raise ValueError(
                "conditional_effective_backoff requires "
                "miner_strategy conditional_llm_seed_only"
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

    @staticmethod
    def _bounded_candidates(case,cap,seed):
        candidates=list(case["candidates"])
        if not cap or len(candidates)<=cap: return candidates
        stable=f"{seed}:{case.get('id')}".encode(); local=random.Random(int.from_bytes(hashlib.sha256(stable).digest()[:8],"big"))
        selected=set(local.sample(candidates,min(len(candidates),cap)))
        return [aid for aid in candidates if aid in selected]

    @staticmethod
    def _bootstrap_interval_raw(values,seed,repetitions=1000):
        if not values: return None
        local=random.Random(seed); count=len(values); estimates=[]
        for _ in range(repetitions):
            estimates.append(sum(values[local.randrange(count)] for _ in range(count))/count)
        estimates.sort()
        return (estimates[int(0.025*(repetitions-1))],
                estimates[int(0.975*(repetitions-1))])

    @staticmethod
    def _bootstrap_interval(values,seed,repetitions=1000):
        interval=Lab._bootstrap_interval_raw(values,seed,repetitions)
        return ([round(interval[0],4),round(interval[1],4)]
                if interval is not None else None)

    def _balanced_margin_probability_evaluator(self):
        """Build the exact pre-tournament balanced-margin pair evaluator."""
        if (self.config.get("pair_aggregation")!="proof_margin"
                or self.config.get("pair_family_fusion")!="balanced_margin"):
            return None
        dependency_families={}
        for rule in self.pair_rules:
            dependency_id=rule.get("dependency_id",rule["id"])
            family=self._pair_rule_family(rule)
            if (family=="text_semantic"
                    or dependency_id not in dependency_families):
                dependency_families[dependency_id]=family
        # Reliability runs before the timed production tournament. It must not
        # pre-populate the live margin cache and turn a cold ranking into a
        # cache-hit measurement.
        reliability_margin_cache={}

        def evaluate(forward_case,reverse_case,forward_proofs,reverse_proofs):
            forward_margins=self._proof_dependency_margins(
                forward_case,forward_proofs,cache=reliability_margin_cache
            )
            reverse_margins=self._proof_dependency_margins(
                reverse_case,reverse_proofs,cache=reliability_margin_cache
            )
            family_totals={}; family_counts={}
            for dependency_id in sorted(
                    set(forward_margins)|set(reverse_margins)):
                dependency_margin=(
                    forward_margins.get(dependency_id,0.0)
                    -reverse_margins.get(dependency_id,0.0)
                )
                family=dependency_families.get(
                    dependency_id,"structured_symbolic"
                )
                family_totals[family]=(
                    family_totals.get(family,0.0)+dependency_margin
                )
                family_counts[family]=family_counts.get(family,0)+1
            if not family_totals:
                return None
            active_family_margins=[
                total/family_counts[family]
                for family,total in family_totals.items()
            ]
            fused_margin=(
                sum(active_family_margins)/len(active_family_margins)
            )
            return self._logistic(fused_margin)

        return evaluate

    @staticmethod
    def _pair_proof_reliability(
            prepared,planned,proof_map,bins=10,seed=7,
            comparison_probability=None):
        """Held-out reliability of individual active ``PairSignal`` proofs.

        The final tournament is an ordinal ranker and is not presented as a
        calibrated probability.  This diagnostic instead evaluates the only
        probability-bearing objects in that path: returned pair-proof STVs,
        converted to ``0.5 + confidence * (strength - 0.5)``.  Every source
        impression has total weight one, so mirrored orientations and large
        slates cannot masquerade as independent samples.
        """
        bins=int(bins)
        if bins<2 or bins>100:
            raise ValueError("pair proof reliability bins must be in [2, 100]")
        by_impression={}; aggregated_by_impression={}
        eligible_roots=proved_roots=0
        eligible_comparisons=active_aggregated_comparisons=0
        for (case,_specs,_start,_end),(rows,plan) in zip(prepared,planned):
            relevant=set(case["relevant"])
            identity=str(case.get("source_impression_id") or case.get("id"))
            observations=[]; aggregated_observations=[]
            comparisons,_pair_specs=plan
            for left_index,right_index,forward_case,reverse_case in comparisons:
                left_relevant=(rows[left_index]["article"]["id"] in relevant)
                right_relevant=(rows[right_index]["article"]["id"] in relevant)
                # The mined target is defined only for clicked-versus-
                # nonclicked pairs from one closed impression.
                if left_relevant==right_relevant:
                    continue
                eligible_comparisons+=1
                forward_proofs=proof_map.get(forward_case,())
                reverse_proofs=proof_map.get(reverse_case,())
                for pair_case,target,proofs in (
                    (forward_case,left_relevant,forward_proofs),
                    (reverse_case,right_relevant,reverse_proofs),
                ):
                    eligible_roots+=1
                    proved_roots+=bool(proofs)
                    for proof in proofs:
                        strength,confidence=proof_tv(proof)
                        posterior=0.5+confidence*(strength-0.5)
                        observations.append((
                            max(0.0,min(1.0,posterior)),
                            max(0.0,min(1.0,strength)),
                            1.0 if target else 0.0,
                        ))
                if comparison_probability is not None:
                    probability=comparison_probability(
                        forward_case,reverse_case,
                        forward_proofs,reverse_proofs,
                    )
                    if probability is not None:
                        active_aggregated_comparisons+=1
                        probability=max(0.0,min(1.0,float(probability)))
                        aggregated_observations.append((
                            probability,probability,
                            1.0 if left_relevant else 0.0,
                        ))
            if observations:
                by_impression.setdefault(identity,[]).extend(observations)
            if aggregated_observations:
                aggregated_by_impression.setdefault(identity,[]).extend(
                    aggregated_observations
                )
        evaluated_probability={
            "name":"active_pair_signal_left_clicked_probability",
            "definition":"q = 0.5 + confidence * (strength - 0.5)",
            "event":"the left candidate is clicked",
            "conditioning":(
                "a held-out same-impression pair contains exactly one clicked "
                "and one exposed nonclicked candidate, and this individual "
                "PairSignal rule is active"
            ),
            "unit":"one returned active PairSignal proof",
            "ordinal_ranking_used":False,
        }
        if not by_impression:
            return {
                "status":"no_active_pair_proofs",
                "evaluated_probability":evaluated_probability,
                "target_population":(
                    "held-out same-impression clicked-versus-nonclicked "
                    "orientations only"
                ),
                "eligible_direction_roots":eligible_roots,
                "proved_direction_roots":proved_roots,
                "aggregated_balanced_margin_pair_preference":{
                    "status":(
                        "no_active_balanced_margin_comparisons"
                        if comparison_probability is not None
                        else "not_applicable"
                    ),
                    "eligible_comparisons":eligible_comparisons,
                    "active_comparisons":active_aggregated_comparisons,
                },
            }

        def reliability(groups,column=None,constant=None):
            if (column is None)==(constant is None):
                raise ValueError(
                    "reliability requires exactly one prediction column or constant"
                )
            brier=[]; log_loss=[]; buckets=[[] for _ in range(bins)]
            for observations in groups.values():
                local_brier=[]; local_log=[]
                weight=1.0/len(observations)
                for row in observations:
                    prediction=(float(constant) if constant is not None
                                else row[column])
                    target=row[2]
                    local_brier.append((prediction-target)**2)
                    bounded=max(1e-12,min(1.0-1e-12,prediction))
                    local_log.append(-(
                        target*math.log(bounded)
                        +(1.0-target)*math.log(1.0-bounded)
                    ))
                    bucket=min(bins-1,int(prediction*bins))
                    buckets[bucket].append((prediction,target,weight))
                brier.append(math.fsum(local_brier)/len(local_brier))
                log_loss.append(math.fsum(local_log)/len(local_log))
            total_weight=float(len(groups))
            table=[]; ece=0.0
            for index,values in enumerate(buckets):
                mass=math.fsum(value[2] for value in values)
                if not mass:
                    continue
                mean_prediction=math.fsum(
                    value[0]*value[2] for value in values
                )/mass
                observed_rate=math.fsum(
                    value[1]*value[2] for value in values
                )/mass
                ece+=(mass/total_weight)*abs(
                    mean_prediction-observed_rate
                )
                table.append({
                    "lower":index/bins,"upper":(index+1)/bins,
                    "impression_weight_mass":mass,
                    "mean_prediction":mean_prediction,
                    "observed_target_rate":observed_rate,
                    "absolute_gap":abs(mean_prediction-observed_rate),
                })
            return {
                "macro_impression_brier":math.fsum(brier)/len(brier),
                "macro_impression_log_loss":math.fsum(log_loss)/len(log_loss),
                "equal_impression_weighted_ece":ece,
                "reliability_bins":table,
            },{"brier":brier,"log_loss":log_loss}

        empirical_active_proof_rate=math.fsum(
            math.fsum(row[2] for row in observations)/len(observations)
            for observations in by_impression.values()
        )/len(by_impression)
        shrunk,shrunk_components=reliability(by_impression,column=0)
        raw_strength,raw_strength_components=reliability(by_impression,column=1)
        balanced_constant,balanced_components=reliability(
            by_impression,constant=0.5
        )
        empirical_constant,empirical_components=reliability(
            by_impression,
            constant=empirical_active_proof_rate
        )

        def paired_deltas(left,right,seed_offset):
            brier=[left_value-right_value
                   for left_value,right_value in zip(
                       left["brier"],right["brier"]
                   )]
            log_loss=[left_value-right_value
                      for left_value,right_value in zip(
                          left["log_loss"],right["log_loss"]
                      )]
            return {
                "macro_impression_brier":math.fsum(brier)/len(brier),
                "macro_impression_brier_95_ci":Lab._bootstrap_interval(
                    brier,int(seed)^seed_offset
                ),
                "macro_impression_log_loss":math.fsum(log_loss)/len(log_loss),
                "macro_impression_log_loss_95_ci":Lab._bootstrap_interval(
                    log_loss,int(seed)^(seed_offset<<1)
                ),
                "interpretation":"negative favors the first named predictor",
            }

        def descriptive_deltas(left,right):
            brier=[left_value-right_value
                   for left_value,right_value in zip(
                       left["brier"],right["brier"]
                   )]
            log_loss=[left_value-right_value
                      for left_value,right_value in zip(
                          left["log_loss"],right["log_loss"]
                      )]
            return {
                "macro_impression_brier":math.fsum(brier)/len(brier),
                "macro_impression_log_loss":math.fsum(log_loss)/len(log_loss),
                "inference":(
                    "point estimate only; no confidence interval because the "
                    "empirical constant was fitted on this same cohort"
                ),
                "interpretation":"negative favors the first named predictor",
            }

        shrunk_minus_raw=paired_deltas(
            shrunk_components,raw_strength_components,0xB13E2
        )
        shrunk_minus_balanced=paired_deltas(
            shrunk_components,balanced_components,0xB4500
        )
        raw_minus_balanced=paired_deltas(
            raw_strength_components,balanced_components,0xBA500
        )
        shrunk_minus_empirical=descriptive_deltas(
            shrunk_components,empirical_components
        )
        aggregated={
            "status":"not_applicable",
            "reason":(
                "requires pair_aggregation=proof_margin and "
                "pair_family_fusion=balanced_margin"
            ),
            "eligible_comparisons":eligible_comparisons,
        }
        if comparison_probability is not None:
            aggregated={
                "status":"no_active_balanced_margin_comparisons",
                "eligible_comparisons":eligible_comparisons,
                "active_comparisons":active_aggregated_comparisons,
            }
            if aggregated_by_impression:
                empirical_aggregate_rate=math.fsum(
                    math.fsum(row[2] for row in observations)/len(observations)
                    for observations in aggregated_by_impression.values()
                )/len(aggregated_by_impression)
                aggregate_probability,aggregate_components=reliability(
                    aggregated_by_impression,column=0
                )
                aggregate_balanced,aggregate_balanced_components=reliability(
                    aggregated_by_impression,constant=0.5
                )
                aggregate_empirical,aggregate_empirical_components=reliability(
                    aggregated_by_impression,constant=empirical_aggregate_rate
                )
                aggregated={
                    "status":"computed",
                    "evaluated_probability":{
                        "name":"balanced_margin_left_clicked_probability",
                        "definition":(
                            "q_pair = sigmoid(mean over active families of "
                            "the mean forward-minus-reverse dependency margin)"
                        ),
                        "event":"the left candidate is clicked",
                        "conditioning":(
                            "a held-out same-impression pair contains exactly "
                            "one clicked and one exposed nonclicked candidate, "
                            "and at least one dependency proof is active"
                        ),
                        "unit":"one unordered candidate comparison",
                        "ordinal_ranking_used":False,
                        "calibration_claim":(
                            "probability-shaped pre-tournament diagnostic; "
                            "reliability is measured, not assumed"
                        ),
                    },
                    "metric_weighting":(
                        "macro mean of within-impression active-comparison means"
                    ),
                    "impressions":len(aggregated_by_impression),
                    "eligible_comparisons":eligible_comparisons,
                    "active_comparisons":active_aggregated_comparisons,
                    "active_comparison_coverage":(
                        active_aggregated_comparisons/eligible_comparisons
                        if eligible_comparisons else 0.0
                    ),
                    "confidence_aware_probability":aggregate_probability,
                    "constant_probability_baselines":{
                        "constant_0_5_neutral_pair_preference":{
                            "probability":0.5,
                            "basis":"neutral left-versus-right pair preference",
                            **aggregate_balanced,
                        },
                        "constant_empirical_left_click_rate":{
                            "probability":empirical_aggregate_rate,
                            "basis":(
                                "equal-impression left-click target rate among "
                                "the same active comparisons"
                            ),
                            "role":(
                                "descriptive same-cohort fitted reference, not "
                                "a deployable or independently estimated baseline"
                            ),
                            **aggregate_empirical,
                        },
                    },
                    "paired_deltas":{
                        "confidence_aware_minus_constant_0_5":paired_deltas(
                            aggregate_components,aggregate_balanced_components,
                            0xA6650
                        ),
                    },
                    "descriptive_same_cohort_deltas":{
                        "confidence_aware_minus_empirical_constant":descriptive_deltas(
                            aggregate_components,aggregate_empirical_components,
                        ),
                    },
                }
        return {
            "status":"computed",
            "scope":"individual_active_pair_proof_stvs_not_final_tournament",
            "evaluated_probability":evaluated_probability,
            "target_population":(
                "held-out same-impression clicked-versus-nonclicked "
                "orientations only; not a universal arbitrary-pair preference"
            ),
            "sampling_unit":(
                "one source impression total weight; all mirrored orientations "
                "and active proof channels share that unit mass"
            ),
            "metric_weighting":(
                "macro mean of within-impression proof-observation means"
            ),
            "impressions":len(by_impression),
            "proof_observations":sum(map(len,by_impression.values())),
            "eligible_direction_roots":eligible_roots,
            "proved_direction_roots":proved_roots,
            "direction_root_coverage":(
                proved_roots/eligible_roots if eligible_roots else 0.0
            ),
            "confidence_shrunk_posterior":shrunk,
            "raw_strength_without_confidence":raw_strength,
            "constant_probability_baselines":{
                "constant_0_5_balanced_pair_target":{
                    "probability":0.5,
                    "basis":(
                        "protocol prior before conditioning on active-proof "
                        "availability; both pair orientations are generated"
                    ),
                    **balanced_constant,
                },
                "constant_empirical_active_proof_rate":{
                    "probability":empirical_active_proof_rate,
                    "basis":(
                        "equal-impression target rate among these same held-out "
                        "active-proof observations"
                    ),
                    "role":(
                        "descriptive same-cohort fitted reference, not a deployable "
                        "or independently estimated baseline"
                    ),
                    **empirical_constant,
                },
            },
            "paired_deltas":{
                "confidence_shrunk_minus_raw_strength":shrunk_minus_raw,
                "confidence_shrunk_minus_constant_0_5":shrunk_minus_balanced,
                "raw_strength_minus_constant_0_5":raw_minus_balanced,
            },
            "descriptive_same_cohort_deltas":{
                "confidence_shrunk_minus_empirical_constant":(
                    shrunk_minus_empirical
                ),
            },
            "aggregated_balanced_margin_pair_preference":aggregated,
            "brier_delta_shrunk_minus_raw_strength":(
                shrunk_minus_raw["macro_impression_brier"]
            ),
            "brier_delta_95_ci":shrunk_minus_raw[
                "macro_impression_brier_95_ci"
            ],
            "interpretation":(
                "Lower Brier/log loss/ECE is better. This is a held-out "
                "reliability check, not training-population calibration and "
                "not a probability claim for the ordinal final rank score."
            ),
        }

    def benchmark(self,config):
        if self._online_events:
            raise ValueError(
                "reload the dataset before benchmarking: live feedback is isolated "
                "from the immutable offline evaluation protocol"
            )
        matched_direct_requested=config.get(
            "matched_direct_reasoner_ablation",False
        )
        if not isinstance(matched_direct_requested,bool):
            raise ValueError(
                "matched_direct_reasoner_ablation must be a boolean"
            )
        force_cold=config.get(
            "force_cold_proof_cache",config.get("force_cold",False)
        )
        if not isinstance(force_cold,bool):
            raise ValueError("force_cold_proof_cache must be a boolean")
        started=time.perf_counter()
        previous_config=self.config.copy()
        mining=None
        try:
            self.configure(config)
            remine=config.get("remine",True) is not False
            stale_keys=sorted(
                key for key in MINING_CONFIG_KEYS
                if key in config and previous_config.get(key)!=self.config.get(key)
            )
            if stale_keys and not remine:
                raise ValueError(
                    "remine=false cannot change mining configuration: "
                    +", ".join(stale_keys)
                )
            if remine:
                mining=self.mine()
        except Exception:
            # ``mine`` atomically retains the previous worker/rule metadata on
            # failure. Restore its matching configuration as well; otherwise a
            # failed benchmark could leave serving facts projected through a
            # feature profile that the still-live scorer was never compiled for.
            self.configure(previous_config)
            raise
        proof_cache_entries_before_clear={
            "point":len(self._proof_cache),
            "point_channel":len(self._point_channel_proof_cache),
            "pair_case":len(self._pair_proof_cache),
            "pair_channel":len(self._pair_channel_proof_cache),
        }
        if force_cold:
            # This is a proof-result cold run, not a model-compilation or fact-
            # materialization cold start. Keeping the identical worker/facts
            # guarantees accuracy is unchanged while reasoner cache latency is
            # measured honestly.
            self._proof_cache.clear()
            self._point_channel_proof_cache.clear()
            self._pair_proof_cache.clear()
            self._pair_channel_proof_cache.clear()
            self._pair_margin_cache.clear()
            self._pair_proof_origins.clear()
        proof_cache_entries_at_evaluation_start={
            "point":len(self._proof_cache),
            "point_channel":len(self._point_channel_proof_cache),
            "pair_case":len(self._pair_proof_cache),
            "pair_channel":len(self._pair_channel_proof_cache),
        }
        model_preparation_seconds=time.perf_counter()-started
        evaluation_started=time.perf_counter()
        benchmark_timeout_sec=self._benchmark_reasoner_timeout()
        tests=self.evaluation_cases()
        if not tests: raise ValueError("dataset has no labeled evaluation impressions")
        prepared=[]; all_specs=[]
        cap=max(0,int(config.get("max_candidates",self.config["max_candidates"])))
        seed=int(config.get("random_seed",self.config["random_seed"]))
        eval_case_limit=max(0,int(config.get("eval_case_limit",0)))
        if eval_case_limit and len(tests)>eval_case_limit:
            indexed=list(enumerate(tests))
            selected=sorted(indexed,key=lambda item:int.from_bytes(hashlib.sha256(
                f'{seed}\0eval\0{item[1].get("source_impression_id") or item[1].get("id")}'.encode()
            ).digest(),"big"))[:eval_case_limit]
            tests=[case for _index,case in sorted(selected)]
        supplied_candidate_count=0; admitted_candidate_count=0
        supplied_relevant_count=0; admitted_relevant_count=0
        complete_relevant_slates=0; relevant_slates=0
        candidate_preparation_started=time.perf_counter()
        candidate_preparation_profile=Counter()
        for case in tests:
            supplied_candidates=list(case["candidates"])
            candidates=self._bounded_candidates(case,cap,seed)
            relevant_ids=set(map(str,case.get("relevant",())))
            supplied_relevant=relevant_ids.intersection(
                map(str,supplied_candidates)
            )
            admitted_relevant=supplied_relevant.intersection(
                map(str,candidates)
            )
            supplied_candidate_count+=len(supplied_candidates)
            admitted_candidate_count+=len(candidates)
            supplied_relevant_count+=len(supplied_relevant)
            admitted_relevant_count+=len(admitted_relevant)
            if supplied_relevant:
                relevant_slates+=1
                complete_relevant_slates+=(
                    admitted_relevant==supplied_relevant
                )
            contexts=case.get("candidate_context",{})
            specs=self._candidate_specs(case["user"],candidates,contexts)
            for key,value in getattr(
                    self,"_last_candidate_preparation_profile",{}).items():
                if key!="candidates": candidate_preparation_profile[key]+=value
            start=len(all_specs); all_specs.extend(specs)
            prepared.append((case,specs,start,len(all_specs)))
        candidate_preparation_seconds=(
            time.perf_counter()-candidate_preparation_started
        )
        point_reasoning_started=time.perf_counter()
        self._point_query_calls=0; self._point_query_roots=0
        self._point_pruned_query_roots=0
        self._point_channel_activations=0
        self._point_reused_channel_activations=0
        self._last_point_completeness={}
        self._ensure_candidate_specs(
            all_specs,timeout_sec=benchmark_timeout_sec
        )
        groups,reasoner_calls=self._proofs_for_specs(
            all_specs,timeout_sec=benchmark_timeout_sec
        )
        point_cache_stats=dict(self._last_point_cache_stats)
        point_reasoning_seconds=time.perf_counter()-point_reasoning_started
        self._pair_query_calls=0; self._pair_query_roots=0
        self._pair_pruned_query_roots=0; self._pair_channel_activations=0
        self._pair_reused_channel_activations=0; ranked=[]
        unordered_pair_comparisons=0; oriented_pair_cases=0
        maximum_slate_candidates=max(
            (len(specs) for _case,specs,_start,_end in prepared),default=0
        )
        maximum_slate_comparisons=0
        pair_planning_seconds=0.0; pair_reasoning_seconds=0.0
        parity_audit_seconds=0.0; rank_aggregation_seconds=0.0
        pair_reliability_seconds=0.0
        direct_ablation_seconds=0.0; direct_ranked=None; direct_planned=[]
        matched_direct_reasoner_ablation={
            "requested":matched_direct_requested,
            "status":"not_requested",
            "normal_ranking_path":"PeTTaChainer proofs",
            "direct_path_used_for_serving":False,
            "live_proof_margin_cache_mutated":False,
            "interpretation":(
                "Opt-in matched scoring-output diagnostic only; normal benchmark "
                "ranking remains PeTTaChainer and has no direct fallback."
            ),
        }
        reasoner_semantic_parity={
            "status":"not_applicable",
            "interpretation":(
                "Matched direct-versus-PeTTa semantic parity only; this is not "
                "an accuracy causal ablation or a unique-AUC estimate."
            ),
        }
        heldout_pair_proof_reliability={"status":"not_applicable"}
        pair_plan_profile=Counter()
        pair_materialization_profile={}
        pair_reasoning_profile={}
        point_row_preparation_seconds=0.0
        pair_budget_accounting_seconds=0.0
        all_pair_specs=[]
        if self.config["ranking_mode"]=="pairwise" and self.pair_rules:
            pair_planning_started=time.perf_counter()
            planned=[]
            for case,specs,start,end in prepared:
                point_rows_started=time.perf_counter()
                rows=self._rank(specs,groups[start:end],0,apply_pairwise=False)
                point_row_preparation_seconds+=(
                    time.perf_counter()-point_rows_started
                )
                plan=self._pairwise_plan(rows,specs)
                for key,value in getattr(self,"_last_pair_plan_profile",{}).items():
                    if isinstance(value,(int,float)) and not isinstance(value,bool):
                        aggregate_key=(
                            "sum_unique_activation_cases_per_slate"
                            if key=="unique_activation_cases_in_slate" else key
                        )
                        pair_plan_profile[aggregate_key]+=value
                budget_started=time.perf_counter()
                next_total=unordered_pair_comparisons+len(plan[0])
                total_limit=int(self.config["max_total_pair_comparisons"])
                if next_total>total_limit:
                    raise ValueError(
                        "benchmark pairwise comparison budget exceeded: "
                        f"more than {total_limit} unordered pairs across the "
                        "evaluation cohort; reduce eval_case_limit, candidate "
                        "slates, or pairwise_opponents"
                    )
                planned.append((rows,plan)); all_pair_specs.extend(plan[1])
                unordered_pair_comparisons=next_total
                maximum_slate_comparisons=max(
                    maximum_slate_comparisons,len(plan[0])
                )
                pair_budget_accounting_seconds+=(
                    time.perf_counter()-budget_started
                )
            oriented_pair_cases=len(all_pair_specs)
            pair_planning_seconds=time.perf_counter()-pair_planning_started
            pair_reasoning_started=time.perf_counter()
            self._ensure_pair_specs(
                all_pair_specs,timeout_sec=benchmark_timeout_sec
            )
            pair_materialization_profile=dict(getattr(
                self,"_last_pair_materialization_profile",{}
            ))
            pair_proof_map,_pair_calls=self._proofs_for_pair_specs(
                all_pair_specs,timeout_sec=benchmark_timeout_sec
            )
            pair_reasoning_profile=dict(getattr(
                self,"_last_pair_reasoning_profile",{}
            ))
            pair_cache_stats=dict(self._last_pair_cache_stats)
            pair_reasoning_seconds=time.perf_counter()-pair_reasoning_started
            pair_reliability_started=time.perf_counter()
            heldout_pair_proof_reliability=self._pair_proof_reliability(
                prepared,planned,pair_proof_map,seed=seed,
                comparison_probability=(
                    self._balanced_margin_probability_evaluator()
                ),
            )
            pair_reliability_seconds=(
                time.perf_counter()-pair_reliability_started
            )
            if matched_direct_requested:
                # Preserve the exact pre-tournament point rows and pair plans.
                # The normal PeTTa ranking below sorts/mutates its row objects.
                direct_planned=[(copy.deepcopy(rows),plan)
                                for rows,plan in planned]
            parity_audit_started=time.perf_counter()
            # A bounded matched audit answers the criticism that this shallow
            # topology may be reproducible by a direct rule table.  A separate
            # disposable worker receives fresh categorical facts and is asked
            # for every case/channel PairSignal root; it does not reuse the
            # optimized host activation path or its proof caches.  A match
            # demonstrates semantic fidelity only and deliberately does not
            # claim that PeTTa alone caused an AUC lift.
            from ..evaluation.reasoner_parity import (
                UnsupportedParityTopology,
                evaluate_pair_reasoner_parity,
            )
            unique_pair_specs={}
            for case,attrs in all_pair_specs:
                unique_pair_specs.setdefault(case,(case,attrs))
            active_candidates=[]; inactive_parity_specs=[]
            for spec in unique_pair_specs.values():
                _case,attrs=spec
                activated={
                    rule["proof_channel_id"] for rule in self.pair_rules
                    if all(attrs.get(predicate)==value
                           for predicate,value in rule["premises"])
                }
                if activated:
                    active_candidates.append((spec,activated))
                else:
                    inactive_parity_specs.append(spec)
            # Prefer cases that add channel coverage.  The audit remains
            # bounded and reports sampled_match when the evaluation slate
            # cannot exercise every compiled channel.
            active_parity_specs=[]; covered_channels=set()
            for spec,activated in active_candidates:
                if activated.difference(covered_channels):
                    active_parity_specs.append(spec)
                    covered_channels.update(activated)
                    if len(active_parity_specs)>=96:
                        break
            selected_active={case for case,_attrs in active_parity_specs}
            active_parity_specs.extend(
                spec for spec,_activated in active_candidates
                if spec[0] not in selected_active
            )
            active_parity_specs=active_parity_specs[:96]
            parity_case_cap=min(
                128,max(1,32768//max(1,len(self.pair_rules)))
            )
            # Reserve one negative/non-activation case when available. This
            # keeps the independent audit inside its hard case×channel query
            # bound even for the largest accepted rule snapshot.
            if inactive_parity_specs and parity_case_cap>1:
                parity_specs=[
                    *active_parity_specs[:parity_case_cap-1],
                    inactive_parity_specs[0],
                ]
            else:
                parity_specs=active_parity_specs[:parity_case_cap]
            if len(parity_specs)<parity_case_cap:
                selected={case for case,_attrs in parity_specs}
                parity_specs.extend(
                    spec for spec in unique_pair_specs.values()
                    if spec[0] not in selected
                )
                parity_specs=parity_specs[:parity_case_cap]
            try:
                if not parity_specs or not active_parity_specs:
                    raise UnsupportedParityTopology(
                        "no active evaluation pair channel is available"
                    )
                parity=evaluate_pair_reasoner_parity(
                    self,parity_specs,max_cases=parity_case_cap,
                    max_rules=min(1024,max(256,len(self.pair_rules))),
                )
                reasoner_semantic_parity={
                    "status":parity["status"],
                    "parity_scope":parity["parity_scope"],
                    "full_model_semantic_parity":parity[
                        "full_model_semantic_parity"
                    ],
                    "sampled_semantic_parity":parity[
                        "sampled_semantic_parity"
                    ],
                    "report_kind":parity["report_kind"],
                    "interpretation":parity["interpretation"],
                    "supported_topology":parity["supported_topology"],
                    "model_sha256":parity["model_sha256"],
                    "summary":parity["summary"],
                    "errors":parity["errors"],
                }
            except UnsupportedParityTopology as exc:
                reasoner_semantic_parity={
                    "status":"unsupported",
                    "interpretation":(
                        "No direct semantic-parity claim was made; this is not "
                        "an accuracy causal ablation or unique-AUC estimate."
                    ),
                    "reason":str(exc),
                }
            parity_audit_seconds=time.perf_counter()-parity_audit_started
            rank_aggregation_started=time.perf_counter()
            for (case,specs,_start,_end),(rows,plan) in zip(prepared,planned):
                ranked.append(self._pairwise_rank(rows,specs,plan,pair_proof_map))
            rank_aggregation_seconds=time.perf_counter()-rank_aggregation_started
            if matched_direct_requested:
                direct_ablation_started=time.perf_counter()
                if (self.config["pair_aggregation"]!="proof_margin"
                        or self.config["aggregation"]!="weighted"):
                    matched_direct_reasoner_ablation.update(
                        status="unsupported",
                        reason=(
                            "exact direct scoring-output reconstruction supports "
                            "weighted point aggregation and compiler-issued "
                            "isolated proof_margin pair channels only"
                        ),
                    )
                else:
                    try:
                        from ..evaluation.reasoner_parity import (
                            DirectPairReconstructor,
                            DirectPointReconstructor,
                            UnsupportedParityTopology,
                        )
                        direct_point=DirectPointReconstructor(
                            self,max_rules=min(
                                1024,max(256,len(self.mined_rules))
                            ),
                        )
                        direct_pair=DirectPairReconstructor(
                            self,max_rules=min(
                                1024,max(256,len(self.pair_rules))
                            ),
                        )
                        direct_point_groups=[]; direct_point_by_case={}
                        for _aid,candidate_case,attrs,_raw_attrs in all_specs:
                            reconstructed=direct_point.proof_rows(
                                candidate_case,attrs
                            )
                            previous=direct_point_by_case.setdefault(
                                candidate_case,reconstructed
                            )
                            if previous!=reconstructed:
                                raise RuntimeError(
                                    "one normalized candidate case produced "
                                    "conflicting direct point activations"
                                )
                            direct_point_groups.append(reconstructed)
                        direct_proof_map={}
                        for pair_case,attrs in all_pair_specs:
                            reconstructed=direct_pair.proof_rows(
                                pair_case,attrs
                            )
                            previous=direct_proof_map.setdefault(
                                pair_case,reconstructed
                            )
                            if previous!=reconstructed:
                                raise RuntimeError(
                                    "one normalized pair case produced "
                                    "conflicting direct activations"
                                )
                        direct_ranked=[]
                        point_order_matches=0
                        point_signature_matches=0
                        point_signatures_compared=0
                        margin_cache_keys_before=set(self._pair_margin_cache)
                        try:
                            for ((case,specs,start,end),
                                 (petta_point_rows,plan)) in zip(
                                     prepared,direct_planned):
                                rows=self._rank(
                                    specs,direct_point_groups[start:end],0,
                                    apply_pairwise=False,
                                )
                                for row in rows:
                                    row["engine"]="DirectPointReconstructor"
                                petta_index_to_id=[
                                    row["article"]["id"]
                                    for row in petta_point_rows
                                ]
                                direct_index_by_id={
                                    row["article"]["id"]:index
                                    for index,row in enumerate(rows)
                                }
                                if (len(direct_index_by_id)!=len(rows)
                                        or set(petta_index_to_id)
                                        !=set(direct_index_by_id)):
                                    raise RuntimeError(
                                        "direct point reconstruction changed "
                                        "the prepared candidate slate"
                                    )
                                direct_point_order=[
                                    row["article"]["id"] for row in rows
                                ]
                                point_order_matches+=(
                                    petta_index_to_id==direct_point_order
                                )
                                petta_point_by_id={
                                    row["article"]["id"]:row
                                    for row in petta_point_rows
                                }
                                direct_point_by_id={
                                    row["article"]["id"]:row for row in rows
                                }
                                for article_id in petta_index_to_id:
                                    point_signature_matches+=(
                                        self._pointwise_signature(
                                            petta_point_by_id[article_id]
                                        )==self._pointwise_signature(
                                            direct_point_by_id[article_id]
                                        )
                                    )
                                    point_signatures_compared+=1
                                comparisons,pair_specs=plan
                                remapped_comparisons=[
                                    (
                                        direct_index_by_id[
                                            petta_index_to_id[left_index]
                                        ],
                                        direct_index_by_id[
                                            petta_index_to_id[right_index]
                                        ],
                                        forward_case,reverse_case,
                                    )
                                    for (left_index,right_index,forward_case,
                                         reverse_case) in comparisons
                                ]
                                direct_ranked.append(
                                    self._pairwise_rank(
                                        rows,specs,
                                        (remapped_comparisons,pair_specs),
                                        direct_proof_map,
                                    )
                                )
                        finally:
                            # Diagnostic records must not accumulate in the
                            # live scorer's PeTTa proof-margin cache.
                            for cache_key in (
                                set(self._pair_margin_cache)
                                .difference(margin_cache_keys_before)
                            ):
                                self._pair_margin_cache.pop(cache_key,None)
                        matched_direct_reasoner_ablation.update(
                            status="computed",
                            scope="full_point_and_pair_scoring_output_path",
                            supported_topologies={
                                "point":direct_point.supported_topology,
                                "pair":direct_pair.supported_topology,
                            },
                            model_sha256=hashlib.sha256(
                                (direct_point.model_sha256+":"
                                 +direct_pair.model_sha256).encode("utf-8")
                            ).hexdigest(),
                            point_model_sha256=direct_point.model_sha256,
                            pair_model_sha256=direct_pair.model_sha256,
                            reconstructed_point_cases=len(
                                direct_point_by_case
                            ),
                            reconstructed_pair_cases=len(direct_proof_map),
                            compiled_point_rules=len(direct_point.channels),
                            compiled_pair_channels=len(direct_pair.channels),
                            exact_point_slate_order_matches=(
                                point_order_matches
                            ),
                            exact_point_slate_order_match_rate=(
                                point_order_matches/len(prepared)
                                if prepared else None
                            ),
                            exact_point_signature_matches=(
                                point_signature_matches
                            ),
                            point_signatures_compared=(
                                point_signatures_compared
                            ),
                            exact_point_signature_match_rate=(
                                point_signature_matches
                                /point_signatures_compared
                                if point_signatures_compared else None
                            ),
                            interpretation=(
                                "Matched full-scoring-output ablation for "
                                "the supported shallow point and pair topologies: "
                                "candidate slates, comparison edges/orientations, "
                                "host aggregation, proof-margin transforms, "
                                "family fusion and tie policy are identical. "
                                "PeTTa-returned point and pair activations/STVs "
                                "are replaced by lazy direct reconstruction. The "
                                "normal result remains PeTTaChainer; the direct "
                                "path is diagnostic and never a fallback."
                            ),
                        )
                    except UnsupportedParityTopology as exc:
                        matched_direct_reasoner_ablation.update(
                            status="unsupported",reason=str(exc)
                        )
                direct_ablation_seconds=(
                    time.perf_counter()-direct_ablation_started
                )
        else:
            pair_cache_stats={
                "case_root_requests":0,"case_root_hits":0,
                "case_root_misses":0,"channel_root_requests":0,
                "channel_root_hits":0,"channel_root_misses":0,
                "requests":0,"hits":0,"misses":0,
            }
            rank_aggregation_started=time.perf_counter()
            ranked=[self._rank(specs,groups[start:end],0)
                    for _case,specs,start,end in prepared]
            rank_aggregation_seconds=time.perf_counter()-rank_aggregation_started
            if matched_direct_requested:
                matched_direct_reasoner_ablation.update(
                    status="unsupported",
                    reason=(
                        "matched direct point-and-pair ablation requires an active "
                        "pairwise model"
                    ),
                )

        evaluation_relational_activation_audit=(
            self._relational_evaluation_activation_audit(
                all_specs,all_pair_specs
            )
        )
        metric_started=time.perf_counter()
        hits=mrr=ndcg5=ndcg10=auc_total=auc_primary_total=0.0
        auc_values=[]; auc_proof_values=[]; auc_primary_values=[]; auc_pointwise_values=[]
        auc_popularity_values=[]
        auc_per_impression=[]
        auc_cases=proof_candidates=0; missing_positive=0
        pair_coverage_total=directional_coverage_total=0.0
        score_counts=Counter(); ranking_counts=Counter(); proof_ranking_counts=Counter()
        for case_index,((case,specs,_start,_end),rows) in enumerate(zip(prepared,ranked)):
            score_counts.update(row["score"] for row in rows)
            ranking_counts.update(self._ranking_signature(row) for row in rows)
            proof_ranking_counts.update(self._proof_ranking_signature(row) for row in rows)
            proof_candidates+=sum(bool(row["proofs"]) for row in rows)
            pair_coverage_total+=sum(row.get("pairwise_proof_coverage",0.0) for row in rows)
            directional_coverage_total+=sum(
                row.get("pairwise_directional_coverage",0.0) for row in rows
            )
            relevant=set(case["relevant"]).intersection(
                aid for aid,_candidate,_attrs,_raw_attrs in specs
            )
            ranks=[index+1 for index,row in enumerate(rows) if row["article"]["id"] in relevant]
            if not ranks:
                missing_positive+=1
                continue
            hits+=min(ranks)<=self.config["top_k"]
            mrr+=sum(1/rank for rank in ranks)/len(ranks)
            for cutoff,target in ((5,"ndcg5"),(10,"ndcg10")):
                dcg=sum(1/math.log2(rank+1) for rank in ranks if rank<=cutoff)
                ideal=sum(1/math.log2(rank+1) for rank in range(1,min(len(relevant),cutoff)+1))
                if target=="ndcg5": ndcg5+=dcg/ideal if ideal else 0
                else: ndcg10+=dcg/ideal if ideal else 0
            positives=[row for row in rows if row["article"]["id"] in relevant]
            negatives=[row for row in rows if row["article"]["id"] not in relevant]
            if positives and negatives:
                case_auc=sum(
                    (self._ranking_signature(p)>self._ranking_signature(n))
                    +0.5*(self._ranking_signature(p)==self._ranking_signature(n))
                    for p in positives for n in negatives
                )/(len(positives)*len(negatives))
                case_primary_auc=sum(
                    (p["score"]>n["score"])+0.5*(p["score"]==n["score"])
                    for p in positives for n in negatives
                )/(len(positives)*len(negatives))
                case_pointwise_auc=sum(
                    (self._pointwise_signature(p)>self._pointwise_signature(n))
                    +0.5*(self._pointwise_signature(p)==self._pointwise_signature(n))
                    for p in positives for n in negatives
                )/(len(positives)*len(negatives))
                case_popularity_auc=sum(
                    (p["baseline"]>n["baseline"])
                    +0.5*(p["baseline"]==n["baseline"])
                    for p in positives for n in negatives
                )/(len(positives)*len(negatives))
                auc_total+=case_auc; auc_primary_total+=case_primary_auc
                auc_values.append(case_auc); auc_primary_values.append(case_primary_auc)
                auc_popularity_values.append(case_popularity_auc)
                case_proof_auc=sum(
                    (self._proof_ranking_signature(p)>self._proof_ranking_signature(n))
                    +0.5*(self._proof_ranking_signature(p)==self._proof_ranking_signature(n))
                    for p in positives for n in negatives
                )/(len(positives)*len(negatives))
                auc_proof_values.append(case_proof_auc)
                auc_pointwise_values.append(case_pointwise_auc)
                case_identity=(case.get("source_impression_id") or case.get("id")
                               or f"evaluation_{case_index}")
                auc_per_impression.append({
                    "index":case_index,"id":str(case_identity),
                    "user":case["user"],
                    "candidates":len(rows),"positives":len(positives),
                    "negatives":len(negatives),"auc":case_auc,
                    "auc_proof_only":case_proof_auc,
                    "slate_digest":case.get("training_confirmation_slate_digest"),
                })
                auc_cases+=1
        if (matched_direct_reasoner_ablation.get("status")=="computed"
                and direct_ranked is not None):
            petta_ablation_auc=[]; direct_ablation_auc=[]
            exact_order=0; exact_order_and_signature=0
            signature_matches=0; signature_candidates=0
            mismatched_impressions=[]
            for case_index,((case,specs,_start,_end),petta_rows,direct_rows) in enumerate(
                    zip(prepared,ranked,direct_ranked)):
                petta_order=[row["article"]["id"] for row in petta_rows]
                direct_order=[row["article"]["id"] for row in direct_rows]
                if (len(set(petta_order))!=len(petta_order)
                        or set(petta_order)!=set(direct_order)):
                    raise RuntimeError(
                        "matched direct ablation changed the candidate slate"
                    )
                order_match=petta_order==direct_order
                exact_order+=order_match
                petta_by_id={row["article"]["id"]:row for row in petta_rows}
                direct_by_id={row["article"]["id"]:row for row in direct_rows}
                slate_signature_match=True
                for article_id in petta_order:
                    matched=(
                        article_id in direct_by_id
                        and self._ranking_signature(petta_by_id[article_id])
                        ==self._ranking_signature(direct_by_id[article_id])
                    )
                    signature_matches+=matched
                    signature_candidates+=1
                    slate_signature_match=slate_signature_match and matched
                exact_order_and_signature+=(
                    order_match and slate_signature_match
                )
                if (not order_match or not slate_signature_match) and len(
                        mismatched_impressions)<32:
                    mismatched_impressions.append(str(
                        case.get("source_impression_id") or case.get("id")
                        or f"evaluation_{case_index}"
                    ))
                relevant=set(case["relevant"]).intersection(
                    aid for aid,_candidate,_attrs,_raw_attrs in specs
                )
                petta_positive=[row for row in petta_rows
                                if row["article"]["id"] in relevant]
                petta_negative=[row for row in petta_rows
                                if row["article"]["id"] not in relevant]
                direct_positive=[row for row in direct_rows
                                 if row["article"]["id"] in relevant]
                direct_negative=[row for row in direct_rows
                                 if row["article"]["id"] not in relevant]
                if not petta_positive or not petta_negative:
                    continue
                denominator=len(petta_positive)*len(petta_negative)
                petta_ablation_auc.append(sum(
                    (self._ranking_signature(positive)
                     >self._ranking_signature(negative))
                    +0.5*(self._ranking_signature(positive)
                          ==self._ranking_signature(negative))
                    for positive in petta_positive
                    for negative in petta_negative
                )/denominator)
                direct_ablation_auc.append(sum(
                    (self._ranking_signature(positive)
                     >self._ranking_signature(negative))
                    +0.5*(self._ranking_signature(positive)
                          ==self._ranking_signature(negative))
                    for positive in direct_positive
                    for negative in direct_negative
                )/denominator)
            if len(petta_ablation_auc)!=len(direct_ablation_auc):
                raise RuntimeError(
                    "matched direct ablation produced an unmatched AUC cohort"
                )
            petta_auc=(sum(petta_ablation_auc)/len(petta_ablation_auc)
                       if petta_ablation_auc else None)
            direct_auc=(sum(direct_ablation_auc)/len(direct_ablation_auc)
                        if direct_ablation_auc else None)
            auc_delta=(direct_auc-petta_auc
                       if direct_auc is not None and petta_auc is not None
                       else None)
            paired_deltas=[direct-petta for direct,petta in zip(
                direct_ablation_auc,petta_ablation_auc
            )]
            complete_match=(
                exact_order_and_signature==len(prepared)
                and signature_matches==signature_candidates
                and matched_direct_reasoner_ablation.get(
                    "exact_point_slate_order_matches"
                )==len(prepared)
                and matched_direct_reasoner_ablation.get(
                    "exact_point_signature_matches"
                )==matched_direct_reasoner_ablation.get(
                    "point_signatures_compared"
                )
            )
            matched_direct_reasoner_ablation.update(
                status=("completed_match" if complete_match
                        else "completed_difference"),
                auc_definition=(
                    "same served-order macro impression AUC used by the main "
                    "benchmark; ties receive half credit"
                ),
                eligible_auc_impressions=len(petta_ablation_auc),
                pettachainer_auc=petta_auc,
                direct_auc=direct_auc,
                auc_delta_direct_minus_pettachainer=auc_delta,
                auc_delta_95_ci=self._bootstrap_interval(
                    paired_deltas,seed^0xD1CE37
                ),
                exact_slate_order_matches=exact_order,
                exact_slate_order_match_rate=(
                    exact_order/len(prepared) if prepared else None
                ),
                exact_candidate_signature_matches=signature_matches,
                candidate_signatures_compared=signature_candidates,
                exact_candidate_signature_match_rate=(
                    signature_matches/signature_candidates
                    if signature_candidates else None
                ),
                exact_slate_order_and_signature_matches=(
                    exact_order_and_signature
                ),
                exact_slate_order_and_signature_match_rate=(
                    exact_order_and_signature/len(prepared)
                    if prepared else None
                ),
                mismatch_impression_ids=mismatched_impressions,
                finding=(
                    "The supported shallow point-and-pair topology has "
                    "equivalent scoring/ranking output for this matched cohort. "
                    "PeTTaChainer supplies executable "
                    "semantics and proof provenance here, but this comparison "
                    "does not show a ranking/AUC lift over its exact direct "
                    "reconstruction."
                    if complete_match else
                    "The direct reconstruction and PeTTaChainer scoring output differ; "
                    "inspect the reported mismatch impressions before making "
                    "a scoring-output-equivalence claim."
                ),
            )
        cases=len(prepared)
        if not cases: raise ValueError("candidate configuration left no evaluable positive impressions")
        pair_delta_values=[
            pair_auc-point_auc
            for pair_auc,point_auc in zip(auc_values,auc_pointwise_values)
        ]
        pair_delta_ci=(
            self._bootstrap_interval(pair_delta_values,seed^0xC0FFEE)
            if self.config["ranking_mode"]=="pairwise" else [0.0,0.0]
        )
        if self.config["ranking_mode"]!="pairwise":
            auc_signal="lexicographic pointwise proof score, STV, and exact-tie prior"
        elif self.config["pair_aggregation"]=="proof_margin":
            family_fusion=self.config.get("pair_family_fusion","flat_margin")
            auc_signal=(f"PeTTaChainer pairwise proof_margin {family_fusion} "
                        f"{self.config['pair_margin_transform']} signed-power "
                        f"{self.config['pair_margin_power']:g} ProofRank tournament "
                        "over pointwise proofs")
        else:
            auc_signal=(f"PeTTaChainer pairwise {self.config['pair_aggregation']} "
                        "ProofRank tournament over pointwise proofs")
        metric_seconds=time.perf_counter()-metric_started
        evaluation_seconds=time.perf_counter()-evaluation_started
        total_seconds=time.perf_counter()-started
        ranking_pipeline_seconds=sum((
            candidate_preparation_seconds,
            point_reasoning_seconds,
            pair_planning_seconds,
            pair_reasoning_seconds,
            rank_aggregation_seconds,
        ))
        candidate_subtotal=math.fsum(candidate_preparation_profile.values())
        inner_pair_plan_seconds=float(pair_plan_profile.get("total_seconds",0.0))
        materialization_seconds=float(
            pair_materialization_profile.get("total_seconds",0.0)
        )
        proof_preparation_seconds=math.fsum(
            float(pair_reasoning_profile.get(key,0.0))
            for key in PAIR_PROOF_PREPARATION_TIMERS
        )
        pair_preparation_profile={
            "candidate_features":{
                **{key:round(value,6)
                   for key,value in candidate_preparation_profile.items()},
                "residual_admission_and_collection_seconds":round(max(
                    0.0,candidate_preparation_seconds-candidate_subtotal
                ),6),
                "total_seconds":round(candidate_preparation_seconds,6),
                "definition":(
                    "Candidate/history context is calculated once per candidate, "
                    "then projected and serialized once; the resulting raw_attrs "
                    "objects are reused by every pair containing that candidate."
                ),
            },
            "pair_construction":{
                **{key:round(value,6) for key,value in pair_plan_profile.items()},
                "point_row_preparation_seconds":round(
                    point_row_preparation_seconds,6
                ),
                "pair_budget_accounting_seconds":round(
                    pair_budget_accounting_seconds,6
                ),
                "outer_residual_seconds":round(max(
                    0.0,pair_planning_seconds-inner_pair_plan_seconds
                    -point_row_preparation_seconds-pair_budget_accounting_seconds
                ),6),
                "outer_total_seconds":round(pair_planning_seconds,6),
                "definition":(
                    "Each unordered pair derives forward categorical relations "
                    "once; the reverse is an exact antisymmetric transform. An "
                    "inverted premise join identifies active rules without "
                    "rescanning the rule table. Pair case serialization hashes "
                    "only active-rule vectors."
                ),
            },
            "candidate_pair_materialization":{
                **{key:(round(value,6) if isinstance(value,float) else value)
                   for key,value in pair_materialization_profile.items()},
                "definition":(
                    "Indexes unique pair cases and, only for non-factorized "
                    "aggregation, serializes and inserts candidate-pair facts."
                ),
            },
            "proof_channel_preparation":(
                _proof_channel_preparation_profile(pair_reasoning_profile)
            ),
            "profiled_pre_query_total_seconds":round(
                candidate_preparation_seconds+pair_planning_seconds
                +materialization_seconds+proof_preparation_seconds,6
            ),
            "timing_sum_definition":(
                "candidate_features.total + pair_construction.outer_total + "
                "candidate_pair_materialization.total + proof-channel topology, "
                "activation join, template serialization and AtomSpace insertion"
            ),
        }
        cache_hits=point_cache_stats.get("hits",0)+pair_cache_stats.get("hits",0)
        cache_misses=(point_cache_stats.get("misses",0)
                      +pair_cache_stats.get("misses",0))
        cache_observed_mode=(
            "cold" if cache_hits==0 else
            "warm" if cache_misses==0 else
            "mixed"
        )
        result={"id":uuid.uuid4().hex[:8],"dataset":self.dataset_info(),"cases":cases,
                "candidates":len(all_specs),"hit_rate":round(hits/cases,4),"mrr":round(mrr/cases,4),
                "ndcg_at_5":round(ndcg5/cases,4),"ndcg_at_10":round(ndcg10/cases,4),
                "ndcg":round(ndcg10/cases,4),
                "metric_definitions":{
                    "auc":(
                        "Macro mean, over eligible impressions, of the probability "
                        "that a clicked candidate ranks above a non-clicked candidate; "
                        "ties receive half credit."
                    ),
                    "mrr":(
                        "Official MIND convention: within each impression average "
                        "1/rank over every clicked candidate, then macro-average "
                        "those impression values."
                    ),
                    "ndcg":(
                        "Normalized discounted cumulative gain from all clicked "
                        "candidates, macro-averaged over impressions."
                    ),
                },
                "auc":round(auc_total/auc_cases,4) if auc_cases else None,
                "auc_proof_only":round(sum(auc_proof_values)/auc_cases,4)
                                 if auc_cases else None,
                "auc_primary_score":round(auc_primary_total/auc_cases,4) if auc_cases else None,
                "auc_95_ci":self._bootstrap_interval(auc_values,seed),
                "auc_proof_only_95_ci":self._bootstrap_interval(
                    auc_proof_values,seed^0x50524F4F
                ),
                "auc_primary_95_ci":self._bootstrap_interval(
                    auc_primary_values,seed^0xA5A5A5A5
                ),
                "auc_pointwise_ranking":round(sum(auc_pointwise_values)/auc_cases,4)
                                        if auc_cases else None,
                "same_cohort_baselines":{
                    "candidate_population":(
                        "identical held-out impressions and candidate slates"
                    ),
                    "random_expected_auc":0.5,
                    "training_click_count_popularity_auc":round(
                        sum(auc_popularity_values)/auc_cases,4
                    ) if auc_cases else None,
                    "training_click_count_popularity_auc_95_ci":(
                        self._bootstrap_interval(
                            auc_popularity_values,seed^0xB45311E
                        ) if auc_cases else [None,None]
                    ),
                    "pointwise_symbolic_auc":round(
                        sum(auc_pointwise_values)/auc_cases,4
                    ) if auc_cases else None,
                    "pointwise_symbolic_auc_95_ci":self._bootstrap_interval(
                        auc_pointwise_values,seed^0x50117
                    ) if auc_cases else [None,None],
                    "statement":(
                        "Both empirical baselines use the same frozen training "
                        "snapshot and the exact candidate population evaluated "
                        "by the pairwise ranker. Popularity is diagnostic only "
                        "and is not a serving fallback."
                    ),
                },
                "auc_pairwise_delta":round(
                    sum(pair_delta_values)/auc_cases,4
                ) if auc_cases and self.config["ranking_mode"]=="pairwise" else 0.0,
                "auc_pairwise_delta_95_ci":pair_delta_ci,
                "pairwise_pointwise_diagnostic":{
                    "kind":"diagnostic_not_architecture_gate",
                    "status":(
                        "pass" if pair_delta_ci and pair_delta_ci[0]>0.0
                        else "hold"
                    ) if self.config["ranking_mode"]=="pairwise" else "not_applicable",
                    "criterion":"paired held-out AUC delta 95% lower bound > 0",
                    "comparator":"same-run pointwise symbolic ranker",
                    "delta_95_ci":pair_delta_ci,
                    "native_complete_impressions":not bool(cap),
                    "temporal_rule_stability_required":True,
                },
                "auc_signal":auc_signal,
                "proof_coverage":round(proof_candidates/len(all_specs),4) if all_specs else 0,
                "pairwise_proof_coverage":round(pair_coverage_total/len(all_specs),4) if all_specs else 0,
                "pairwise_directional_coverage":round(
                    directional_coverage_total/len(all_specs),4
                ) if all_specs else 0,
                "seconds":round(total_seconds,3),"reasoner_batches":reasoner_calls,
                "timings":{
                    "model_preparation_seconds":round(model_preparation_seconds,6),
                    "candidate_preparation_seconds":round(candidate_preparation_seconds,6),
                    "point_reasoning_seconds":round(point_reasoning_seconds,6),
                    "pair_planning_seconds":round(pair_planning_seconds,6),
                    "pair_reasoning_seconds":round(pair_reasoning_seconds,6),
                    "pair_proof_reliability_seconds":round(
                        pair_reliability_seconds,6
                    ),
                    "parity_audit_seconds":round(parity_audit_seconds,6),
                    "matched_direct_ablation_seconds":round(
                        direct_ablation_seconds,6
                    ),
                    "rank_aggregation_seconds":round(rank_aggregation_seconds,6),
                    "ranking_pipeline_seconds":round(
                        ranking_pipeline_seconds,6
                    ),
                    "metric_seconds":round(metric_seconds,6),
                    "evaluation_seconds":round(evaluation_seconds,6),
                    "total_seconds":round(total_seconds,6),
                },
                "timing_definitions":{
                    "ranking_pipeline_seconds":(
                        "candidate preparation + point reasoning + pair planning "
                        "+ pair reasoning + rank aggregation; excludes mining, "
                        "pair-proof reliability, parity audit, matched direct "
                        "ablation, metric calculation, and candidate retrieval"
                    ),
                    "pair_proof_reliability_seconds":(
                        "held-out individual PairSignal Brier/log-loss/ECE "
                        "diagnostic; excluded from ranking latency"
                    ),
                    "matched_direct_ablation_seconds":(
                        "opt-in direct channel reconstruction and identical "
                        "tournament/fusion replay; excludes its metric comparison"
                    ),
                },
                "pair_preparation_profile":pair_preparation_profile,
                "reasoner_deadlines":{
                    "serving_seconds":self._serving_reasoner_timeout(),
                    "benchmark_seconds":benchmark_timeout_sec,
                    "scope":"finite parent-process deadline per scorer RPC",
                    "expiry_policy":(
                        "terminate expired worker, reconstruct the last "
                        "acknowledged rule/fact snapshot, and fail the request"
                    ),
                },
                "proof_cache":{
                    "requested_mode":(
                        "force_cold" if force_cold else "reuse_existing"
                    ),
                    "observed_mode":cache_observed_mode,
                    "scope":(
                        "proof-result caches only; compiled rules and grounded "
                        "facts remain identical"
                    ),
                    "accounting_unit":(
                        "unique logical point/pair case roots plus independently "
                        "cached factorized point/pair-channel roots"
                    ),
                    "entries_before_clear":proof_cache_entries_before_clear,
                    "entries_at_evaluation_start":(
                        proof_cache_entries_at_evaluation_start
                    ),
                    "point":point_cache_stats,
                    "pair":pair_cache_stats,
                    "hits":cache_hits,"misses":cache_misses,
                },
                "candidate_retrieval":{
                    "included":False,
                    "concurrency_measured":False,
                    "latency_sla_claimed":False,
                    "production_retrieval_recall_measured":False,
                    "source":"logged evaluation impression slate",
                    "logged_slate_admission":{
                        "candidate_cap":cap,
                        "supplied_candidates":supplied_candidate_count,
                        "admitted_candidates":admitted_candidate_count,
                        "supplied_relevant_items":supplied_relevant_count,
                        "admitted_relevant_items":admitted_relevant_count,
                        "relevant_item_inclusion_recall":(
                            admitted_relevant_count/supplied_relevant_count
                            if supplied_relevant_count else None
                        ),
                        "complete_relevant_slates":complete_relevant_slates,
                        "slates_with_supplied_relevant_items":relevant_slates,
                        "complete_relevant_slate_rate":(
                            complete_relevant_slates/relevant_slates
                            if relevant_slates else None
                        ),
                        "definition":(
                            "Admission after the optional seeded candidate cap, "
                            "relative only to relevant items already present in "
                            "the logged impression slate."
                        ),
                    },
                    "statement":(
                        "This replay evaluates ranking after candidates are supplied; "
                        "logged-slate cap admission is reported separately, but "
                        "production retrieval latency and retrieval recall are not measured."
                    ),
                },
                "ranking_workload":{
                    "candidates":len(all_specs),
                    "maximum_candidates_in_one_slate":maximum_slate_candidates,
                    "unordered_pair_comparisons":unordered_pair_comparisons,
                    "maximum_unordered_pair_comparisons_in_one_slate":(
                        maximum_slate_comparisons
                    ),
                    "oriented_pair_cases":oriented_pair_cases,
                    "opponent_limit":int(self.config["pairwise_opponents"]),
                    "configured_max_unordered_pair_comparisons_per_slate":int(
                        self.config["max_pair_comparisons"]
                    ),
                    "configured_max_total_unordered_pair_comparisons":int(
                        self.config["max_total_pair_comparisons"]
                    ),
                    "proof_cache_entry_limit":int(
                        self.config["max_proof_cache_entries"]
                    ),
                    "reasoner_mutation_journal":self.engine.journal_audit,
                    "comparison_policy":(
                        "complete_all_pairs"
                        if int(self.config["pairwise_opponents"])<=0
                        else "bounded_cyclic_graph"
                    ),
                    "complete_all_pairs_complexity":"sum_i n_i*(n_i-1)/2",
                },
                "reasoner_semantic_parity":reasoner_semantic_parity,
                "matched_direct_reasoner_ablation":(
                    matched_direct_reasoner_ablation
                ),
                "heldout_pair_proof_reliability":(
                    heldout_pair_proof_reliability
                ),
                "pointwise_reasoner_batches":self._point_query_calls,
                "pointwise_reasoner_roots":self._point_query_roots,
                "pointwise_pruned_empty_roots":(
                    self._point_pruned_query_roots
                ),
                "pointwise_unique_case_channel_activations":(
                    self._point_channel_activations
                ),
                "pointwise_reused_channel_activations":(
                    self._point_reused_channel_activations
                ),
                "pointwise_proof_factorization":(
                    "alpha_normalized_isolated_channels"
                    if self.config["aggregation"]=="weighted"
                    else "disabled_shared_engagement_revision"
                ),
                "pointwise_proof_completeness":dict(
                    self._last_point_completeness
                ),
                "pointwise_proof_templates":len(
                    self._point_channel_templates
                ),
                "pointwise_proof_template_audit":sorted(
                    self._point_channel_templates.values(),
                    key=lambda item:(item["rule_id"],item["proof_channel_id"]),
                ),
                "pairwise_reasoner_batches":self._pair_query_calls,
                "pairwise_reasoner_roots":self._pair_query_roots,
                "pairwise_pruned_empty_roots":self._pair_pruned_query_roots,
                "pairwise_unique_case_channel_activations":self._pair_channel_activations,
                "pairwise_reused_channel_activations":self._pair_reused_channel_activations,
                "pairwise_proof_factorization":(
                    "alpha_normalized_isolated_channels"
                    if self.config["pair_aggregation"]=="proof_margin"
                    else "disabled_shared_posterior"
                ),
                "pairwise_proof_templates":len(self._pair_channel_templates),
                "pairwise_proof_template_audit":sorted(
                    self._pair_channel_templates.values(),
                    key=lambda item:(item["dependency_id"],item["proof_channel_id"]),
                ),
                "evaluation_relational_activation_audit":(
                    evaluation_relational_activation_audit
                ),
                "rules":len(self.mined_rules),"pair_rules":len(self.pair_rules),
                "mining":mining,"config":self.config.copy(),
                "requested_cases":len(tests),"sampled_without_positive":missing_positive,
                "auc_cases":auc_cases,"auc_per_impression":auc_per_impression,
                "score_unique":len(score_counts),
                "score_tie_groups":sum(count>1 for count in score_counts.values()),
                "score_tied_candidates":sum(count for count in score_counts.values() if count>1),
                "ranking_unique":len(ranking_counts),
                "proof_ranking_unique":len(proof_ranking_counts),
                "ranking_tie_groups":sum(count>1 for count in ranking_counts.values()),
                "evaluation_protocol":"native_impressions" if not cap else "seeded_uniform_candidate_sample",
                "eval_case_limit":eval_case_limit}
        # Compatibility alias for older clients.  This is intentionally marked
        # as a diagnostic: only ``compare_profiles`` can promote an architecture.
        result["promotion_gate"]={
            **result["pairwise_pointwise_diagnostic"],
            "deprecated_alias":True,
        }
        self.runs.insert(0,result); return result

    def compare_profiles(self,config):
        """Run a proof-backed champion/challenger architecture promotion gate.

        The currently active architecture is the frozen champion.  A narrowly
        scoped ``challenger_config`` may change the mining strategy/depth, pair
        feature profile, and/or proof-margin power; evaluation controls remain
        frozen.  Both sides are independently remined and evaluated through a
        freshly compiled PeTTaChainer worker on the identical cohort. A challenger is
        retained only when its paired impression-level AUC interval clears
        ``min_delta`` on the complete, uncapped cohort.
        """
        if self._online_events:
            raise ValueError(
                "reload the dataset before architecture comparison: live feedback "
                "is isolated from the immutable offline evaluation protocol"
            )
        if not isinstance(config,dict):
            raise ValueError("architecture comparison config must be an object")
        champion_config=self.config.copy()
        champion_profile=str(champion_config["pair_feature_profile"])
        requested_champion=config.get("champion_pair_feature_profile",champion_profile)
        if str(requested_champion)!=champion_profile:
            raise ValueError(
                "champion_pair_feature_profile must equal the currently active "
                f"frozen champion ({champion_profile})"
            )
        raw_challenger_config=config.get("challenger_config",{})
        if raw_challenger_config is None: raw_challenger_config={}
        if not isinstance(raw_challenger_config,dict):
            raise ValueError("challenger_config must be an object")
        unknown_challenger_keys=sorted(
            set(raw_challenger_config)-CHALLENGER_CONFIG_KEYS
        )
        if unknown_challenger_keys:
            raise ValueError(
                "challenger_config contains unsupported keys: "
                +", ".join(unknown_challenger_keys)
            )
        challenger_overrides=dict(raw_challenger_config)
        legacy_challenger_value=config.get(
            "challenger_pair_feature_profile",config.get("pair_feature_profile")
        )
        nested_challenger_value=challenger_overrides.get("pair_feature_profile")
        if (legacy_challenger_value is not None
                and nested_challenger_value is not None
                and str(legacy_challenger_value)!=str(nested_challenger_value)):
            raise ValueError(
                "challenger pair feature profile conflicts with challenger_config"
            )
        challenger_value=(nested_challenger_value
                          if nested_challenger_value is not None
                          else legacy_challenger_value)
        if challenger_value is None and challenger_overrides:
            challenger_value=champion_profile
        if challenger_value is None:
            raise ValueError("challenger_pair_feature_profile is required")
        challenger_profile=str(challenger_value)
        if challenger_profile not in PAIR_FEATURE_PROFILES:
            raise ValueError(
                "challenger_pair_feature_profile must be one of: "
                + ", ".join(sorted(PAIR_FEATURE_PROFILES))
            )
        challenger_overrides["pair_feature_profile"]=challenger_profile
        if "pair_margin_power" in challenger_overrides:
            try:
                challenger_power=float(challenger_overrides["pair_margin_power"])
            except (TypeError,ValueError) as exc:
                raise ValueError(
                    "challenger pair_margin_power must be a finite number between "
                    f"{PAIR_MARGIN_POWER_MIN} and {PAIR_MARGIN_POWER_MAX}"
                ) from exc
            if (not math.isfinite(challenger_power)
                    or not PAIR_MARGIN_POWER_MIN<=challenger_power<=PAIR_MARGIN_POWER_MAX):
                raise ValueError(
                    "challenger pair_margin_power must be a finite number between "
                    f"{PAIR_MARGIN_POWER_MIN} and {PAIR_MARGIN_POWER_MAX}"
                )
            challenger_overrides["pair_margin_power"]=challenger_power
        challenger_effective_config={**champion_config,**challenger_overrides}
        if champion_config.get("ranking_mode")!="pairwise":
            raise ValueError("architecture comparison requires ranking_mode=pairwise")
        try:
            min_delta=float(config.get("min_delta",0.0))
        except (TypeError,ValueError) as exc:
            raise ValueError("min_delta must be a finite number between 0 and 1") from exc
        if not math.isfinite(min_delta) or not 0.0<=min_delta<=1.0:
            raise ValueError("min_delta must be a finite number between 0 and 1")
        try:
            repetitions=int(config.get("bootstrap_repetitions",1000))
            cap=max(0,int(config.get(
                "max_candidates",champion_config.get("max_candidates",0)
            )))
            eval_case_limit=max(0,int(config.get("eval_case_limit",0)))
            bootstrap_seed=int(config.get(
                "bootstrap_seed",champion_config.get("random_seed",7)
            ))
        except (TypeError,ValueError) as exc:
            raise ValueError(
                "bootstrap_repetitions, max_candidates, eval_case_limit, and "
                "bootstrap_seed must be integers"
            ) from exc
        if repetitions<1 or repetitions>100000:
            raise ValueError("bootstrap_repetitions must be between 1 and 100000")

        full_case_count=len(self.evaluation_cases())
        common={"max_candidates":cap,"eval_case_limit":eval_case_limit,
                "remine":True}
        champion_run=challenger_run=None
        try:
            champion_run=self.benchmark({
                **champion_config,**common,
            })
            challenger_run=self.benchmark({
                **challenger_effective_config,**common,
            })
            champion_records=champion_run["auc_per_impression"]
            challenger_records=challenger_run["auc_per_impression"]
            identity_fields=("index","id","candidates","positives","negatives")
            champion_identities=[tuple(row[key] for key in identity_fields)
                                 for row in champion_records]
            challenger_identities=[tuple(row[key] for key in identity_fields)
                                   for row in challenger_records]
            if champion_identities!=challenger_identities:
                raise RuntimeError(
                    "champion and challenger did not score an identical "
                    "held-out impression cohort"
                )
            if not champion_records:
                raise ValueError("evaluation cohort has no impressions with pairwise AUC")
            paired=[]; deltas=[]; proof_deltas=[]
            for champion_row,challenger_row in zip(
                champion_records,challenger_records
            ):
                delta=challenger_row["auc"]-champion_row["auc"]
                proof_delta=(challenger_row["auc_proof_only"]
                             -champion_row["auc_proof_only"])
                deltas.append(delta); proof_deltas.append(proof_delta)
                paired.append({
                    "index":champion_row["index"],"id":champion_row["id"],
                    "champion_auc":round(champion_row["auc"],8),
                    "challenger_auc":round(challenger_row["auc"],8),
                    "delta":round(delta,8),
                })
            raw_interval=self._bootstrap_interval_raw(
                deltas,bootstrap_seed^0x4348414D,repetitions
            )
            interval=[round(raw_interval[0],6),round(raw_interval[1],6)]
            raw_proof_interval=self._bootstrap_interval_raw(
                proof_deltas,bootstrap_seed^0x50524F4D,repetitions
            )
            proof_interval=[round(raw_proof_interval[0],6),
                            round(raw_proof_interval[1],6)]
            full_cohort=(
                cap==0 and eval_case_limit==0
                and champion_run["cases"]==challenger_run["cases"]==full_case_count
                and champion_run["auc_cases"]==champion_run["cases"]
                and challenger_run["auc_cases"]==challenger_run["cases"]
                and champion_run["sampled_without_positive"]==0
                and challenger_run["sampled_without_positive"]==0
            )
            passed=bool(full_cohort and raw_interval[0]>min_delta)
            restoration=None
            if not passed:
                self.configure(champion_config)
                restoration=self.mine()

            def architecture_summary(run,profile):
                run_config=run["config"]
                engine={
                    "fixed_combinations":"fpMiner -> PeTTaChainer",
                    "target_aware":
                        "fpMiner unary + target-aware expansion -> PeTTaChainer",
                    "conditional_llm":
                        "fpMiner LLM unary + stable conditional expansion -> PeTTaChainer",
                    "conditional_llm_seed_only":
                        "fpMiner LLM discovery seeds + stable conditional children -> PeTTaChainer",
                }[run_config["miner_strategy"]]
                return {
                    "profile":profile,"run_id":run["id"],"auc":run["auc"],
                    "pair_margin_power":run_config["pair_margin_power"],
                    "config":run_config.copy(),
                    "architecture_config":{
                        key:run_config[key] for key in sorted(CHALLENGER_CONFIG_KEYS)
                    },
                    "auc_proof_only":run["auc_proof_only"],"cases":run["cases"],
                    "auc_cases":run["auc_cases"],"candidates":run["candidates"],
                    "proof_coverage":run["proof_coverage"],
                    "pairwise_proof_coverage":run["pairwise_proof_coverage"],
                    "rules":run["rules"],"pair_rules":run["pair_rules"],
                    "reasoner_batches":run["reasoner_batches"],
                    "pairwise_reasoner_batches":run["pairwise_reasoner_batches"],
                    "mining":run["mining"],
                    "engine":engine,
                }

            reasons=[]
            if cap: reasons.append("max_candidates must be 0")
            if eval_case_limit: reasons.append("eval_case_limit must be 0")
            if champion_run["auc_cases"]!=full_case_count or challenger_run["auc_cases"]!=full_case_count:
                reasons.append("every held-out impression must have a complete AUC case")
            cohort_payload=json.dumps(
                champion_identities,ensure_ascii=False,separators=(",",":")
            ).encode("utf-8")
            return {
                "kind":"champion_challenger_architecture_gate",
                "status":"pass" if passed else "hold",
                "decision":("promote_challenger" if passed else "retain_champion"),
                "criterion":"paired held-out AUC delta 95% lower bound > min_delta",
                "metric":"serving_ranking_auc",
                "min_delta":min_delta,
                "full_cohort_eligible":full_cohort,
                "ineligibility_reasons":reasons,
                "cohort_fingerprint":hashlib.sha256(cohort_payload).hexdigest(),
                "paired_impressions":len(deltas),
                "mean_auc_delta":round(sum(deltas)/len(deltas),6),
                "delta_95_ci":interval,
                "mean_proof_only_auc_delta":round(
                    sum(proof_deltas)/len(proof_deltas),6
                ),
                "proof_only_delta_95_ci":proof_interval,
                "bootstrap":{"seed":bootstrap_seed,"repetitions":repetitions,
                             "unit":"held-out impression"},
                "champion":architecture_summary(champion_run,champion_profile),
                "challenger":architecture_summary(challenger_run,challenger_profile),
                "per_impression":paired,
                "active_pair_feature_profile":self.config["pair_feature_profile"],
                "active_pair_margin_power":self.config["pair_margin_power"],
                "active_miner_strategy":self.config["miner_strategy"],
                "active_conjunctions":self.config["conjunctions"],
                "active_pair_conjunctions":self.config["pair_conjunctions"],
                "champion_restored":not passed,
                "restoration_mining":restoration,
                "distinct_from":"pairwise_pointwise_diagnostic",
            }
        except BaseException as exc:
            try:
                self.configure(champion_config)
                self.mine()
            except BaseException as restore_exc:
                raise RuntimeError(
                    "architecture comparison failed and champion restoration also failed: "
                    f"{restore_exc}"
                ) from exc
            raise

    def training_confirmation(self,config):
        """Confirm one challenger on a chronological tail of training only."""
        from ..evaluation.training_gate import run_training_confirmation
        return run_training_confirmation(
            self,config,context_features=CONTEXT_FEATURES,
            positive_actions=POSITIVE,
            challenger_config_keys=CHALLENGER_CONFIG_KEYS,
        )

    def tune(self,config):
        """Select mining/scoring settings on one slice and report once on another."""
        objective=str(config.get("objective", "auc"))
        objective_fields={"auc":"auc","mrr":"mrr","ndcg_at_10":"ndcg_at_10"}
        if objective not in objective_fields:
            raise ValueError("objective must be auc, mrr, or ndcg_at_10")
        objective_field=objective_fields[objective]
        source_key=next((key for key in ("eval_impressions","evaluation","tests","impressions")
                         if isinstance(self.data.get(key),list)),None)
        if source_key is None or len(self.data[source_key])<20:
            raise ValueError("auto-tuning needs at least 20 labeled evaluation impressions")
        original_source=self.data[source_key]
        split=max(10,min(len(original_source)-10,int(len(original_source)*0.7)))
        tuning_source=original_source[:split]; heldout_source=original_source[split:]
        base_support=max(1,int(config.get("min_support",self.config["min_support"])))
        base_ratio=max(0,int(config.get("negative_ratio",self.config["negative_ratio"])))
        base_rules=max(1,int(config.get("max_rules",self.config["max_rules"])))
        supports=config.get("support_grid",[base_support,base_support*2])
        depths=config.get("depth_grid",[2,3])
        ratios=config.get("negative_ratio_grid",[0,base_ratio] if base_ratio else [0,4])
        aggregations=config.get("aggregation_grid",["max","weighted"])
        supports=list(dict.fromkeys(max(1,int(value)) for value in supports))[:4]
        depths=list(dict.fromkeys(int(value) for value in depths if 2<=int(value)<=4))[:3]
        ratios=list(dict.fromkeys(max(0,int(value)) for value in ratios))[:3]
        aggregations=list(dict.fromkeys(str(value) for value in aggregations
                                       if str(value) in {"max","weighted"}))[:2]
        if not depths or not aggregations: raise ValueError("auto-tuning grid is empty")
        common={"top_k":int(config.get("top_k",self.config["top_k"])),
                "chain_steps":int(config.get("chain_steps",self.config["chain_steps"])),
                "max_candidates":int(config.get("max_candidates",0)),
                "random_seed":int(config.get("random_seed",self.config["random_seed"])),
                "feature_profile":str(config.get("feature_profile",self.config["feature_profile"])),
                "rule_rank":"quality"}
        tuning_results=[]; heldout_result=None; evaluation_cache={}
        original_config=self.config.copy(); original_runs=list(self.runs)
        original_pending=self.pending_events; succeeded=False

        def discard_run(result):
            if self.runs and self.runs[0].get("id")==result.get("id"): self.runs.pop(0)

        try:
            self.data[source_key]=tuning_source
            for depth in depths:
                for support in supports:
                    for ratio in ratios:
                        mining_config={**common,"conjunctions":depth,"min_support":support,
                                       "negative_ratio":ratio,
                                       "max_rules":max(base_rules,60 if depth>2 else base_rules)}
                        self.configure(mining_config); mining=self.mine()
                        fingerprint=hashlib.sha256(json.dumps([
                            [rule["premises"],rule["target"],rule["strength"],rule["confidence"],
                             rule["negative_strength"],rule["negative_confidence"]]
                            for rule in self.mined_rules
                        ],sort_keys=True,separators=(",",":")).encode()).hexdigest()
                        for aggregation in aggregations:
                            trial_config={**mining_config,"aggregation":aggregation,"remine":False}
                            cache_key=(fingerprint,aggregation,common["top_k"],common["max_candidates"],
                                       common["random_seed"],common["chain_steps"])
                            reused=cache_key in evaluation_cache
                            if reused:
                                self.configure({"aggregation":aggregation})
                                result=evaluation_cache[cache_key]
                            else:
                                result=self.benchmark(trial_config); discard_run(result)
                                evaluation_cache[cache_key]=result
                            tuning_results.append({"config":{key:trial_config[key] for key in
                                ("min_support","conjunctions","max_rules","negative_ratio","aggregation")},
                                "auc":result["auc"],"mrr":result["mrr"],
                                "ndcg_at_5":result["ndcg_at_5"],"ndcg_at_10":result["ndcg_at_10"],
                                "proof_coverage":result["proof_coverage"],"rules":result["rules"],
                                "premise_atoms":sum(len(rule["premises"]) for rule in self.mined_rules),
                                "seconds":result["seconds"],"mining_seconds":mining["seconds"],
                                "reused_evaluation":reused})
            peak_objective=max(row[objective_field] for row in tuning_results)
            objective_tolerance=max(0.0,float(config.get(
                "objective_tolerance",config.get("ndcg_tolerance",0.0025))))
            finalists=[row for row in tuning_results
                       if row[objective_field]>=peak_objective-objective_tolerance]
            # A small validation fluctuation should not promote a much larger
            # symbolic model.  Within the configured nDCG tolerance, apply a
            # deterministic parsimony tie-break before secondary metrics.
            best=min(finalists,key=lambda row:(row["premise_atoms"],row["rules"],
                -(row["auc"] if row["auc"] is not None else -1),-row["mrr"],
                row["mining_seconds"],row["seconds"]))
            best_config={**common,**best["config"]}
            self.data[source_key]=heldout_source
            self.configure(best_config); mining=self.mine()
            heldout_result=self.benchmark({**best_config,"remine":False})
            heldout_result["kind"]="auto_tune_holdout"; heldout_result["mining"]=mining
            succeeded=True
        finally:
            self.data[source_key]=original_source
            if not succeeded:
                self.runs[:]=original_runs
                try:
                    self.configure(original_config); self.mine()
                except Exception:
                    pass
                self.pending_events=original_pending
        return {"best_config":best_config,"tuning_results":tuning_results,
                "heldout_result":heldout_result,"split":{"tuning":len(tuning_source),"heldout":len(heldout_source)},
                "objective":f"{objective} with parsimony tolerance, then secondary metrics and mining time",
                "objective_metric":objective_field,"objective_tolerance":objective_tolerance,
                # Retain the old key for clients that already render it.
                "ndcg_tolerance":objective_tolerance,"state":self.state()}


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
    print(f"Recommendation lab: http://{args.host}:{args.port}")
    server=ProductionHTTPServer((args.host,args.port),Handler)
    try: server.serve_forever()
    finally:
        server.server_close()
        if LAB is not None: LAB.close()
if __name__=="__main__": main()
