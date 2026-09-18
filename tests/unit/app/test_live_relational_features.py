"""Live relation projection follows user revisions and remains proof gated."""

import copy
import hashlib
import threading
from collections import Counter, OrderedDict
from unittest.mock import patch

import pytest

from recommendation.app.async_mining import MiningSnapshot
from recommendation.app.server import BACKGROUND_MODEL_FIELDS, Lab
from recommendation.features.relational_workspace import (
    CANONICAL_CONCEPT_BRIDGE_RULE_ID,
    CONCEPT_CONTINUITY_RULE_ID,
    ENGAGED_CONCEPT_RULE_ID,
    REL_CONCEPT_CONTINUITY_PROOF_IDS,
    REL_CONCEPT_CONTINUITY_SCOPE,
    REL_ENTITY_CONTINUITY_PROOF_IDS,
    REL_ENTITY_CONTINUITY_SCOPE,
    build_concept_relational_proof_plan,
)


def annotation(article_id, lexical):
    concept = f"concept:{lexical}~" + hashlib.sha256(lexical.encode()).hexdigest()
    return {
        "concepts": [concept],
        "provenance": {
            "article_id": article_id,
            "article_content_sha256": "a" * 64,
            "anchored_statements": [
                f'(: anchor_{article_id.lower()} '
                f'(HasConcept "{article_id}" "{lexical}") (STV 1 1))'
            ],
            "canonicalization": {
                "registry_sha256": "b" * 64,
                "mappings": {"concepts": [{
                    "canonical_id": concept, "lexical": lexical,
                }]},
            },
        },
    }


def concept_proof(plan, origin):
    history = origin.history_bridge_paths[0]
    candidate = origin.candidate_bridge_paths[0]
    return (
        f"(: (by {CONCEPT_CONTINUITY_RULE_ID} "
        f"{plan.case_candidate_fact_id} "
        f"(by {ENGAGED_CONCEPT_RULE_ID} {origin.observed_click_fact_id} "
        f"(by {CANONICAL_CONCEPT_BRIDGE_RULE_ID} "
        f"{history.annotation_anchor_fact_id} {history.canonical_mapping_fact_id})) "
        f"(by {CANONICAL_CONCEPT_BRIDGE_RULE_ID} "
        f"{candidate.annotation_anchor_fact_id} {candidate.canonical_mapping_fact_id})) "
        f"(RelConceptContinuity {plan.case_id} {origin.origin_id} "
        f"{candidate.concept_atom}) "
        "(STV 1.0 0.999))"
    )


class RecordingReasoner:
    def __init__(self):
        self.expected = {}
        self.added = []
        self.query_calls = []

    def register(self, plan):
        origins = {origin.origin_id: origin for origin in plan.origins}
        for root in plan.proof_roots:
            origin = origins[root.origin_id]
            self.expected[root.query] = concept_proof(plan, origin)

    def add_atoms_no_check(self, statements, *, timeout_sec):
        self.added.extend(statements)

    def query_many(self, queries, *, steps, timeout_sec):
        self.query_calls.append((tuple(queries), steps, timeout_sec))
        return [[self.expected[query]] for query in queries]


class IdleMining:
    def schedule(self):
        return False

    def status(self):
        return {"state": "idle", "running": False}


class ClosableEngine:
    def __init__(self, pid):
        self.pid = pid
        self.closed = False

    def close(self):
        self.closed = True


def live_lab():
    lab = object.__new__(Lab)
    lab.config = {
        "relational_evidence_mode": "chained",
        "feature_profile": "accuracy_detail_relational",
        "query_batch_size": 32,
        "max_proof_cache_entries": 100,
        "serving_reasoner_timeout_seconds": 10.0,
    }
    lab._articles = {
        "C": {"id": "C", "topic": "science", "subcategory": "space"},
        "H": {"id": "H", "topic": "science", "subcategory": "space"},
        "H2": {"id": "H2", "topic": "science", "subcategory": "space"},
        "D": {"id": "D", "topic": "sports", "subcategory": "football"},
    }
    lab._llm_article_annotations = {
        article_id: annotation(article_id, "space")
        for article_id in lab._articles
    }
    lab._llm_article_annotations["D"] = annotation("D", "football")
    lab.data = {"users": {"u": {"history": ["H"]}}}
    lab._relational_feature_cache = {}
    lab._loaded_relational_statements = set()
    lab._live_relational_proof_ledger = {}
    lab._candidate_relational_proof_refs = {}
    lab._relational_query_calls = 0
    lab._relational_query_roots = 0
    lab.engine = RecordingReasoner()
    return lab


def event_ready(lab):
    lab.config.update({
        "mine_interval": 100,
        "top_k": 1,
        "feed_window": 1,
        "random_seed": 0,
    })
    lab.data.setdefault("events", [])
    lab._online_events = []
    lab._offline_event_count = len(lab.data["events"])
    lab._popularity = Counter()
    lab.feed_cache = OrderedDict()
    lab._feed_sessions = {}
    lab._event_sequence = 0
    lab._last_mined_event_sequence = 0
    lab.pending_events = 0
    lab.version = 1
    lab.lock = threading.RLock()
    lab._background_mining = IdleMining()
    lab.contextual_features = lambda *_args, **_kwargs: {}
    return lab


def test_live_relation_is_proved_cached_and_invalidated_by_history_revision():
    lab = live_lab()
    first_plan = build_concept_relational_proof_plan(
        "C", ["H"], lab._llm_article_annotations, user_id="u",
    )
    lab.engine.register(first_plan)
    first = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )
    assert first["C"][REL_CONCEPT_CONTINUITY_SCOPE] == "recent"
    assert lab._relational_query_calls == 1
    assert lab._relational_query_roots == 1
    assert len(lab._live_relational_proof_ledger) == 1

    repeated = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )
    assert repeated == first
    assert lab._relational_query_calls == 1

    lab.data["users"]["u"]["history"].append("H2")
    revised_plan = build_concept_relational_proof_plan(
        "C", ["H", "H2"], lab._llm_article_annotations, user_id="u",
    )
    lab.engine.register(revised_plan)
    revised = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )
    assert revised["C"][REL_CONCEPT_CONTINUITY_SCOPE] == "recent"
    assert revised_plan.case_id != first_plan.case_id
    assert lab._relational_query_calls == 2
    assert lab._relational_query_roots == 3


def test_persisted_unknown_relation_is_not_recomputed_from_latest_profile():
    lab = live_lab()
    lab._feature_vocabulary = {}
    contexts = {"C": {
        "history_size_bucket": "cold",
        REL_CONCEPT_CONTINUITY_SCOPE: "unknown",
    }}
    specs = lab._candidate_specs("u", ["C"], contexts)
    assert REL_CONCEPT_CONTINUITY_SCOPE not in specs[0][2]
    assert lab.engine.query_calls == []


def test_unknown_or_truncated_history_cannot_manufacture_none():
    lab = live_lab()
    lab.data["users"]["u"]["history"] = ["missing"]
    unknown = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )
    assert REL_CONCEPT_CONTINUITY_SCOPE not in unknown["C"]

    lab.data["users"]["u"]["history"] = ["D"] * 50
    complete = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )
    assert complete["C"][REL_CONCEPT_CONTINUITY_SCOPE] == "none"

    lab.data["users"]["u"]["history"] = ["D"] * 51
    truncated = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )
    assert REL_CONCEPT_CONTINUITY_SCOPE not in truncated["C"]


def test_live_older_proof_abstains_when_recent_annotations_are_incomplete():
    lab = live_lab()
    lab._articles["N"] = {
        "id": "N", "topic": "business", "subcategory": "markets",
    }
    lab._llm_article_annotations["N"] = annotation("N", "neutral")
    history = ["H", "N", "N", "N", "N", "missing"]
    lab.data["users"]["u"]["history"] = list(history)
    plan = build_concept_relational_proof_plan(
        "C", history, lab._llm_article_annotations, user_id="u",
    )
    assert plan.complete_recent_concept_evidence is False
    lab.engine.register(plan)

    projected = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )
    assert REL_CONCEPT_CONTINUITY_SCOPE not in projected["C"]
    assert projected["C"][REL_CONCEPT_CONTINUITY_PROOF_IDS] == []
    assert lab._live_relational_proof_ledger == {}


def test_missing_or_malformed_history_abstains_but_explicit_empty_is_complete():
    for profile in ({}, {"history": None}, {"history": "H"},
                    {"history": [None]}, {"history": [""]}):
        lab = live_lab()
        lab.data["users"]["u"] = profile
        projected = lab._live_relational_features(
            "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
        )
        assert REL_CONCEPT_CONTINUITY_SCOPE not in projected["C"]
        assert projected["C"][REL_CONCEPT_CONTINUITY_PROOF_IDS] == []

    lab = live_lab()
    lab.data["users"]["u"] = {"history": []}
    projected = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )
    assert projected["C"][REL_CONCEPT_CONTINUITY_SCOPE] == "none"
    assert projected["C"][REL_CONCEPT_CONTINUITY_PROOF_IDS] == []


def test_positive_event_changes_relation_from_none_to_proof_derived_recent():
    lab = event_ready(live_lab())
    lab.data["users"]["u"]["history"] = ["D"]
    before = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )
    assert before["C"][REL_CONCEPT_CONTINUITY_SCOPE] == "none"
    assert before["C"][REL_CONCEPT_CONTINUITY_PROOF_IDS] == []

    result = lab.event("u", "H", "click", context={})
    assert result["profile_changed"] is True
    assert lab.data["users"]["u"]["history"] == ["D", "H"]
    revised_plan = build_concept_relational_proof_plan(
        "C", ["D", "H"], lab._llm_article_annotations, user_id="u",
    )
    lab.engine.register(revised_plan)

    after = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )
    references = after["C"][REL_CONCEPT_CONTINUITY_PROOF_IDS]
    assert after["C"][REL_CONCEPT_CONTINUITY_SCOPE] == "recent"
    assert len(references) == 1
    assert references[0] in lab._live_relational_proof_ledger


def test_evaluation_activation_audit_counts_only_proof_positive_relation_use():
    lab = object.__new__(Lab)
    lab.config = {"aggregation": "weighted", "pair_aggregation": "proof_margin"}
    lab._last_point_completeness = {"complete": True}
    lab._candidate_relational_proof_refs = {
        "case_recent": {
            REL_CONCEPT_CONTINUITY_PROOF_IDS: ("proof_1",),
        },
        "case_none": {
            REL_CONCEPT_CONTINUITY_PROOF_IDS: (),
        },
    }
    lab.mined_rules = [
        {"id": "point_recent", "premises": [
            [REL_CONCEPT_CONTINUITY_SCOPE, "recent"],
        ]},
        {"id": "point_none", "premises": [
            [REL_CONCEPT_CONTINUITY_SCOPE, "none"],
        ]},
    ]
    pair_field = f"pair_{REL_CONCEPT_CONTINUITY_SCOPE}"
    lab.pair_rules = [
        {"id": "pair_left", "proof_channel_id": "channel_left",
         "premises": [[pair_field, "left"]]},
        {"id": "pair_equal", "proof_channel_id": "channel_equal",
         "premises": [[pair_field, "equal"]]},
    ]
    point_specs = [
        ("C", "case_recent", {REL_CONCEPT_CONTINUITY_SCOPE: "recent"}, {
            REL_CONCEPT_CONTINUITY_SCOPE: "recent",
            REL_CONCEPT_CONTINUITY_PROOF_IDS: ["proof_1"],
        }),
        ("D", "case_none", {REL_CONCEPT_CONTINUITY_SCOPE: "none"}, {
            REL_CONCEPT_CONTINUITY_SCOPE: "none",
            REL_CONCEPT_CONTINUITY_PROOF_IDS: [],
        }),
    ]
    pair_specs = [
        ("pair_left_case", {pair_field: "left"}),
        ("pair_equal_case", {pair_field: "equal"}),
    ]
    audit = lab._relational_evaluation_activation_audit(
        point_specs, pair_specs,
    )
    assert audit["proof_positive_candidate_contexts"] == 1
    assert audit["point"]["relational_rule_activations"] == 2
    assert audit["point"]["proof_positive_relational_rule_activations"] == 1
    assert audit["pair"]["relational_rule_activations"] == 2
    assert audit["pair"]["proof_positive_relational_rule_activations"] == 1
    assert audit["proof_gate_complete"] is True


def test_served_proof_ids_reach_accepted_event_and_resolve():
    lab = event_ready(live_lab())
    plan = build_concept_relational_proof_plan(
        "C", ["H"], lab._llm_article_annotations, user_id="u",
    )
    lab.engine.register(plan)
    projected = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )["C"]
    references = projected[REL_CONCEPT_CONTINUITY_PROOF_IDS]
    lab.data["candidate_sets"] = {"u": ["C"]}

    def scored(_user, candidates, _contexts, limit=0, **_options):
        return [{
            "article": lab._articles[aid],
            "relational_evidence": {
                "scopes": {
                    REL_CONCEPT_CONTINUITY_SCOPE:
                        projected[REL_CONCEPT_CONTINUITY_SCOPE],
                },
                "proof_ids": {
                    REL_CONCEPT_CONTINUITY_PROOF_IDS: list(references),
                },
            },
        } for aid in candidates]

    lab.score = scored
    page = lab.feed_page("u", limit=1)
    served = page["feed"][0]
    assert served["context"][REL_CONCEPT_CONTINUITY_PROOF_IDS] == references
    result = lab.event(
        "u", "C", "click",
        served["context"], served["impression"],
        feed_session=page["session"],
        queue_revision=page["queue_revision"],
        feed_position=page["position"],
    )
    assert result["recorded_context"][REL_CONCEPT_CONTINUITY_PROOF_IDS] == references
    assert lab.data["events"][-1][REL_CONCEPT_CONTINUITY_PROOF_IDS] == references
    assert all(
        proof_id in lab._live_relational_proof_ledger
        for proof_id in references
    )


def test_positive_scope_without_proof_reference_is_rejected():
    lab = live_lab()
    with pytest.raises(ValueError, match="requires proof references"):
        lab._validate_relational_context_provenance(
            {REL_CONCEPT_CONTINUITY_SCOPE: "recent",
             REL_CONCEPT_CONTINUITY_PROOF_IDS: []},
            user="u", article="C",
        )


def test_relational_scope_is_bound_to_family_recency_and_one_history():
    lab = live_lab()
    base = {
        "user_id": "u", "candidate_id": "C", "case_id": "case_1",
        "scope_id": "scope_1", "causal_history_id": "history_1",
    }
    lab._live_relational_proof_ledger = {
        "concept_old": {
            **base, "relation_family": "canonical_concept_continuity",
            "recency": "older",
        },
        "concept_other_history": {
            **base, "case_id": "case_2", "scope_id": "scope_2",
            "causal_history_id": "history_2",
            "relation_family": "canonical_concept_continuity",
            "recency": "recent",
        },
        "entity_recent": {
            **base, "relation_family": "wikidata_entity_continuity",
            "recency": "recent",
        },
    }
    with pytest.raises(ValueError, match="recency"):
        lab._validate_relational_context_provenance({
            REL_CONCEPT_CONTINUITY_SCOPE: "recent",
            REL_CONCEPT_CONTINUITY_PROOF_IDS: ["concept_old"],
        }, user="u", article="C")
    with pytest.raises(ValueError, match="another family"):
        lab._validate_relational_context_provenance({
            REL_CONCEPT_CONTINUITY_SCOPE: "recent",
            REL_CONCEPT_CONTINUITY_PROOF_IDS: ["entity_recent"],
        }, user="u", article="C")
    with pytest.raises(ValueError, match="different causal histories"):
        lab._validate_relational_context_provenance({
            REL_CONCEPT_CONTINUITY_SCOPE: "recent",
            REL_CONCEPT_CONTINUITY_PROOF_IDS: ["concept_other_history"],
            REL_ENTITY_CONTINUITY_SCOPE: "recent",
            REL_ENTITY_CONTINUITY_PROOF_IDS: ["entity_recent"],
        }, user="u", article="C")
    with pytest.raises(ValueError, match="invalid relational scope"):
        lab._validate_relational_context_provenance({
            REL_CONCEPT_CONTINUITY_SCOPE: "very_recent",
            REL_CONCEPT_CONTINUITY_PROOF_IDS: [],
        }, user="u", article="C")


def test_direct_event_cannot_inject_server_issued_relational_context():
    lab = event_ready(live_lab())
    with pytest.raises(ValueError, match="verified live feed item"):
        lab.event("u", "C", "click", context={
            REL_CONCEPT_CONTINUITY_SCOPE: "none",
            REL_CONCEPT_CONTINUITY_PROOF_IDS: [],
        })


def test_score_cache_key_is_sensitive_to_relational_proof_origins():
    lab = object.__new__(Lab)
    lab.data = {"users": {"u": {}}}
    lab._articles = {"C": {"id": "C"}}
    lab.config = {"top_k": 1}
    lab.version = 1
    lab.feed_cache = OrderedDict()
    lab._feed_rank_cache_hits = 0
    lab._feed_rank_cache_misses = 0
    lab.article = lambda aid: lab._articles[aid]
    lab._serving_reasoner_timeout = lambda: 1.0
    calls = []

    def specs(_user, _candidates, contexts):
        references = tuple(contexts["C"][REL_CONCEPT_CONTINUITY_PROOF_IDS])
        calls.append(references)
        return [("C", "case", {}, {
            REL_CONCEPT_CONTINUITY_SCOPE: "recent",
            REL_CONCEPT_CONTINUITY_PROOF_IDS: list(references),
        })]

    lab._candidate_specs = specs
    lab._ensure_candidate_specs = lambda *_args, **_kwargs: None
    lab._proofs_for_specs = lambda *_args, **_kwargs: ([[]], 0)
    lab._rank = lambda rows, *_args, **_kwargs: [{
        "proof_ids": list(rows[0][3][REL_CONCEPT_CONTINUITY_PROOF_IDS]),
    }]
    first_context = {
        "C": {
            REL_CONCEPT_CONTINUITY_SCOPE: "recent",
            REL_CONCEPT_CONTINUITY_PROOF_IDS: ["proof_1"],
        },
    }
    second_context = copy.deepcopy(first_context)
    second_context["C"][REL_CONCEPT_CONTINUITY_PROOF_IDS] = ["proof_2"]

    assert lab.score("u", ["C"], first_context)[0]["proof_ids"] == ["proof_1"]
    assert lab.score("u", ["C"], second_context)[0]["proof_ids"] == ["proof_2"]
    assert calls == [("proof_1",), ("proof_2",)]


def test_background_promotion_clears_caches_but_retains_served_provenance():
    lab = event_ready(live_lab())
    plan = build_concept_relational_proof_plan(
        "C", ["H"], lab._llm_article_annotations, user_id="u",
    )
    lab.engine.register(plan)
    projected = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )["C"]
    proof_id = projected[REL_CONCEPT_CONTINUITY_PROOF_IDS][0]
    proof_record = copy.deepcopy(lab._live_relational_proof_ledger[proof_id])
    lab._live_relational_proof_ledger["unreferenced"] = {"unused": True}
    session = "a" * 32
    impression = f"live_{session}_0_1"
    served_context = {
            REL_CONCEPT_CONTINUITY_SCOPE: "recent",
            REL_CONCEPT_CONTINUITY_PROOF_IDS: [proof_id],
    }
    lab._feed_sessions = {session: {
        "user": "u", "items": [], "queue": [], "position": 1,
        "source_position": 1, "queue_revision": 0, "version": lab.version,
        "last_feedback_revision": None, "last_access": 0.0,
        "feedback_contexts": {(impression, "C"): served_context},
        "feedback_positions": {(impression, "C"): 1},
        "deliveries": [{
            "start_position": 0, "end_position": 1, "queue_revision": 0,
            "rows": [],
        }],
    }}

    old_engine = ClosableEngine(1)
    lab.engine = old_engine
    lab._closed = False
    lab._prune_mining_workspace_cache = lambda: {
        "removed": [], "support_removed": [], "error": None,
    }
    lab.last_mined_at = None
    lab.last_mining = {}
    lab._loaded_candidates = {"candidate"}
    lab._loaded_pairs = {"pair"}
    lab._loaded_point_channels = {"point"}
    lab._point_channel_proof_cache = {"point": True}
    lab._point_channel_templates = {"point": True}
    lab._point_query_calls = lab._point_query_roots = 1
    lab._point_pruned_query_roots = 1
    lab._point_channel_activations = 1
    lab._point_reused_channel_activations = 1
    lab._last_point_completeness = {"complete": True}
    lab._last_point_cache_stats = {"hits": 1}
    lab._loaded_pair_channels = {"pair"}
    lab._proof_cache = {"point": True}
    lab._pair_proof_cache = {"pair": True}
    lab._pair_channel_proof_cache = {"pair": True}
    lab._pair_channel_templates = {"pair": True}
    lab._pair_case_attrs = {"pair": True}
    lab._pair_margin_cache = {"pair": True}
    lab._pair_proof_origins = {"pair": True}
    lab._candidate_relational_proof_refs = {"candidate": {}}
    lab._relational_query_calls = lab._relational_query_roots = 1
    lab.feed_cache = OrderedDict((("feed", True),))

    staged = copy.copy(lab)
    staged.engine = ClosableEngine(2)
    staged.config = lab.config.copy()
    staged.version = lab.version + 1
    staged.last_mining = {}
    for field in BACKGROUND_MODEL_FIELDS:
        if not hasattr(staged, field):
            setattr(staged, field, None)
    snapshot = MiningSnapshot(
        event_sequence=lab._event_sequence,
        base_version=lab.version,
        payload=staged,
    )

    outcome = lab._promote_background_mining(snapshot, staged)
    assert outcome["promoted"] is True
    assert old_engine.closed is True
    assert lab._relational_feature_cache == {}
    assert lab._loaded_relational_statements == set()
    assert lab._candidate_relational_proof_refs == {}
    assert lab._relational_query_calls == 0
    assert lab._relational_query_roots == 0
    assert lab._live_relational_proof_ledger == {proof_id: proof_record}

    accepted = lab.event(
        "u", "C", "click", served_context, impression,
        feed_session=session, queue_revision=0, feed_position=1,
    )
    assert accepted["recorded_context"][
        REL_CONCEPT_CONTINUITY_PROOF_IDS
    ] == [proof_id]
    assert lab.data["events"][-1][REL_CONCEPT_CONTINUITY_PROOF_IDS] == [proof_id]


def test_background_snapshot_carries_records_referenced_by_live_events():
    lab = event_ready(live_lab())
    lab.instance_id = "live-test"
    lab._closed = False
    lab.petta = object()
    plan = build_concept_relational_proof_plan(
        "C", ["H"], lab._llm_article_annotations, user_id="u",
    )
    lab.engine.register(plan)
    projected = lab._live_relational_features(
        "u", ["C"], {REL_CONCEPT_CONTINUITY_SCOPE},
    )["C"]
    proof_id = projected[REL_CONCEPT_CONTINUITY_PROOF_IDS][0]
    proof_record = copy.deepcopy(lab._live_relational_proof_ledger[proof_id])
    lab.data["events"].append({
        "user": "u", "article": "C", "action": "click",
        REL_CONCEPT_CONTINUITY_SCOPE: "recent",
        REL_CONCEPT_CONTINUITY_PROOF_IDS: [proof_id],
    })

    with patch(
        "recommendation.app.server.IsolatedPeTTaChainer",
        lambda: ClosableEngine(2),
    ):
        snapshot = lab._capture_background_mining()

    assert snapshot.payload._relational_feature_cache == {}
    assert snapshot.payload._loaded_relational_statements == set()
    assert snapshot.payload._candidate_relational_proof_refs == {}
    assert snapshot.payload._live_relational_proof_ledger == {
        proof_id: proof_record,
    }
