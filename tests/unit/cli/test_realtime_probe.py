"""Focused checks for the external live-probe evidence gate."""

from recommendation.cli.realtime_probe import _negative_demotion_audit


def test_negative_demotion_requires_rank_and_proof_provenance():
    revision = {
        "negative_proof_candidates": [
            {"article": "n5", "rule_ids": ["feedback_skip_topic"]}
        ],
        "causal_demotion": {
            "article": "n5",
            "before_rank": 1,
            "after_rank": 3,
            "before_score": 0.8,
            "after_score": 0.2,
            "before_ranking_score": 0.9,
            "after_ranking_score": 0.1,
            "feedback_evidence_changed": True,
            "before_feedback_signature": None,
            "after_feedback_signature": "proof-v1",
            "score_method": "pettachainer_live_feedback_revision",
            "rule_ids": ["feedback_skip_topic"],
            "proofs": ["(by feedback_skip_topic fact_candidate_n5_topic)"],
        },
    }

    audit = _negative_demotion_audit(revision)

    assert audit["passed"] is True
    assert all(audit["checks"].values())


def test_negative_demotion_rejects_metadata_without_rule_in_proof():
    revision = {
        "negative_proof_candidates": [
            {"article": "n5", "rule_ids": ["feedback_skip_topic"]}
        ],
        "causal_demotion": {
            "article": "n5",
            "before_rank": 1,
            "after_rank": 3,
            "before_score": 0.8,
            "after_score": 0.2,
            "before_ranking_score": 0.9,
            "after_ranking_score": 0.1,
            "feedback_evidence_changed": True,
            "before_feedback_signature": None,
            "after_feedback_signature": "proof-v1",
            "score_method": "pettachainer_live_feedback_revision",
            "rule_ids": ["feedback_skip_topic"],
            "proofs": ["(unrelated proof)"],
        },
    }

    audit = _negative_demotion_audit(revision)

    assert audit["passed"] is False
    assert audit["checks"]["feedback_rule_present_in_proof"] is False


def test_negative_demotion_accepts_proven_score_drop_when_rank_is_unchanged():
    revision = {
        "negative_proof_candidates": [
            {"article": "n5", "rule_ids": ["feedback_skip_topic"]}
        ],
        "causal_demotion": {
            "article": "n5",
            "before_rank": 3,
            "after_rank": 3,
            "before_score": 0.8,
            "after_score": 0.2,
            "before_ranking_score": 0.7,
            "after_ranking_score": 0.7,
            "feedback_evidence_changed": True,
            "before_feedback_signature": None,
            "after_feedback_signature": "proof-v1",
            "score_method": "pettachainer_live_feedback_revision",
            "rule_ids": ["feedback_skip_topic"],
            "proofs": ["(by feedback_skip_topic fact_candidate_n5_topic)"],
        },
    }

    audit = _negative_demotion_audit(revision)

    assert audit["passed"] is True
    assert audit["checks"]["rank_demoted"] is False
    assert audit["checks"]["score_decreased"] is True


def test_negative_demotion_rejects_indirect_rank_only_movement():
    revision = {
        "negative_proof_candidates": [
            {"article": "n5", "rule_ids": ["feedback_skip_topic"]}
        ],
        "causal_demotion": {
            "article": "n5",
            "before_rank": 1,
            "after_rank": 3,
            "before_score": 0.2,
            "after_score": 0.2,
            "before_ranking_score": 0.4,
            "after_ranking_score": 0.4,
            "feedback_evidence_changed": True,
            "before_feedback_signature": None,
            "after_feedback_signature": "proof-v1",
            "score_method": "pettachainer_live_feedback_revision",
            "rule_ids": ["feedback_skip_topic"],
            "proofs": ["(by feedback_skip_topic fact_candidate_n5_topic)"],
        },
    }

    audit = _negative_demotion_audit(revision)

    assert audit["checks"]["rank_demoted"] is True
    assert audit["checks"]["score_decreased"] is False
    assert audit["checks"]["ranking_score_decreased"] is False
    assert audit["passed"] is False


def test_negative_demotion_accepts_own_ranking_score_drop():
    revision = {
        "negative_proof_candidates": [
            {"article": "n5", "rule_ids": ["feedback_skip_topic"]}
        ],
        "causal_demotion": {
            "article": "n5",
            "before_rank": 2,
            "after_rank": 2,
            "before_score": 0.2,
            "after_score": 0.2,
            "before_ranking_score": 0.6,
            "after_ranking_score": 0.4,
            "feedback_evidence_changed": True,
            "before_feedback_signature": "topic-v1",
            "after_feedback_signature": "exact-v2",
            "score_method": "pettachainer_live_feedback_revision",
            "rule_ids": ["feedback_skip_exact"],
            "proofs": ["(by feedback_skip_exact fact_candidate_n5_exact)"],
        },
    }

    audit = _negative_demotion_audit(revision)

    assert audit["checks"]["ranking_score_decreased"] is True
    assert audit["passed"] is True
