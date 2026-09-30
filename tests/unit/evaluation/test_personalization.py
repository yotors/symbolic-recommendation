from __future__ import annotations

from recommendation.evaluation.personalization import (
    bind_personalization_cohort,
    chronological_user_cases,
    compare_personalization_modes,
    PrivateReplayState,
    private_contexts,
    run_prequential_personalization,
)


class FakeLab:
    def __init__(self, cases):
        self.data = {
            "events": [], "tests": cases,
            "subcategory_transition_model": {},
        }
        self._online_events = []
        self.version = 3
        self._articles = {
            "A": {"id": "A", "title": "AI", "topic": "tech",
                  "subcategory": "ai", "format": "short"},
            "B": {"id": "B", "title": "Football", "topic": "sports",
                  "subcategory": "football", "format": "short"},
        }
        self._article_entity_vectors = {}
        self._article_text_vectors = {}
        self._title_idf_model = {}
        self._lexical_idf_model = {}
        self._semantic_workspace_model = {}
        self._recency_workspace = {}
        self._llm_workspace = {}
        self._llm_article_annotations = {}
        self.seen_contexts = []

    def serving_model(self):
        return {"model_sha256": "frozen-model"}

    @staticmethod
    def _ranking_signature(row):
        return (row["score"],)

    def score(self, user, candidates, contexts, **_kwargs):
        self.seen_contexts.append({key: dict(value) for key, value in contexts.items()})
        rows = [
            {
                "article": self._articles[article],
                "score": 0.0 if contexts.get(article, {}).get(
                    "recent_negative_match"
                ) == "exact" else 1.0,
                "engine": "PeTTaChainer",
                "score_method": "proof_gated_base_rate_posterior",
                "proofs": ["proof"],
            }
            for article in candidates
        ]
        return sorted(rows, key=lambda row: -row["score"])


def case(identity, timestamp, candidates, relevant):
    return {
        "id": identity,
        "source_impression_id": identity,
        "user": "U",
        "timestamp": timestamp,
        "timestamp_source": "official_mind_behaviors",
        "source_line_number": int(identity[1:]),
        "history": [],
        "hour": "9AM",
        "candidates": list(candidates),
        "relevant": list(relevant),
        "candidate_context": {
            article: {
                "history_size_bucket": "cold",
                "ctr_bucket": "cold",
                "position_bucket": "top" if index == 0 else "early",
            }
            for index, article in enumerate(candidates)
        },
    }


def test_replay_uses_only_completed_prior_timestamp_groups():
    cases = [
        # Burn-in: A clicked, B exposed but not clicked.
        case("I1", "2019-11-15T09:00:00", ["A", "B"], ["A"]),
        # B must be negative evidence here, never in I1 itself.
        case("I2", "2019-11-15T10:00:00", ["B", "A"], ["A"]),
    ]
    lab = FakeLab(cases)

    result = run_prequential_personalization(
        lab, bootstrap_repetitions=100, update_mode="click_and_nonclick"
    )

    assert result["global_model_frozen"] is True
    assert result["burn_in_impressions"] == 1
    assert result["evaluated_impressions"] == 1
    assert result["mean_delta"] == 0.5
    # First score call is frozen I2; second is adaptive I2.
    assert lab.seen_contexts[0]["B"]["recent_negative_match"] == "none"
    assert lab.seen_contexts[1]["B"]["recent_negative_match"] == "exact"
    assert result["per_impression"][0]["prior_impressions"] == 1
    assert lab.data["events"] == []
    assert lab._online_events == []
    assert lab.version == 3
    assert result["reasoner_audit"]["all_rows_pettachainer"] is True


def test_click_only_arm_does_not_invent_nonclick_evidence():
    cases = [
        case("I1", "2019-11-15T09:00:00", ["A", "B"], ["A"]),
        case("I2", "2019-11-15T10:00:00", ["B", "A"], ["A"]),
    ]
    lab = FakeLab(cases)

    result = run_prequential_personalization(
        lab, bootstrap_repetitions=100, update_mode="click_only"
    )

    assert result["mean_delta"] == 0.0
    assert lab.seen_contexts[1]["B"]["recent_negative_match"] == "none"


def test_chronology_rejects_synthetic_source_row_timestamps():
    invalid = case("I1", "valid-row-000000001", ["A", "B"], ["A"])
    invalid["timestamp_source"] = "reczoo_source_row"

    try:
        chronological_user_cases([invalid])
    except ValueError as exc:
        assert "official MIND behavior timestamps" in str(exc)
    else:
        raise AssertionError("synthetic chronology was accepted")


def test_three_arm_comparison_reuses_exact_frozen_control():
    cases = [
        case("I1", "2019-11-15T09:00:00", ["A", "B"], ["A"]),
        case("I2", "2019-11-15T10:00:00", ["B", "A"], ["A"]),
    ]
    result = compare_personalization_modes(
        FakeLab(cases), bootstrap_repetitions=100,
    )

    assert result["control_reconstruction_exact"] is True
    assert result["arms"]["no_private_updates"]["auc"] == 0.5
    assert result["arms"]["click_only"]["mean_delta"] == 0.0
    assert result["arms"]["click_and_nonclick"]["mean_delta"] == 0.5
    assert result["arms"]["click_and_nonclick"]["reasoner_audit"][
        "all_rows_pettachainer"
    ] is True
    assert "not an explicit user Skip" in result["zero_label_semantics"]


def test_bind_cohort_preserves_frozen_article_space_and_adds_user():
    cases = [
        case("I1", "2019-11-15T09:00:00", ["A", "B"], ["A"]),
        case("I2", "2019-11-15T10:00:00", ["B", "A"], ["A"]),
    ]
    cases[0]["history"] = ["A"]
    cases[1]["history"] = ["A"]
    workspace = {
        "articles": [
            {"id": "A", "topic": "tech", "subcategory": "ai"},
            {"id": "B", "topic": "sports", "subcategory": "football"},
        ],
        "users": {},
        "tests": [],
        "metadata": {"llm_workspace": {"frozen": True}},
        "llm_article_annotations": {},
    }
    cohort = {
        "articles": [
            {"id": "A", "topic": "tech", "subcategory": "ai"},
            {"id": "B", "topic": "sports", "subcategory": "football"},
        ],
        "tests": cases,
        "metadata": {
            "official_behavior_source": "dev.zip",
            "official_behavior_join": {"selected_users": 1},
            "evaluation_chronology": "official",
        },
    }

    bound = bind_personalization_cohort(workspace, cohort)

    assert [article["id"] for article in bound["articles"]] == ["A", "B"]
    assert "llm_history_coverage" not in bound["tests"][0][
        "candidate_context"
    ]["A"]
    assert bound["users"]["U"]["history"] == ["A"]
    assert bound["metadata"]["personalization_cohort_binding"][
        "workspace_extended"
    ] is False
    assert bound["metadata"]["personalization_cohort_binding"][
        "static_feature_models_exact"
    ] is True
    assert bound["metadata"]["personalization_cohort_binding"][
        "history_features_rebuilt_from_workspace_for_both_arms"
    ] is True


def test_bind_rejects_a_different_feature_model_snapshot():
    cases = [
        case("I1", "2019-11-15T09:00:00", ["A", "B"], ["A"]),
        case("I2", "2019-11-15T10:00:00", ["B", "A"], ["A"]),
    ]
    articles = [
        {"id": "A", "topic": "tech", "subcategory": "ai"},
        {"id": "B", "topic": "sports", "subcategory": "football"},
    ]
    workspace = {
        "articles": articles, "users": {}, "tests": [],
        "metadata": {}, "title_idf_model": {"version": "one"},
    }
    cohort = {
        "articles": articles, "tests": cases, "metadata": {},
        "title_idf_model": {"version": "two"},
    }

    try:
        bind_personalization_cohort(workspace, cohort)
    except ValueError as exc:
        assert "feature model mismatch" in str(exc)
    else:
        raise AssertionError("mismatched feature snapshots were accepted")


def test_private_click_regenerates_relational_proof_for_augmented_history():
    lab = FakeLab([])
    lab.config = {"relational_evidence_mode": "chained"}
    lab.data["users"] = {"U": {"history": []}}
    seen_histories = []

    def derive(user, candidates, _required):
        seen_histories.append(tuple(lab.data["users"][user]["history"]))
        return {
            article: {
                "rel_entity_continuity_scope": "recent",
                "rel_entity_continuity_proof_ids": [f"proof_{article}"],
            }
            for article in candidates
        }

    lab._live_relational_features = derive
    replay_state = PrivateReplayState(())
    replay_state.clicks.append("A")
    replay_state.completed_impressions = 1

    contexts = private_contexts(
        lab,
        case("I2", "2019-11-15T10:00:00", ["B", "A"], ["A"]),
        replay_state,
    )

    assert seen_histories == [("A",)]
    assert contexts["B"]["rel_entity_continuity_scope"] == "recent"
    assert contexts["B"]["rel_entity_continuity_proof_ids"] == ["proof_B"]
    assert lab.data["users"]["U"]["history"] == []


def test_comparison_rejects_engine_labels_without_returned_proofs():
    class NoProofLab(FakeLab):
        def score(self, *args, **kwargs):
            rows = super().score(*args, **kwargs)
            for row in rows:
                row["proofs"] = []
            return rows

    cases = [
        case("I1", "2019-11-15T09:00:00", ["A", "B"], ["A"]),
        case("I2", "2019-11-15T10:00:00", ["B", "A"], ["A"]),
    ]
    try:
        compare_personalization_modes(
            NoProofLab(cases), bootstrap_repetitions=100,
        )
    except RuntimeError as exc:
        assert "PeTTaChainer proof path" in str(exc)
    else:
        raise AssertionError("empty proof rows were accepted")
