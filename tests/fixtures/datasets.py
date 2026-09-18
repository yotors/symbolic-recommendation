"""Deterministic, label-safe datasets shared by integration tests."""

from __future__ import annotations

import copy
import hashlib

from recommendation.adapters.mind import history_feature_context
from recommendation.features.lexical_workspace import (
    build_lexical_workspace_facts,
    fit_lexical_idf_model,
)
from recommendation.features.text_embeddings import article_text
from recommendation.pipelines.llm_data import build_llm_projection


def lexical_fixture():
    """Return a tiny causal lexical-ranking dataset."""

    common = {"topic": "general", "subcategory": "general", "format": "article"}
    articles = [
        {
            **common,
            "id": "past",
            "title": "Telescope observes distant planets",
            "abstract": "Astronomy space research",
        },
        {
            **common,
            "id": "matched",
            "title": "Telescope finds distant planets",
            "abstract": "Space astronomy research",
        },
        {
            **common,
            "id": "unrelated",
            "title": "Football tournament stadium crowd",
            "abstract": "Soccer championship results",
        },
    ]
    by_id = {article["id"]: article for article in articles}
    model = fit_lexical_idf_model(articles)
    contexts = {}
    for candidate in articles[1:]:
        contexts[candidate["id"]] = {
            **history_feature_context(candidate, ["past"], by_id),
            **build_lexical_workspace_facts(candidate, [articles[0]], model),
        }
    events = []
    for index in range(24):
        for article_id, action in (("matched", "click"), ("unrelated", "skip")):
            events.append(
                {
                    "user": "reader",
                    "article": article_id,
                    "action": action,
                    "impression": f"closed_{index}",
                    **contexts[article_id],
                }
            )
    return {
        "articles": articles,
        "users": {"reader": {"topics": ["general"], "history": ["past"]}},
        "events": events,
        "tests": [
            {
                "user": "reader",
                "candidates": ["matched", "unrelated"],
                "relevant": ["matched"],
                "candidate_context": contexts,
                "history": ["past"],
            }
        ],
        "lexical_idf_model": model,
        # Strict-symbolic construction must remove these without opening them.
        "article_entity_vectors": {"matched": [1.0, 0.0]},
        "article_text_vectors": {"matched": [1.0, 0.0]},
        "metadata": {"text_embedding_sidecar": "/must/not/be/opened.npz"},
    }


def conditional_annotation_fixture():
    """Return a projected dataset with a stable conditional LLM preference."""

    common = {"format": "article", "subcategory": "general"}
    articles = [
        {**common, "id": "past", "title": "Space history", "abstract": "Astronomy"},
        {
            **common,
            "id": "same_match",
            "topic": "shared",
            "title": "Space discovery",
            "abstract": "Astronomy",
        },
        {
            **common,
            "id": "same_other",
            "topic": "shared",
            "title": "Football result",
            "abstract": "Sport",
        },
        {
            **common,
            "id": "diff_other",
            "topic": "left_topic",
            "title": "Football update",
            "abstract": "Sport",
        },
        {
            **common,
            "id": "diff_match",
            "topic": "right_topic",
            "title": "Space update",
            "abstract": "Astronomy",
        },
    ]
    events = []
    for index in range(24):
        pairs = (
            (("same_match", "click"), ("same_other", "skip"))
            if index % 2 == 0
            else (("diff_other", "click"), ("diff_match", "skip"))
        )
        for article_id, action in pairs:
            events.append(
                {
                    "user": "reader",
                    "article": article_id,
                    "action": action,
                    "impression": f"closed_{index:02d}",
                    "history_size_bucket": "light",
                }
            )
    source = {
        "articles": articles,
        "users": {"reader": {"topics": [], "history": ["past"]}},
        "events": events,
        "tests": [
            {
                "id": "conditional_eval",
                "user": "reader",
                "candidates": ["same_match", "same_other"],
                "history": ["past"],
                "relevant": ["same_match"],
                "labels": {"same_match": 1, "same_other": 0},
                "candidate_context": {
                    "same_match": {"history_size_bucket": "light"},
                    "same_other": {"history_size_bucket": "light"},
                },
            }
        ],
        "metadata": {"dataset": "conditional-llm-fixture"},
    }
    histories = copy.deepcopy(source)
    for event in histories["events"]:
        event["history"] = ["past"]
    annotations = {}
    for article in articles:
        concept = (
            "astronomy"
            if article["id"] in {"past", "same_match", "diff_match"}
            else "football"
        )
        annotations[article["id"]] = {
            "concepts": [concept],
            "format": "report",
            "event_types": [],
            "intents": [],
            "audiences": [],
            "provenance": {
                "article_content_sha256": hashlib.sha256(
                    article_text(article["title"], article["abstract"]).encode("utf-8")
                ).hexdigest()
            },
        }
    return build_llm_projection(source, histories, annotations)
