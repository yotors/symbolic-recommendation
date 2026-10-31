from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from recommendation.adapters.ebnerd import load_ebnerd


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_loader_builds_causal_global_and_recent_ctr(tmp_path):
    _write(tmp_path / "articles.parquet", [
        {
            "article_id": article_id,
            "title": f"Article {article_id}",
            "subtitle": "Summary",
            "url": f"https://example.test/{article_id}",
            "article_type": "article_default",
            "topics": [topic],
            "category_str": topic,
            "subcategory": [article_id],
            "ner_clusters": [f"entity_{article_id}"],
            "entity_groups": ["group"],
            "sentiment_label": "neutral",
            "premium": False,
        }
        for article_id, topic in ((1, "history"), (2, "news"), (3, "sport"))
    ])
    _write(tmp_path / "train/history.parquet", [
        {"user_id": 10, "article_id_fixed": [1]},
    ])
    _write(tmp_path / "validation/history.parquet", [
        {"user_id": 20, "article_id_fixed": [1]},
    ])
    start = datetime(2023, 5, 18, 8)
    train = [
        {
            "impression_id": impression,
            "impression_time": start + timedelta(minutes=offset),
            "user_id": 10,
            "session_id": 1,
            "device_type": 1,
            "article_ids_inview": [2, 3],
            "article_ids_clicked": [2],
            "next_read_time": 70,
            "next_scroll_percentage": 90.0,
        }
        for offset, impression in enumerate((100, 101))
    ]
    _write(tmp_path / "train/behaviors.parquet", train)
    _write(tmp_path / "validation/behaviors.parquet", [{
        **train[0],
        "impression_id": 200,
        "impression_time": start + timedelta(hours=1),
        "user_id": 20,
    }])

    data = load_ebnerd(
        tmp_path, max_train_cases=4, max_eval_impressions=1,
        content_workspace=False,
    )

    first = data["events"][:2]
    second = data["events"][2:]
    assert {row["empirical_ctr"] for row in first} == {0.1}
    assert {row["article"]: row["empirical_ctr"] for row in second} == {
        "article_2": 0.18181818,
        "article_3": 0.09090909,
    }
    assert {row["article"]: row["recent_empirical_ctr"] for row in second} == {
        "article_2": 0.31818182,
        "article_3": 0.07575758,
    }

    validation = data["tests"][0]["candidate_context"]
    assert validation["article_2"]["empirical_ctr"] == 0.25
    assert validation["article_3"]["empirical_ctr"] == 0.08333333
    assert validation["article_2"]["recent_empirical_ctr"] == 0.46428571
    assert validation["article_3"]["recent_empirical_ctr"] == 0.05952381


def test_loader_sampling_is_deterministic_and_complete(tmp_path):
    # The causal-value test exercises the full Parquet boundary. This test keeps
    # the core contract visible without introducing a second synthetic corpus.
    from recommendation.adapters.ebnerd import _selected_impressions

    rows = [
        {
            "impression_id": impression,
            "article_ids_inview": candidates,
            "article_ids_clicked": [candidates[0]],
        }
        for impression, candidates in ((1, [1, 2]), (2, [2, 3, 4]), (3, [4, 5]))
    ]
    rows.append({
        "impression_id": 4,
        "article_ids_inview": [6, 7],
        "article_ids_clicked": [6, 7],
    })
    first = _selected_impressions(rows, split="train", seed=7, max_cases=4)
    second = _selected_impressions(rows, split="train", seed=7, max_cases=4)
    assert first == second
    assert sum(
        len(row["article_ids_inview"])
        for row in rows if row["impression_id"] in first
    ) <= 4
    assert 4 not in first
