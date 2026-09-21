from recommendation.app.server import parse_petta_target_rules


def test_decodes_petta_target_metrics_without_recalculation():
    raw = [
        '((targetScoreOf (((topic $_1 "news") '
        '(engagement $_1 "click")) '
        '(CTV (STV 0.75 0.2) (STV 0.25 0.3))) '
        '(Contingency 3 1 1 3) (AUC 0.75) (AUC-Gain 0.25) '
        '(Youden-J 0.5) (WRAcc 0.125) (Information-Gain 0.1887) '
        '(Log-Odds 1.6946) (Parent-Precision 0.5) '
        '(Incremental-Precision 0.25) (Incremental-WRAcc 0.0625) 3))'
    ]
    rules = parse_petta_target_rules(raw, features=("topic",))
    assert len(rules) == 1
    rule = rules[0]
    assert rule["premises"] == (("topic", "news"),)
    assert rule["target_contingency"] == {
        "n11": 3, "n10": 1, "n01": 1, "n00": 3,
    }
    assert rule["target_auc"] == 0.75
    assert rule["target_auc_gain"] == 0.25
    assert rule["target_parent_precision"] == 0.5
    assert rule["target_incremental_precision"] == 0.25
    assert rule["target_incremental_wracc"] == 0.0625
    assert rule["petta_target_aware"] is True
    assert rule["source"].endswith("#target-aware")
