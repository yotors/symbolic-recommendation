from recommendation.evaluation.symbolic_experiment import nested_impression_sample


def _events():
    return [
        {"impression": "b", "row": 1},
        {"impression": "a", "row": 2},
        {"impression": "b", "row": 3},
        {"impression": "c", "row": 4},
        {"impression": "c", "row": 5},
    ]


def test_nested_sample_retains_whole_impressions_and_source_order():
    sample = nested_impression_sample(_events(), 3, seed=7)
    retained = {event["impression"] for event in sample}

    assert [event["row"] for event in sample] == sorted(
        event["row"] for event in sample
    )
    assert all(
        sum(event["impression"] == impression for event in sample)
        == sum(event["impression"] == impression for event in _events())
        for impression in retained
    )


def test_larger_limit_contains_smaller_nested_sample():
    small = nested_impression_sample(_events(), 2, seed=11)
    large = nested_impression_sample(_events(), 5, seed=11)

    assert {event["row"] for event in small} <= {
        event["row"] for event in large
    }


def test_zero_limit_keeps_every_event():
    assert nested_impression_sample(_events(), 0) == _events()
