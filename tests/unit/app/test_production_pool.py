"""Production scorer-pool routing and launch contracts."""

from types import SimpleNamespace

import pytest

from recommendation.app.production_pool import (
    GatewayState,
    ScorerPool,
    Worker,
    cookie_shard,
    stable_shard,
)
from recommendation.app.server import Lab, _scorer_worker_command


def test_stable_shard_is_repeatable_and_bounded():
    first = [stable_shard(f"user-{index}", 4) for index in range(100)]
    second = [stable_shard(f"user-{index}", 4) for index in range(100)]
    assert first == second
    assert set(first).issubset({0, 1, 2, 3})
    assert len(set(first)) == 4


@pytest.mark.parametrize(
    ("header", "workers", "expected"),
    [
        ("a=1; recommendation_shard=2; b=3", 4, 2),
        ("recommendation_shard=4", 4, None),
        ("recommendation_shard=bad", 4, None),
        (None, 4, None),
    ],
)
def test_cookie_shard_rejects_invalid_routes(header, workers, expected):
    assert cookie_shard(header, workers) == expected


def test_worker_command_preserves_immutable_model_and_dataset():
    args = SimpleNamespace(
        max_train_cases=20_000,
        max_eval_impressions=500,
        seed=7,
        symbolic_data=None,
        semantic_data=None,
        replay_data="dataset/replay.json.gz",
        config_file=None,
        serving_model="model.json",
        text_embeddings=None,
        fixture=False,
        mind="dataset/MIND.zip",
        workers=2,
        serving_only=False,
    )
    command = _scorer_worker_command(args, 7171)
    assert command[:3] == (__import__("sys").executable, "-m", "recommendation")
    assert command[command.index("--port") + 1] == "7171"
    assert command[command.index("--workers") + 1] == "1"
    assert command[command.index("--replay-data") + 1] == "dataset/replay.json.gz"
    assert command[command.index("--serving-model") + 1] == "model.json"
    assert "--serving-only" in command
    assert "--mind" not in command


def test_stable_shard_rejects_empty_pool():
    with pytest.raises(ValueError, match="workers must be positive"):
        stable_shard("user", 0)


def test_serving_only_lab_refuses_mining():
    lab=object.__new__(Lab)
    lab.serving_only=True
    with pytest.raises(ValueError,match="cannot mine"):
        lab.mine()


def test_session_route_expires_when_worker_restarts():
    worker=Worker(index=0,port=7171,command=("scorer",))
    state=GatewayState(ScorerPool([worker]),backend_timeout=1)
    state.remember_session("session-1",0)
    assert state.session_shard("session-1")==(0,False)
    worker.restarts+=1
    assert state.session_shard("session-1")==(0,True)
    assert state.session_shard("session-1")==(None,False)
