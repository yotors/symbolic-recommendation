"""Unit checks for the synchronized external feed-load probe."""

import copy
import threading
import time
from urllib.parse import parse_qs, urlparse

import pytest

import recommendation.cli.concurrent_feed_probe as probe_module
from recommendation.cli.concurrent_feed_probe import _peak_overlap, probe, run_level


def test_peak_overlap_treats_intervals_as_half_open():
    assert _peak_overlap([(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)]) == 2
    assert _peak_overlap([]) == 0


def test_peak_overlap_rejects_reversed_interval():
    with pytest.raises(ValueError, match="finishes before"):
        _peak_overlap([(2.0, 1.0)])


def test_run_level_releases_each_round_concurrently():
    active = 0
    peak = 0
    requested_paths = []
    lock = threading.Lock()

    def fake_request(base_url, path, timeout):
        nonlocal active, peak
        requested_paths.append(path)
        started = time.time()
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        finished = time.time()
        return {
            "feed": [{
                "article": {"id": "n1"},
                "engine":"PeTTaChainer",
                "pairwise_score":0.5,
            }],
            "session":"session-1",
            "rule_version":3,
        }, {
            "method": "GET",
            "path": path,
            "status": 200,
            "elapsed_ms": (finished - started) * 1000.0,
            "started_at_epoch": started,
            "finished_at_epoch": finished,
            "response_bytes": 10,
        }

    result = run_level(
        "http://127.0.0.1:7070",
        users=[f"u{index}" for index in range(8)],
        concurrency=4,
        rounds=2,
        page_size=5,
        expected_rule_version=3,
        request_timeout=1.0,
        request_fn=fake_request,
    )

    assert result["passed"] is True
    assert result["requests_succeeded"] == 8
    assert result["requests_failed"] == 0
    assert result["client_observed_peak_in_flight"] == 4
    assert peak == 4
    assert len({parse_qs(urlparse(path).query)["user"][0]
                for path in requested_paths}) == 8
    assert all(parse_qs(urlparse(path).query)["limit"] == ["5"]
               for path in requested_paths)


def _fake_live_server(*,drift_after_feed=False):
    lock=threading.Lock()
    requested_users=[]
    state={
        "instance_id":"lab-instance-1",
        "version":4,
        "users":[{"id":f"u{index}"} for index in range(8)],
        "config":{
            "feed_window":40,"pairwise_opponents":0,
            "max_pair_comparisons":32768,
        },
        "dataset":{"name":"fake-mind","sha256":"dataset-1"},
        "engine":{
            "worker_pid":4321,
            "feed_rank_cache":{"entries":0,"hits":0,"misses":0},
            "point_case_cache_entries":0,
            "pair_case_cache_entries":0,
            "point_channel_cache_entries":0,
            "pair_channel_cache_entries":0,
            "point_reasoner_query_calls":0,
            "pair_reasoner_query_calls":0,
        },
    }

    def request(base_url,path,timeout=1.0):
        started=time.time()
        if path=="/api/state":
            payload=copy.deepcopy(state)
        else:
            query=parse_qs(urlparse(path).query)
            assert query["limit"]==["5"]
            user=query["user"][0]
            with lock:
                requested_users.append(user)
                cache=state["engine"]["feed_rank_cache"]
                cache["misses"]+=1; cache["entries"]+=1
                if drift_after_feed:
                    state["config"]["feed_window"]=41
            payload={
                "feed":[{
                    "article":{"id":f"article-{user}"},
                    "engine":"PeTTaChainer","pairwise_score":0.5,
                }],
                "session":f"session-{user}","rule_version":4,
            }
        finished=time.time()
        return payload,{
            "method":"GET","path":path,"status":200,
            "elapsed_ms":(finished-started)*1000.0,
            "started_at_epoch":started,"finished_at_epoch":finished,
            "response_bytes":10,
        }
    return request,requested_users


def test_probe_verifies_fresh_misses_and_allocates_distinct_users(monkeypatch):
    request,requested_users=_fake_live_server()
    monkeypatch.setattr(probe_module,"_json_request",request)

    result=probe(
        "http://127.0.0.1:7070",
        concurrency_levels=[1,2],rounds=1,warmup=1,page_size=5,
        random_seed=7,request_timeout=1.0,server_pid=None,
        resource_interval=0.01,
    )

    assert result["passed"] is True
    assert len(requested_users)==4
    assert len(set(requested_users))==4
    assert all(level["ranked_feed_cache_misses_verified"]
               for level in result["levels"])
    assert all(level["scorer_identity_stable"] for level in result["levels"])
    assert all(level["server_counter_delta"]["feed_rank_cache_hits"]==0
               for level in result["levels"])


def test_probe_fails_when_config_identity_drifts(monkeypatch):
    request,_requested_users=_fake_live_server(drift_after_feed=True)
    monkeypatch.setattr(probe_module,"_json_request",request)

    result=probe(
        "http://127.0.0.1:7070",
        concurrency_levels=[1],rounds=1,warmup=0,page_size=5,
        random_seed=7,request_timeout=1.0,server_pid=None,
        resource_interval=0.01,
    )

    assert result["passed"] is False
    assert result["levels"][0]["scorer_identity_stable"] is False


def test_probe_rejects_scorer_drift_during_warmup(monkeypatch):
    request,_requested_users=_fake_live_server(drift_after_feed=True)
    monkeypatch.setattr(probe_module,"_json_request",request)

    with pytest.raises(probe_module.ProbeError,match="changed during warm-up"):
        probe(
            "http://127.0.0.1:7070",
            concurrency_levels=[1],rounds=1,warmup=1,page_size=5,
            random_seed=7,request_timeout=1.0,server_pid=None,
            resource_interval=0.01,
        )


@pytest.mark.parametrize("levels",[[],[2],[1,1],[0,1]])
def test_probe_rejects_invalid_concurrency_contract(levels):
    with pytest.raises(ValueError,match="concurrency levels"):
        probe(
            "http://127.0.0.1:7070",
            concurrency_levels=levels,rounds=1,warmup=0,page_size=5,
            random_seed=7,request_timeout=1.0,server_pid=None,
            resource_interval=0.01,
        )
