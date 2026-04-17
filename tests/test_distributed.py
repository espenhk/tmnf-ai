"""
Tests for distributed grid search — coordinator re-queue logic and JSON round-trip.

These tests do NOT require a live TMInterface session; they use synthetic data.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from distributed.protocol import (
    ComboSpec,
    ResultPayload,
    combo_to_dict,
    combo_from_dict,
    result_to_dict,
    result_from_dict,
    experiment_to_json,
    experiment_from_dict,
)
from distributed.coordinator import Coordinator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_combo(name: str = "test_combo") -> ComboSpec:
    return ComboSpec(
        name=name,
        track="a03_centerline",
        training_params={"speed": 10.0, "n_sims": 5, "mutation_scale": 0.05,
                         "in_game_episode_s": 30.0},
        reward_params={"progress_weight": 10000.0},
    )


def _make_experiment_data_dict(name: str = "test_combo") -> dict:
    """Return a minimal ExperimentData-compatible dict for round-trip tests."""
    from framework.analytics import (
        ExperimentData, GreedySimResult, RunTrace,
    )
    data = ExperimentData(
        experiment_name=name,
        probe_results=[],
        cold_start_restarts=[],
        greedy_sims=[
            GreedySimResult(
                sim=1, reward=123.4, improved=True,
                throttle_counts=[5, 10, 85], total_steps=100,
                trace=RunTrace(pos_x=[0.0, 1.0], pos_z=[0.0, 2.0],
                               throttle_state=[(0, 1)], total_reward=123.4),
            ),
            GreedySimResult(
                sim=2, reward=100.0, improved=False,
                throttle_counts=[10, 10, 80], total_steps=100,
            ),
        ],
        probe_floor=None,
        weights_file="experiments/a03_centerline/test_combo/policy_weights.yaml",
        reward_config_file="experiments/a03_centerline/test_combo/reward_config.yaml",
        training_params={"speed": 10.0, "n_sims": 5},
        timings={"greedy_s": 42.0},
        track="a03_centerline",
    )
    return data


# ---------------------------------------------------------------------------
# Protocol serialisation
# ---------------------------------------------------------------------------

class TestComboSpecSerialization:
    def test_round_trip(self):
        spec = _make_combo()
        d = combo_to_dict(spec)
        recovered = combo_from_dict(d)
        assert recovered.name == spec.name
        assert recovered.track == spec.track
        assert recovered.training_params == spec.training_params
        assert recovered.reward_params == spec.reward_params

    def test_dict_is_json_serialisable(self):
        spec = _make_combo()
        json.dumps(combo_to_dict(spec))  # must not raise


class TestResultPayloadSerialization:
    def test_round_trip(self):
        payload = ResultPayload(name="x", data_json='{"experiment_name": "x"}')
        d = result_to_dict(payload)
        recovered = result_from_dict(d)
        assert recovered.name == payload.name
        assert recovered.data_json == payload.data_json


class TestExperimentDataSerialization:
    def test_to_json_produces_valid_json(self):
        data = _make_experiment_data_dict()
        json_str = experiment_to_json(data)
        parsed = json.loads(json_str)
        assert parsed["experiment_name"] == "test_combo"

    def test_round_trip_greedy_sims(self):
        original = _make_experiment_data_dict()
        json_str = experiment_to_json(original)
        recovered = experiment_from_dict(json.loads(json_str))

        assert len(recovered.greedy_sims) == 2
        assert recovered.greedy_sims[0].reward == pytest.approx(123.4)
        assert recovered.greedy_sims[0].improved is True
        assert recovered.greedy_sims[1].improved is False

    def test_round_trip_preserves_throttle_counts(self):
        original = _make_experiment_data_dict()
        recovered = experiment_from_dict(json.loads(experiment_to_json(original)))
        assert recovered.greedy_sims[0].throttle_counts == [5, 10, 85]

    def test_round_trip_preserves_trace(self):
        original = _make_experiment_data_dict()
        recovered = experiment_from_dict(json.loads(experiment_to_json(original)))
        trace = recovered.greedy_sims[0].trace
        assert trace is not None
        assert trace.pos_x == [0.0, 1.0]
        assert trace.total_reward == pytest.approx(123.4)

    def test_round_trip_none_trace(self):
        original = _make_experiment_data_dict()
        recovered = experiment_from_dict(json.loads(experiment_to_json(original)))
        assert recovered.greedy_sims[1].trace is None

    def test_round_trip_metadata(self):
        original = _make_experiment_data_dict()
        recovered = experiment_from_dict(json.loads(experiment_to_json(original)))
        assert recovered.training_params == {"speed": 10.0, "n_sims": 5}
        assert recovered.timings["greedy_s"] == pytest.approx(42.0)
        assert recovered.track == "a03_centerline"

    def test_numpy_arrays_serialised(self):
        """Numpy arrays in RunTrace should be serialised to lists without error."""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")

        from framework.analytics import ExperimentData, GreedySimResult, RunTrace
        data = ExperimentData(
            experiment_name="np_test",
            probe_results=[], cold_start_restarts=[],
            greedy_sims=[GreedySimResult(
                sim=1, reward=1.0, improved=True,
                throttle_counts=[0, 0, 100], total_steps=100,
                trace=RunTrace(
                    pos_x=np.array([1.0, 2.0, 3.0]),
                    pos_z=np.array([4.0, 5.0, 6.0]),
                    throttle_state=[],
                    total_reward=1.0,
                ),
            )],
            probe_floor=None, weights_file="", reward_config_file="",
            training_params={}, timings={},
        )
        json_str = experiment_to_json(data)
        recovered = experiment_from_dict(json.loads(json_str))
        assert recovered.greedy_sims[0].trace.pos_x == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Coordinator — work queue and re-queue logic
# ---------------------------------------------------------------------------

class TestCoordinator:
    def _start_coord(self, combos, port=15900, hb_timeout=2.0):
        coord = Coordinator(combos, port=port, heartbeat_timeout=hb_timeout)
        coord.start()
        return coord

    def test_work_queue_serves_all_combos(self):
        import urllib.request

        combos = [_make_combo(f"c{i}") for i in range(3)]
        coord = self._start_coord(combos, port=15901, hb_timeout=30.0)
        time.sleep(0.1)

        names_received = []
        for _ in range(3):
            with urllib.request.urlopen("http://localhost:15901/work", timeout=5) as r:
                assert r.status == 200
                spec = combo_from_dict(json.loads(r.read()))
                names_received.append(spec.name)

        # Fourth request should be 204 (queue empty, no re-queue yet)
        with urllib.request.urlopen("http://localhost:15901/work", timeout=5) as r:
            assert r.status == 204

        coord.stop()
        assert set(names_received) == {"c0", "c1", "c2"}

    def test_status_endpoint(self):
        import urllib.request

        combos = [_make_combo("s1"), _make_combo("s2")]
        coord = self._start_coord(combos, port=15902)
        time.sleep(0.1)

        with urllib.request.urlopen("http://localhost:15902/status", timeout=5) as r:
            status = json.loads(r.read())

        assert status["total"] == 2
        assert status["queued"] == 2
        assert status["done"] == 0
        coord.stop()

    def test_result_accepted_and_done_event_fires(self):
        import urllib.request

        combo = _make_combo("done_test")
        coord = self._start_coord([combo], port=15903)
        time.sleep(0.1)

        # Fetch work
        with urllib.request.urlopen("http://localhost:15903/work", timeout=5) as r:
            assert r.status == 200

        # Post result
        data = _make_experiment_data_dict("done_test")
        payload = ResultPayload(name="done_test", data_json=experiment_to_json(data))
        body = json.dumps(result_to_dict(payload)).encode()
        req = urllib.request.Request(
            "http://localhost:15903/result", data=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 200

        # wait_for_all should return immediately
        assert coord._done_event.is_set()
        runs = coord.wait_for_all()
        assert len(runs) == 1
        assert runs[0][0] == "done_test"
        coord.stop()

    def test_stale_worker_item_requeued(self):
        """An item not heartbeated within heartbeat_timeout should be re-queued."""
        import urllib.request

        combo = _make_combo("requeue_test")
        # hb_timeout=0.5 → monitor fires every ~0.25s; item stale after 0.5s
        coord = self._start_coord([combo], port=15904, hb_timeout=0.5)
        time.sleep(0.1)

        # Fetch work (marks as in-progress) but send NO heartbeats
        with urllib.request.urlopen("http://localhost:15904/work", timeout=5) as r:
            assert r.status == 200

        # Wait long enough for the monitor to re-queue the stale item
        time.sleep(1.5)

        # Item should now be back in the queue
        with urllib.request.urlopen("http://localhost:15904/work", timeout=5) as r:
            assert r.status == 200
            spec = combo_from_dict(json.loads(r.read()))
            assert spec.name == "requeue_test"

        coord.stop()

    def test_heartbeat_prevents_requeue(self):
        """A worker sending heartbeats should NOT have its item re-queued."""
        import urllib.request

        combo = _make_combo("hb_test")
        coord = self._start_coord([combo], port=15905, hb_timeout=1.0)
        time.sleep(0.1)

        with urllib.request.urlopen("http://localhost:15905/work", timeout=5) as r:
            assert r.status == 200

        # Send heartbeats every 0.3 s for 1.5 s — should keep item alive
        def send_hbs():
            for _ in range(5):
                body = json.dumps({"name": "hb_test", "worker_id": "tester"}).encode()
                req = urllib.request.Request(
                    "http://localhost:15905/heartbeat", data=body,
                    headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
                    method="POST",
                )
                try:
                    urllib.request.urlopen(req, timeout=5)
                except Exception:
                    pass
                time.sleep(0.3)

        hb_thread = threading.Thread(target=send_hbs, daemon=True)
        hb_thread.start()
        hb_thread.join()

        # Queue should still be empty (item is still in-progress, not re-queued)
        with urllib.request.urlopen("http://localhost:15905/work", timeout=5) as r:
            assert r.status == 204

        coord.stop()

    def test_empty_queue_returns_immediately(self):
        coord = Coordinator([], port=15906)
        coord.start()
        time.sleep(0.05)
        runs = coord.wait_for_all()
        assert runs == []
        coord.stop()
