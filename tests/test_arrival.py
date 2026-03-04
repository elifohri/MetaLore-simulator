"""Tests for arrival patterns (NoDeparture and PoissonArrival)."""

import numpy as np
import pytest

from metalore.core.arrival import NoDeparture, PoissonArrival


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SHARED_KWARGS = {
    "ep_max_time": 100,
    "seed": 42,
    "reset_rng_episode": True,
}


class MockEntity:
    """Minimal stand-in for UE / Sensor."""
    def __init__(self):
        self.stime = None
        self.extime = None


def make_entities(n):
    """Create a dict of mock entities mimicking the environment's entity dicts."""
    return {i: MockEntity() for i in range(n)}


# ===========================================================================
# NoDeparture tests
# ===========================================================================

class TestNoDeparture:

    def setup_method(self):
        self.pattern = NoDeparture(**SHARED_KWARGS)
        self.pattern.reset()

    def test_arrival_always_zero(self):
        entities = make_entities(10)
        self.pattern.arrival(entities)
        for e in entities.values():
            assert e.stime == 0

    def test_departure_always_ep_time(self):
        entities = make_entities(10)
        self.pattern.departure(entities)
        for e in entities.values():
            assert e.extime == SHARED_KWARGS["ep_max_time"]

    def test_all_entities_active_entire_episode(self):
        """Every entity satisfies stime <= t < extime for all t in [0, ep_time)."""
        ep_time = SHARED_KWARGS["ep_max_time"]
        entities = make_entities(5)
        self.pattern.arrival(entities)
        self.pattern.departure(entities)
        for e in entities.values():
            assert e.stime == 0
            assert e.extime == ep_time
            for t in range(ep_time):
                assert e.stime <= t < e.extime

    def test_reset_is_idempotent(self):
        """Calling reset multiple times gives the same results."""
        entities = make_entities(3)
        self.pattern.reset()
        self.pattern.arrival(entities)
        times1 = [e.stime for e in entities.values()]

        self.pattern.reset()
        self.pattern.arrival(entities)
        times2 = [e.stime for e in entities.values()]
        assert times1 == times2


# ===========================================================================
# PoissonArrival tests
# ===========================================================================

class TestPoissonArrival:

    def _make(self, num_entities=20, **overrides):
        params = {
            "arrival_rate": 0.5,
            "mean_duration": 15,
            "min_active_users": 2,
            **SHARED_KWARGS,
            **overrides,
        }
        pattern = PoissonArrival(**params)
        pattern.reset()
        entities = make_entities(num_entities)
        pattern.arrival(entities)
        pattern.departure(entities)
        return pattern, entities

    def _times(self, entities):
        return (
            [e.stime for e in entities.values()],
            [e.extime for e in entities.values()],
        )

    # --- basic validity ---------------------------------------------------

    def test_arrival_before_departure(self):
        _, entities = self._make()
        for e in entities.values():
            assert e.stime < e.extime, f"arrival {e.stime} >= departure {e.extime}"

    def test_arrival_within_episode(self):
        _, entities = self._make()
        ep = SHARED_KWARGS["ep_max_time"]
        for e in entities.values():
            assert 0 <= e.stime < ep
            assert 0 < e.extime <= ep

    def test_arrival_times_non_decreasing(self):
        _, entities = self._make()
        arrivals = [e.stime for e in entities.values()]
        for i in range(1, len(arrivals)):
            assert arrivals[i] >= arrivals[i - 1]

    # --- min_active_users constraint --------------------------------------------

    def test_min_active_users_satisfied(self):
        """At every timestep at least min_active_users entities are active."""
        _, entities = self._make(num_entities=30, min_active_users=3)
        arrivals, departures = self._times(entities)
        arr = np.array(arrivals)
        dep = np.array(departures)
        for t in range(SHARED_KWARGS["ep_max_time"]):
            active = int(np.sum((arr <= t) & (dep > t)))
            assert active >= 3, f"Only {active} active at t={t}"

    def test_min_active_users_one(self):
        _, entities = self._make(num_entities=15, min_active_users=1)
        arrivals, departures = self._times(entities)
        arr = np.array(arrivals)
        dep = np.array(departures)
        for t in range(SHARED_KWARGS["ep_max_time"]):
            active = int(np.sum((arr <= t) & (dep > t)))
            assert active >= 1, f"Zero active at t={t}"

    # --- reproducibility --------------------------------------------------

    def test_deterministic_with_same_seed(self):
        _, e1 = self._make(num_entities=10)
        _, e2 = self._make(num_entities=10)
        assert self._times(e1) == self._times(e2)

    def test_different_seed_gives_different_schedule(self):
        _, e1 = self._make(num_entities=10, seed=1)
        _, e2 = self._make(num_entities=10, seed=2)
        assert self._times(e1) != self._times(e2)

    def test_reset_reproduces_schedule(self):
        pattern, e1 = self._make(num_entities=10)
        times1 = self._times(e1)

        pattern.reset()
        e2 = make_entities(10)
        pattern.arrival(e2)
        pattern.departure(e2)
        assert times1 == self._times(e2)

    # --- edge cases -------------------------------------------------------

    def test_single_entity(self):
        _, entities = self._make(num_entities=1, min_active_users=1)
        e = list(entities.values())[0]
        assert e.stime == 0  # min_active_users forces arrival at 0
        assert e.extime > e.stime

    def test_short_episode(self):
        _, entities = self._make(num_entities=5, ep_max_time=5, mean_duration=2)
        for e in entities.values():
            assert 0 <= e.stime < 5
            assert e.stime < e.extime <= 5

    def test_high_arrival_rate(self):
        """With a very high arrival rate most entities arrive early."""
        _, entities = self._make(num_entities=20, arrival_rate=10.0)
        arrivals = [e.stime for e in entities.values()]
        mid = SHARED_KWARGS["ep_max_time"] // 2
        early = sum(1 for a in arrivals[:10] if a < mid)
        assert early >= 5, "High arrival rate should cluster arrivals early"

    def test_dynamic_traffic_varies(self):
        """Active entity count should change over time (not constant)."""
        _, entities = self._make(num_entities=20, min_active_users=1)
        arrivals, departures = self._times(entities)
        arr = np.array(arrivals)
        dep = np.array(departures)
        counts = [
            int(np.sum((arr <= t) & (dep > t)))
            for t in range(SHARED_KWARGS["ep_max_time"])
        ]
        assert max(counts) > min(counts), "Active count should vary over time"
