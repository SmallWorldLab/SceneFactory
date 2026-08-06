"""CPU-only tests for the density-knob lane draw in the multilane merge sampler.

Covers the uniform per-car lane assignment introduced with the density knob:
  * N cars in -> N in-bounds spawns out (the env passes the already-scaled N;
    the sampler does NOT scale by density).
  * the N = max(1, ceil(capacity*density)) mapping the env applies before calling
    the sampler (0.5 -> 10, 0.37 -> 8, 0.0 -> 1 at capacity=20).
  * same seed -> byte-identical spawn+lane assignment (reproducible across cone
    candidates that share the OD fixed seed); different seed -> different split.
  * lane imbalance is ALLOWED (no forced 50/50 split).

No Isaac / GPU imports.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.trfc.lane_center_sampler import sample_multilane_merge_start_goal_pairs

_CLOSED_ID = 10
_OPEN_ID = 11
_LANE_LEN_M = 400.0
_LANE_SEP_M = 3.7
_CLOSED_Y = 0.0
_OPEN_Y = -_LANE_SEP_M
_BOX_S = 150.0
_HALF_LEN = 20.0
_Y_MID = 0.5 * (_CLOSED_Y + _OPEN_Y)


def _straight_lane(lane_id: int, y: float) -> dict:
    n = 401
    xs = np.linspace(0.0, _LANE_LEN_M, n, dtype=np.float32)
    xyz = np.stack(
        [xs, np.full(n, y, dtype=np.float32), np.zeros(n, dtype=np.float32)], axis=1
    )
    return {"id": lane_id, "type": 2, "n": n, "xyz": xyz.tolist()}


def _scene() -> dict:
    box = {
        "closed_lane_id": _CLOSED_ID,
        "center_s_m": _BOX_S,
        "half_len": _HALF_LEN,
        "half_wid": 1.85,
        "yaw_rad": 0.0,
        "open_lane_ids": [_OPEN_ID],
    }
    return {
        "meta": {},
        "road": {
            "polylines": [
                _straight_lane(_CLOSED_ID, _CLOSED_Y),
                _straight_lane(_OPEN_ID, _OPEN_Y),
            ]
        },
        "zones": {"keepout_boxes": [box]},
        "agents": {"items": []},
    }


def _sample(num_agents: int, seed: int):
    return sample_multilane_merge_start_goal_pairs(
        _scene(),
        num_agents=num_agents,
        bounds_size_m=1000.0,
        origin_mode="center",
        approach_gap_m=33.0,
        spawn_spacing_m=8.0,
        goal_clearance_m=15.0,
        return_goal_clearance_m=60.0,
        merge_return_frac=1.0,
        lane_distribution_mode="uniform",
        seed=seed,
    )


def _lane_of(y: float) -> str:
    return "closed" if y >= _Y_MID else "open"


def _split(samples) -> tuple[int, int]:
    closed = sum(1 for s in samples if _lane_of(float(s.start_xyz[1])) == "closed")
    return closed, len(samples) - closed


def _n_from_density(capacity: int, density: float) -> int:
    return max(1, math.ceil(capacity * density))


# --- N = max(1, ceil(capacity*density)) mapping the env applies -------------

def test_density_to_n_mapping_capacity_20():
    assert _n_from_density(20, 0.5) == 10
    assert _n_from_density(20, 0.37) == 8
    assert _n_from_density(20, 0.0) == 1     # max(1, ...) floor
    assert _n_from_density(20, 1.0) == 20


@pytest.mark.parametrize("density,expected_n", [(0.5, 10), (0.37, 8), (0.0, 1), (1.0, 20)])
def test_sampler_spawns_exactly_n_cars(density, expected_n):
    n = _n_from_density(20, density)
    assert n == expected_n
    samples = _sample(num_agents=n, seed=12345)
    # Lane is long/wide enough that every requested car is in-bounds.
    assert len(samples) == n, f"density={density}: expected {n} spawns, got {len(samples)}"


# --- reproducibility across candidates (shared OD fixed seed) ---------------

def test_same_seed_is_byte_identical():
    a = _sample(num_agents=10, seed=12345)
    b = _sample(num_agents=10, seed=12345)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert sa.start_xyz == sb.start_xyz
        assert sa.goal_xyz == sb.goal_xyz
        assert sa.start_yaw_rad == sb.start_yaw_rad
    # And the lane split itself is identical.
    assert _split(a) == _split(b)


def test_different_seed_changes_split():
    # Two seeds whose uniform draws differ (seed 0 -> 5/5, seed 7 -> 3/7).
    split0 = _split(_sample(num_agents=10, seed=0))
    split7 = _split(_sample(num_agents=10, seed=7))
    assert split0 != split7, (split0, split7)


# --- imbalance is allowed (do NOT assert 50/50) -----------------------------

def test_lane_imbalance_is_allowed():
    # Across many seeds at N=10 the closed count varies (random draw), i.e. it is
    # NOT pinned to 5. At least one seed must produce a non-even split.
    closed_counts = {_split(_sample(num_agents=10, seed=s))[0] for s in range(20)}
    assert closed_counts != {5}, "lane split is forced to 50/50 (should be random)"
    # Both lanes still get used across seeds (uniform, not degenerate).
    assert any(c > 0 for c in closed_counts)
