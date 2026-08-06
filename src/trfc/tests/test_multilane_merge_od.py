"""CPU-only test for the multilane merge OD sampler.

Two straight parallel same-direction lanes (closed id 10 / open id 11) ~3.7 m
apart, with a merge contract on the closed lane. Verifies destinations land on
BOTH lanes: closed-lane cars start+goal on the closed lane (return), open-lane
cars run straight on the open lane. No Isaac / GPU imports.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.trfc.lane_center_sampler import sample_multilane_merge_start_goal_pairs

_CLOSED_ID = 10
_OPEN_ID = 11
_LANE_LEN_M = 300.0   # >= 250 m
_LANE_SEP_M = 3.7
_CLOSED_Y = 0.0
_OPEN_Y = -_LANE_SEP_M
_BOX_S = 150.0        # mid-lane
_HALF_LEN = 20.0
_TAPER_LEN_M = 30.0
_APPROACH_GAP_M = 33.0
_GOAL_CLEARANCE_M = 15.0
_RETURN_GOAL_CLEARANCE_M = 60.0  # closed-lane returners' downstream clearance

# Classification midline between the two lanes (in Y).
_Y_MID = 0.5 * (_CLOSED_Y + _OPEN_Y)


def _straight_lane(lane_id: int, y: float) -> dict:
    n = 301
    xs = np.linspace(0.0, _LANE_LEN_M, n, dtype=np.float32)
    xyz = np.stack(
        [xs, np.full(n, y, dtype=np.float32), np.zeros(n, dtype=np.float32)], axis=1
    )
    # Lanes run along +X from x=0, so a point's x coordinate is its arc-length.
    return {"id": lane_id, "type": 2, "n": n, "xyz": xyz.tolist()}


def _scene(with_open: bool = True) -> dict:
    box = {
        "closed_lane_id": _CLOSED_ID,
        "center_s_m": _BOX_S,
        "half_len": _HALF_LEN,
        "half_wid": 1.85,
        "yaw_rad": 0.0,
        "parallel_s_window_m": 60.0,
    }
    if with_open:
        box["open_lane_ids"] = [_OPEN_ID]
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


def _lane_of(y: float) -> str:
    return "closed" if y >= _Y_MID else "open"


def _sample(num_agents=8, merge_return_frac=1.0):
    return sample_multilane_merge_start_goal_pairs(
        _scene(),
        num_agents=num_agents,
        bounds_size_m=1000.0,
        origin_mode="center",
        approach_gap_m=_APPROACH_GAP_M,
        goal_clearance_m=_GOAL_CLEARANCE_M,
        merge_return_frac=merge_return_frac,
        seed=0,
    )


def test_destinations_land_on_both_lanes_and_closed_cars_return():
    samples = _sample(num_agents=8, merge_return_frac=1.0)
    assert samples, "sampler returned no spawns"

    closed_cars = []  # start & goal on closed lane (returners)
    open_cars = []    # start & goal on open lane (straight-through)
    for s in samples:
        s_lane = _lane_of(float(s.start_xyz[1]))
        g_lane = _lane_of(float(s.goal_xyz[1]))
        if s_lane == "closed" and g_lane == "closed":
            closed_cars.append(s)
        elif s_lane == "open" and g_lane == "open":
            open_cars.append(s)

    # ~half on each lane; destinations land on BOTH lanes.
    assert len(closed_cars) >= 3, f"expected ~4 closed-lane returners, got {len(closed_cars)}"
    assert len(open_cars) >= 3, f"expected ~4 open-lane cars, got {len(open_cars)}"
    assert len(closed_cars) + len(open_cars) == len(samples)

    entry_taper_limit = _BOX_S - _HALF_LEN - _TAPER_LEN_M  # 150-20-30 = 100
    body_exit_edge = _BOX_S + _HALF_LEN                     # 170

    # Closed-lane returners clear the box/taper and stay on the closed lane.
    for s in closed_cars:
        start_s = float(s.start_xyz[0])
        goal_s = float(s.goal_xyz[0])
        assert start_s <= entry_taper_limit + 1e-3, (
            f"closed start s={start_s:.2f} not upstream of taper limit {entry_taper_limit:.2f}"
        )
        assert goal_s >= body_exit_edge - 1e-3, (
            f"closed goal s={goal_s:.2f} not downstream of box edge {body_exit_edge:.2f}"
        )
        assert abs(float(s.start_xyz[1]) - _CLOSED_Y) < 1.0
        assert abs(float(s.goal_xyz[1]) - _CLOSED_Y) < 1.0

    # Open-lane cars run straight on the open lane.
    for s in open_cars:
        assert abs(float(s.start_xyz[1]) - _OPEN_Y) < 1.0
        assert abs(float(s.goal_xyz[1]) - _OPEN_Y) < 1.0


def test_return_goal_clears_exit_taper_and_open_clearance_unchanged():
    # Default return_goal_clearance_m=60 -> closed-lane returners' goals must
    # land beyond the box exit edge + 60 m (past a 40 m worst-case exit taper +
    # merge-back room), while open-lane cars keep the smaller goal_clearance_m.
    samples = _sample(num_agents=8, merge_return_frac=1.0)

    closed_goal_min = _BOX_S + _HALF_LEN + _RETURN_GOAL_CLEARANCE_M  # 150+20+60 = 230
    # s_open == box_s here (parallel lane, projection is exact), so the closest
    # open goal sits at box_s + half_len + goal_clearance_m.
    open_goal_expected_min = _BOX_S + _HALF_LEN + _GOAL_CLEARANCE_M  # 150+20+15 = 185

    closed_goals = []
    open_goals = []
    for s in samples:
        s_lane = _lane_of(float(s.start_xyz[1]))
        g_lane = _lane_of(float(s.goal_xyz[1]))
        if s_lane == "closed" and g_lane == "closed":
            closed_goals.append(float(s.goal_xyz[0]))
        elif s_lane == "open" and g_lane == "open":
            open_goals.append(float(s.goal_xyz[0]))

    assert closed_goals, "no closed-lane returners found"
    assert open_goals, "no open-lane cars found"

    for g in closed_goals:
        assert g >= closed_goal_min - 1e-3, (
            f"closed return goal s={g:.2f} inside exit-taper zone "
            f"(need >= {closed_goal_min:.2f})"
        )

    # Open-lane clearance unchanged: closest open goal is at goal_clearance_m,
    # and every open goal stays below the (larger) closed return-goal floor.
    assert abs(min(open_goals) - open_goal_expected_min) < 1e-2, (
        f"open goal clearance changed: closest open goal {min(open_goals):.2f} "
        f"!= expected {open_goal_expected_min:.2f}"
    )
    assert min(open_goals) < closed_goal_min, (
        "open goal clearance was widened to the return clearance"
    )


def _scene_custom(*, lane_len: float, box_s: float, half_len: float, with_open: bool = True) -> dict:
    """Fixture with a configurable (possibly short) lane length + box position."""
    n = 201
    xs = np.linspace(0.0, float(lane_len), n, dtype=np.float32)

    def lane(lane_id: int, y: float) -> dict:
        xyz = np.stack(
            [xs, np.full(n, y, dtype=np.float32), np.zeros(n, dtype=np.float32)], axis=1
        )
        return {"id": lane_id, "type": 2, "n": n, "xyz": xyz.tolist()}

    box = {
        "closed_lane_id": _CLOSED_ID,
        "center_s_m": float(box_s),
        "half_len": float(half_len),
        "half_wid": 1.85,
        "yaw_rad": 0.0,
    }
    if with_open:
        box["open_lane_ids"] = [_OPEN_ID]
    return {
        "meta": {},
        "road": {"polylines": [lane(_CLOSED_ID, _CLOSED_Y), lane(_OPEN_ID, _OPEN_Y)]},
        "zones": {"keepout_boxes": [box]},
        "agents": {"items": []},
    }


def test_one_shared_goal_per_lane_and_distinct_spawns():
    samples = _sample(num_agents=8, merge_return_frac=1.0)
    assert samples

    lanes_used = {_lane_of(float(s.start_xyz[1])) for s in samples}
    goals = {tuple(round(c, 3) for c in s.goal_xyz) for s in samples}
    starts = [tuple(round(c, 4) for c in s.start_xyz) for s in samples]

    # Exactly one shared goal per lane used (here: closed + open = 2).
    assert len(goals) == len(lanes_used), (
        f"expected {len(lanes_used)} unique goals (one per lane), got {len(goals)}"
    )
    # Every car on a given lane shares an IDENTICAL goal_xyz.
    for lane in lanes_used:
        lane_goals = {
            tuple(round(c, 3) for c in s.goal_xyz)
            for s in samples
            if _lane_of(float(s.start_xyz[1])) == lane
        }
        assert len(lane_goals) == 1, f"lane {lane} has {len(lane_goals)} goals, expected 1"
    # Spawns stay per-car unique (physical cars can't overlap).
    assert len(set(starts)) == len(starts), "spawn positions are not all distinct"


def test_short_closed_lane_clamps_goal_and_keeps_cars_spawned():
    # Box near the lane end: box_exit + return_clearance (60) overflows the lane,
    # so the shared closed goal must clamp to lane_len - margin and the closed
    # cars must still spawn (they were being dropped before the shared goal).
    lane_len = 120.0
    box_s = 90.0
    half_len = 20.0
    end_margin = 2.0  # == sampler's min_upstream_margin_m default
    scene = _scene_custom(lane_len=lane_len, box_s=box_s, half_len=half_len)

    samples = sample_multilane_merge_start_goal_pairs(
        scene,
        num_agents=8,
        bounds_size_m=1000.0,
        origin_mode="center",
        approach_gap_m=_APPROACH_GAP_M,
        goal_clearance_m=_GOAL_CLEARANCE_M,
        return_goal_clearance_m=_RETURN_GOAL_CLEARANCE_M,
        merge_return_frac=1.0,
        seed=0,
    )

    closed_cars = [
        s
        for s in samples
        if _lane_of(float(s.start_xyz[1])) == "closed"
        and _lane_of(float(s.goal_xyz[1])) == "closed"
    ]
    assert closed_cars, "short closed lane dropped all closed-lane cars"

    unclamped = box_s + half_len + _RETURN_GOAL_CLEARANCE_M  # 170 > lane
    clamp_cap = lane_len - end_margin                        # 118
    assert unclamped > clamp_cap  # sanity: the fixture really overflows
    for s in closed_cars:
        goal_s = float(s.goal_xyz[0])
        assert goal_s <= clamp_cap + 1e-3, (
            f"closed goal s={goal_s:.2f} not clamped to lane end {clamp_cap:.2f}"
        )
        assert abs(goal_s - clamp_cap) < 1e-2, (
            f"closed goal s={goal_s:.2f} expected clamped to {clamp_cap:.2f}"
        )


def test_missing_open_lane_ids_raises():
    scene = _scene(with_open=False)
    with pytest.raises(RuntimeError):
        sample_multilane_merge_start_goal_pairs(
            scene, num_agents=8, bounds_size_m=1000.0, origin_mode="center"
        )
