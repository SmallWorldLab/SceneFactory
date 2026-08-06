"""CPU-only tests for `compute_multilane_capacity` (pure-numpy, no Isaac/GPU).

The per-map capacity is the geometric holding room upstream of the workzone box:

    capacity_map = Sum over involved lanes L of floor(upstream_length(L) / spacing)

where the involved lanes are the merge contract's `closed_lane_id` + `open_lane_ids`,
`upstream_length` is the in-crop arc length from the box entry BACKWARD (upstream),
and the box entry is `center_s_m - half_len` (closed) or the projected box centre
(open). Covers: positive-int result with a per-lane breakdown, ~linear scaling with
the runway (double the upstream length -> ~2x capacity), crop clamping, and a graceful
0 on a missing contract.
"""

from __future__ import annotations

import math

import numpy as np

from src.trfc.lane_center_sampler import compute_multilane_capacity

_CLOSED_ID = 10
_OPEN_ID = 11
_LANE_LEN_M = 400.0
_LANE_SEP_M = 3.7
_CLOSED_Y = 0.0
_OPEN_Y = -_LANE_SEP_M
_SPACING_M = 8.0


def _straight_lane(lane_id: int, y: float, length_m: float = _LANE_LEN_M) -> dict:
    n = int(length_m) + 1
    xs = np.linspace(0.0, length_m, n, dtype=np.float32)
    xyz = np.stack(
        [xs, np.full(n, y, dtype=np.float32), np.zeros(n, dtype=np.float32)], axis=1
    )
    return {"id": lane_id, "type": 2, "n": n, "xyz": xyz.tolist()}


def _scene(box_s: float, half_len: float = 0.0, length_m: float = _LANE_LEN_M) -> dict:
    box = {
        "closed_lane_id": _CLOSED_ID,
        "center_s_m": float(box_s),
        "half_len": float(half_len),
        "half_wid": 1.85,
        "yaw_rad": 0.0,
        "open_lane_ids": [_OPEN_ID],
    }
    return {
        "meta": {},
        "road": {
            "polylines": [
                _straight_lane(_CLOSED_ID, _CLOSED_Y, length_m),
                _straight_lane(_OPEN_ID, _OPEN_Y, length_m),
            ]
        },
        "zones": {"keepout_boxes": [box]},
        "agents": {"items": []},
    }


# --- basic shape ------------------------------------------------------------

def test_returns_positive_int_with_per_lane_breakdown():
    # Huge crop -> nothing clips; upstream length == box entry arc-length.
    r = compute_multilane_capacity(_scene(box_s=160.0), spawn_spacing_m=_SPACING_M,
                                   bounds_size_m=100000.0)
    assert isinstance(r["capacity"], int) and r["capacity"] > 0
    assert set(r["per_lane"]) == {_CLOSED_ID, _OPEN_ID}
    assert all(isinstance(v, int) and v >= 0 for v in r["per_lane"].values())
    assert r["capacity"] == sum(r["per_lane"].values())
    assert r["spawn_spacing_m"] == _SPACING_M


def test_capacity_matches_closed_form():
    # Straight lanes, crop covers the whole road: closed entry = box_s - half_len,
    # open entry = projected box centre = box_s (lanes are aligned).
    box_s, half_len = 120.0, 10.0
    r = compute_multilane_capacity(_scene(box_s=box_s, half_len=half_len),
                                   spawn_spacing_m=_SPACING_M, bounds_size_m=100000.0)
    exp_closed = math.floor((box_s - half_len) / _SPACING_M)
    exp_open = math.floor(box_s / _SPACING_M)
    assert r["per_lane"][_CLOSED_ID] == exp_closed
    assert r["per_lane"][_OPEN_ID] == exp_open


# --- ~linear scaling with the runway ---------------------------------------

def test_capacity_scales_linearly_with_upstream_length():
    # Double the upstream runway (box_s) -> ~2x capacity (half_len=0 so both lanes
    # scale together). Exact here: 80/8=10 per lane, 160/8=20 per lane.
    small = compute_multilane_capacity(_scene(box_s=80.0), spawn_spacing_m=_SPACING_M,
                                       bounds_size_m=100000.0)
    big = compute_multilane_capacity(_scene(box_s=160.0), spawn_spacing_m=_SPACING_M,
                                     bounds_size_m=100000.0)
    assert small["capacity"] == 20
    assert big["capacity"] == 40
    assert big["capacity"] == 2 * small["capacity"]


# --- crop clamping ----------------------------------------------------------

def test_crop_clamps_upstream_length():
    # A tight crop truncates the usable upstream run, so capacity drops vs an
    # unbounded crop with the same geometry.
    scene = _scene(box_s=160.0)
    wide = compute_multilane_capacity(scene, spawn_spacing_m=_SPACING_M, bounds_size_m=100000.0)
    tight = compute_multilane_capacity(scene, spawn_spacing_m=_SPACING_M, bounds_size_m=60.0)
    assert tight["capacity"] < wide["capacity"]
    assert tight["capacity"] >= 0


# --- graceful failure -------------------------------------------------------

def test_missing_contract_returns_zero():
    scene = {"road": {"polylines": []}, "zones": {"keepout_boxes": []}}
    r = compute_multilane_capacity(scene, spawn_spacing_m=_SPACING_M)
    assert r == {"capacity": 0, "per_lane": {}, "spawn_spacing_m": _SPACING_M}

    # Box present but no merge contract keys -> also 0, no raise.
    scene2 = {
        "road": {"polylines": [_straight_lane(_CLOSED_ID, _CLOSED_Y)]},
        "zones": {"keepout_boxes": [{"host_polyline_id": _CLOSED_ID}]},
    }
    assert compute_multilane_capacity(scene2)["capacity"] == 0


# --- env per-scene capacity resolution + N formula --------------------------
# Mirrors StudentVehicleMultiAgentGoalEnv._resolve_multilane_capacity_for_env
# (baked box `capacity` wins, else geometric fallback, else config default) and
# the multilane N formula N = min(num_agents, max(1, ceil(capacity_e*density_e))).
# The env module pulls in Isaac Lab, so the logic is reproduced here for a CPU test.

def _resolve_capacity(scene: dict, *, spacing: float, bounds: float, fallback: int) -> int:
    boxes = (scene.get("zones", {}) or {}).get("keepout_boxes", []) or []
    cap = None
    if boxes:
        raw = boxes[0].get("capacity", None)
        if raw is not None:
            try:
                cap = int(raw)
            except (TypeError, ValueError):
                cap = None
    if cap is None or cap <= 0:
        computed = int(compute_multilane_capacity(
            scene, spawn_spacing_m=spacing, bounds_size_m=bounds)["capacity"])
        cap = computed if computed > 0 else int(fallback)
    return max(1, int(cap))


def _n_for(capacity: int, density: float, num_agents: int) -> int:
    return min(num_agents, max(1, math.ceil(capacity * density)))


def test_env_prefers_baked_capacity_over_geometry():
    # Bake a capacity that DIFFERS from the geometry so we prove the baked value
    # wins (geometry here is 2*floor(160/8)=40; baked C=27).
    scene = _scene(box_s=160.0)
    scene["zones"]["keepout_boxes"][0]["capacity"] = 27
    assert _resolve_capacity(scene, spacing=_SPACING_M, bounds=200.0, fallback=20) == 27


def test_env_falls_back_to_geometry_when_no_baked_value():
    scene = _scene(box_s=160.0)  # no baked capacity
    cap = _resolve_capacity(scene, spacing=_SPACING_M, bounds=100000.0, fallback=20)
    assert cap == compute_multilane_capacity(
        scene, spawn_spacing_m=_SPACING_M, bounds_size_m=100000.0)["capacity"] == 40


def test_env_n_formula_scales_with_baked_capacity():
    C = 30
    num_agents = 40  # articulation budget
    assert _n_for(C, 0.5, num_agents) == math.ceil(0.5 * C) == 15
    assert _n_for(C, 1.0, num_agents) == C            # density 1.0 -> full capacity
    assert _n_for(C, 0.0, num_agents) == 1            # max(1, ...) floor


def test_env_n_clamped_to_articulation_budget():
    # capacity above the articulation budget clamps: N never exceeds num_agents.
    assert _n_for(50, 1.0, num_agents=40) == 40
    assert _n_for(50, 0.5, num_agents=40) == 25
