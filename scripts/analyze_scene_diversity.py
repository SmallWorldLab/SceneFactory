#!/usr/bin/env python3
"""Characterise the geometric diversity of a SceneFactory scene pool, offline.

Answers the question "what is actually in the scene pool, and how much of it
survives the 200 m crop?" without Isaac Sim, Isaac Lab, USD, or a GPU.  Only
numpy / matplotlib / pyyaml plus two pure-numpy helpers from this repo are used.

    PYTHONPATH=. python scripts/analyze_scene_diversity.py \
        --pool configs/scene_factory/generated/scene_factory_256scene_random_0414_train.yaml \
        --out artifacts/scene_diversity/train_256

    PYTHONPATH=. python scripts/analyze_scene_diversity.py \
        --scene-dir data/processed/waymo_scenes_json --out artifacts/scene_diversity/all

Writes ``summary.json``, ``per_scene.csv`` and four/five PNG figures into --out.

--------------------------------------------------------------------------
WHAT IS EXACT AND WHAT IS HEURISTIC
--------------------------------------------------------------------------
EXACT (recomputed from the scene JSONs, no modelling):
  * unique-scene and world counts, and the world -> scene map.  The map comes
    from ``src.trfc.world_pipeline.prepare_stage_world_specs``, i.e. the same
    function the env uses, so ``assignment_fill_mode: random_fill`` expansion
    is reproduced rather than guessed.
  * polyline counts and polyline lengths per semantic class, raw and cropped.
  * agent counts (``agents.count_valid`` and ``len(agents.items)``).
  * qualifying spawn counts and the per-filter rejection attribution.  The
    headline count is produced by importing the env's own
    ``extract_vehicle_spawns_from_json``; the attribution is a re-implementation
    of the same filter chain and is cross-checked against it for every scene
    (any mismatch is reported in summary.json under ``spawn_crosscheck``).

HEURISTIC (a rule the author wrote; defensible but not ground truth):
  * the topology class of each scene.  Defined exactly below.  Every scene also
    carries an ``topology_ambiguous`` flag and the raw features the rule reads,
    so any individual call can be spot-checked from per_scene.csv.

--------------------------------------------------------------------------
POLYLINE TYPE CODES
--------------------------------------------------------------------------
``road.polylines[].type`` is the RAW Waymo ``roadgraph_samples/type`` enum,
passed through verbatim by scripts/convert_waymo_tfrecord_to_json.py.  Per
CLAUDE.md (authoritative; an earlier version of that table was wrong and caused
a real bug):

    1  LaneCenter-Freeway         drivable
    2  LaneCenter-SurfaceStreet   drivable      <- 1 AND 2 are BOTH lane centers
    3  LaneCenter-BikeLane        not by a car
    6-13 RoadLine                 paint marking, not a boundary
    15 RoadEdgeBoundary           not drivable
    16 RoadEdgeMedian             not drivable
    17/18/19/20  StopSign / Crosswalk / SpeedBump / Driveway
There is no type 4.  Type 0 (UNSET) does occur in the data and is bucketed as
``other``; type 21 is the workzone-boundary code emitted by this repo's
synthetic generator and is also bucketed as ``other``.

--------------------------------------------------------------------------
THE CROP (what "survives geometric filtering" means)
--------------------------------------------------------------------------
Reproduces ``chocolate_waymo_builder._build_road_world_from_json``:

  1. ``scene_center`` = mean of every road polyline XYZ point when
     ``world.origin_center_mode == "mean"`` (via the repo's own
     ``compute_scene_center_from_road``), or the XYZ bbox centre when it is
     ``"bbox"``.  When ``world.origin_mode != "center"`` the centre is the
     origin.
  2. local xy = raw xy - scene_center xy.
  3. A segment (p_i, p_i+1) is KEPT iff both endpoints satisfy
     |x| <= bounds/2 and |y| <= bounds/2, AND its planar length lies in
     (1e-6, jump_break_m].  jump_break_m defaults to the builder default 3.0 m
     and is read from the pool YAML's ``road.jump_break_m`` when present.
  4. Maximal chains of consecutive kept segments form "runs".

Caveat carried over from CLAUDE.md: what the env sees is NOT the JSON.  A scene
whose JSON contains type-1 polylines can end up with none of them inside the
crop.  Both raw and cropped numbers are reported for exactly that reason.

--------------------------------------------------------------------------
TOPOLOGY HEURISTIC (precise definition)
--------------------------------------------------------------------------
Computed on the CROPPED lane-centre runs only (types 1 and 2; bike lanes and
all non-lane types are excluded).  Runs shorter than 5 m are dropped.

Features
  L               total kept lane-centre length (m).
  crossings       each run is resampled at 4 m.  A crossing is a pair of
                  resampled segments belonging to DIFFERENT source polyline ids
                  that intersect strictly in their interiors (both parameters in
                  (0.02, 0.98)).  Interior-only intersection excludes the
                  endpoint-to-endpoint joins by which Waymo chains successive
                  lane pieces, so a plain lane succession produces no crossing.
                  theta = undirected angle between the two segments, in [0,90].
                    conflict crossing : theta >= 30 deg
                    shallow  crossing : 5 deg <= theta < 30 deg
  conflict_junctions  conflict crossings grouped into junctions by greedy
                  single-pass clustering with a 30 m linkage radius, over points
                  sorted by (x, y) so the result is order-independent.  One
                  signalised intersection scatters conflict points over ~30 m
                  and must count once, not once per lane pair.
                  shallow_junctions is the same over shallow crossings.
  heading_modes   length-weighted histogram of undirected segment heading
                  (mod 180 deg) in 5 deg bins, circularly smoothed with
                  [1,2,3,2,1]/9.  Modes = local maxima >= 0.40 x global max,
                  taken greedily in descending order with >= 30 deg separation.
  heading_spread  length-weighted circular std of the doubled heading angle,
                  halved back to the mod-180 domain, in degrees.  0 deg = every
                  lane parallel; ~52 deg = uniformly random orientation.
  circular_runs   runs with |sum of signed turn| >= 200 deg, arc length <= 250 m,
                  mean radius about the run centroid in [4, 40] m and radial
                  coefficient of variation <= 0.35.
  parallel_lanes  number of distinct source polylines with >= 20 m kept length
                  whose length-weighted mean heading is within 15 deg (mod 180)
                  of the dominant heading mode.

Classification -- first rule that fires wins
  R0 sparse             L < 150 m
  R1 roundabout         circular_runs >= 1
  R2 urban_grid         conflict_junctions >= 4
  R3 multi_junction     conflict_junctions in {2, 3}
  R4 single_junction    conflict_junctions == 1
  R5 merge_fork         conflict_junctions == 0 and shallow_junctions >= 1
  R6 curved             no crossings at all and (heading_spread >= 20 deg or
                        heading_modes >= 2)
  R7 straight_multilane otherwise, parallel_lanes >= 2
  R8 straight_single    otherwise

The three junction classes differ only in how many junctions the crop contains;
they are one family, and "at least one junction" = urban_grid + multi_junction
+ single_junction is the number to quote if a single figure is wanted.

``topology_ambiguous`` is set when the decision rests on thin evidence:
  * L < 400 m; or
  * class in {urban_grid, multi_junction, single_junction} and total conflict
    crossings <= 3; or
  * class == merge_fork and total shallow crossings <= 2; or
  * class in {curved, straight_multilane, straight_single} and
    15 deg <= heading_spread < 25 deg (either side of the R5 threshold); or
  * class == roundabout and the only circular run has radial CV > 0.25.

Known limits of the rule, stated so nobody over-reads it: a junction class means
"the crop contains N at-grade conflict points between lane centres", which fires
on a minor side street or a parking aisle touching the crop edge just as it does
on a major signalised junction -- read ``conflict_junctions`` alongside the
label.  Grade-separated
crossings (an overpass) are indistinguishable from at-grade ones because
``flatten_road_z`` discards Z; those are counted as conflicts.  The rule is
geometric only: it never reads lane connectivity, traffic control, or agent
tracks.

--------------------------------------------------------------------------
SPAWN YIELD
--------------------------------------------------------------------------
The env caps usable agents per world at min(num_agents_per_env, len(spawns)).
This script reports the UNCAPPED qualifying spawn count (it calls the env's
extractor with an effectively infinite ``max_controllable``), because that is
the pool property; the cap is a training-config knob and is reported separately
as the fraction of scenes clearing 8 / 16 / 32 / 64.

Filter parameters come from the pool YAML: ``world.bounds_size_m``,
``world.origin_mode``, ``world.origin_center_mode``, and the ``vehicles`` block
(``require_goal_in_bounds``, ``skip_if_start_in_goal``, ``goal_radius_m``,
``start_goal_thresh_m``).  Pools that omit ``vehicles`` (the 256-scene train
pool and the 199-scene unseen pools do) fall back to the same defaults the env
uses, with ``goal_radius_m`` = ``--goal-radius-m`` (default 3.0, the value every
``env.goal_reached_threshold_m`` in configs/scene_factory carries).  The
resolved values are echoed into summary.json under ``spawn_filter_params``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

# --- offline-safe repo imports -------------------------------------------------
# Both modules are pure numpy at import time.  src.scene_factory_multiworld_scene
# calls ensure_isaaclab_source_paths() (a sys.path mutation only) and imports
# isaaclab lazily inside functions this script never calls, so importing it does
# NOT pull in isaaclab / omni / pxr.  Verified by checking sys.modules after
# import.  Do not add imports here without re-checking that.
from src.scene_factory_multiworld_scene import extract_vehicle_spawns_from_json
from src.trfc.lane_center_sampler import compute_scene_center_from_road
from src.trfc.world_pipeline import prepare_stage_world_specs

# ------------------------------------------------------------------ constants --
LANE_CENTER_TYPES = (1, 2)
SEMANTIC_CLASSES: dict[str, tuple[int, ...]] = {
    "lane_center": (1, 2),
    "bike_lane": (3,),
    "road_line": (6, 7, 8, 9, 10, 11, 12, 13),
    "road_edge": (15, 16),
    "feature": (17, 18, 19, 20),
}
_TYPE_TO_CLASS = {t: name for name, codes in SEMANTIC_CLASSES.items() for t in codes}
OTHER_CLASS = "other"
ALL_CLASSES = tuple(SEMANTIC_CLASSES) + (OTHER_CLASS,)

DEFAULT_BOUNDS_SIZE_M = 200.0
DEFAULT_JUMP_BREAK_M = 3.0
DEFAULT_GOAL_RADIUS_M = 3.0
UNCAPPED = 10**9

# topology thresholds (see module docstring)
MIN_RUN_LEN_M = 5.0
RESAMPLE_STEP_M = 4.0
CONFLICT_ANGLE_DEG = 30.0
SHALLOW_ANGLE_MIN_DEG = 5.0
JUNCTION_LINKAGE_M = 30.0
MODE_PEAK_FRAC = 0.40
MODE_MIN_SEP_DEG = 30.0
SPARSE_LANE_LEN_M = 150.0
THIN_LANE_LEN_M = 400.0
CURVED_SPREAD_DEG = 20.0
PARALLEL_TOL_DEG = 15.0
PARALLEL_MIN_LEN_M = 20.0
CIRC_TURN_DEG = 200.0
CIRC_MAX_LEN_M = 250.0
CIRC_RADIUS_RANGE_M = (4.0, 40.0)
CIRC_CV_MAX = 0.35

TOPOLOGY_ORDER = (
    "roundabout",
    "urban_grid",
    "multi_junction",
    "single_junction",
    "merge_fork",
    "curved",
    "straight_multilane",
    "straight_single",
    "sparse",
)
SPAWN_CAPS = (8, 16, 32, 64)

# dataviz reference palette, light mode (colorblind-validated, fixed order).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8b8a84"


# ------------------------------------------------------------------- dataclasses
@dataclass
class SpawnFilterParams:
    bounds_size_m: float
    origin_mode: str
    origin_center_mode: str
    require_goal_in_bounds: bool
    skip_if_start_in_goal: bool
    goal_radius_m: float
    start_goal_thresh_m: float | None
    jump_break_m: float


@dataclass
class SceneRecord:
    scene: str
    # -- exact: pool composition
    n_polylines_raw: int
    n_polylines_cropped: int
    poly_count_raw: dict[str, int] = field(default_factory=dict)
    poly_count_cropped: dict[str, int] = field(default_factory=dict)
    length_raw_m: dict[str, float] = field(default_factory=dict)
    length_cropped_m: dict[str, float] = field(default_factory=dict)
    bbox_raw_m: tuple[float, float] = (0.0, 0.0)
    bbox_cropped_m: tuple[float, float] = (0.0, 0.0)
    # -- exact: agents / spawns
    agents_count_valid: int = 0
    agents_n_items: int = 0
    spawns_qualifying: int = 0
    rej_missing_coords: int = 0
    rej_start_out_of_bounds: int = 0
    rej_goal_out_of_bounds: int = 0
    rej_start_in_goal: int = 0
    spawn_crosscheck_ok: bool = True
    # -- heuristic: topology
    topology: str = "sparse"
    topology_ambiguous: bool = False
    lane_len_cropped_m: float = 0.0
    n_lane_runs: int = 0
    n_lane_polylines: int = 0
    conflict_crossings: int = 0
    conflict_junctions: int = 0
    shallow_crossings: int = 0
    shallow_junctions: int = 0
    heading_modes: int = 0
    heading_spread_deg: float = 0.0
    circular_runs: int = 0
    parallel_lanes: int = 0


# ------------------------------------------------------------------- geometry --
def _polyline_xy(poly: dict[str, Any]) -> np.ndarray | None:
    xyz = poly.get("xyz")
    if not xyz:
        return None
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 2:
        return None
    return arr[:, :2]


def _scene_center_xy(scene: dict[str, Any], *, origin_mode: str, origin_center_mode: str) -> np.ndarray:
    """Mirror chocolate_waymo_builder / scene_factory_multiworld_scene centring."""
    if str(origin_mode).strip().lower() != "center":
        return np.zeros((2,), dtype=np.float64)
    if str(origin_center_mode).strip().lower() == "bbox":
        chunks = [xy for xy in (_polyline_xy(p) for p in scene.get("road", {}).get("polylines", []) or []) if xy is not None]
        if not chunks:
            return np.zeros((2,), dtype=np.float64)
        pts = np.concatenate(chunks, axis=0)
        return 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    return np.asarray(compute_scene_center_from_road(scene), dtype=np.float64)[:2]


def _kept_runs(xy: np.ndarray, *, half: float, jump_break_m: float) -> list[np.ndarray]:
    """Maximal chains of segments passing the builder's in-bounds + jump gate."""
    if xy.shape[0] < 2:
        return []
    in_bounds = (np.abs(xy) <= half).all(axis=1)
    seg_len = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    ok = in_bounds[:-1] & in_bounds[1:] & (seg_len > 1e-6) & (seg_len <= jump_break_m)
    runs: list[np.ndarray] = []
    idx = 0
    n = ok.shape[0]
    while idx < n:
        if not ok[idx]:
            idx += 1
            continue
        end = idx
        while end < n and ok[end]:
            end += 1
        runs.append(xy[idx : end + 1])
        idx = end
    return runs


def _polyline_length(xy: np.ndarray) -> float:
    if xy.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def _resample(xy: np.ndarray, step: float) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    if d[-1] <= step:
        return xy[[0, -1]]
    s = np.concatenate([np.arange(0.0, d[-1], step), [d[-1]]])
    return np.stack([np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])], axis=1)


def _crossings(runs: Sequence[tuple[int, np.ndarray]], *, chunk: int = 512) -> np.ndarray:
    """Interior segment-segment intersections between DIFFERENT polyline ids.

    Returns an (M, 3) array of [x, y, undirected_angle_deg].
    """
    starts, ends, owners = [], [], []
    for pid, xy in runs:
        r = _resample(xy, RESAMPLE_STEP_M)
        if r.shape[0] < 2:
            continue
        starts.append(r[:-1])
        ends.append(r[1:])
        owners.append(np.full(r.shape[0] - 1, pid, dtype=np.int64))
    if not starts:
        return np.zeros((0, 3), dtype=np.float64)
    p = np.concatenate(starts, axis=0)
    q = np.concatenate(ends, axis=0)
    own = np.concatenate(owners, axis=0)
    r_vec = q - p
    n = p.shape[0]
    out: list[np.ndarray] = []
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        a = p[lo:hi, None, :]
        rr = r_vec[lo:hi, None, :]
        c = p[None, :, :]
        ss = r_vec[None, :, :]
        den = rr[..., 0] * ss[..., 1] - rr[..., 1] * ss[..., 0]
        qp = c - a
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (qp[..., 0] * ss[..., 1] - qp[..., 1] * ss[..., 0]) / den
            u = (qp[..., 0] * rr[..., 1] - qp[..., 1] * rr[..., 0]) / den
        hit = (np.abs(den) > 1e-9) & (t > 0.02) & (t < 0.98) & (u > 0.02) & (u < 0.98)
        hit &= own[lo:hi, None] != own[None, :]
        # keep each unordered pair once
        rows = np.arange(lo, hi)[:, None]
        hit &= rows < np.arange(n)[None, :]
        if not hit.any():
            continue
        th1 = np.arctan2(rr[..., 1], rr[..., 0])
        th2 = np.arctan2(ss[..., 1], ss[..., 0])
        ang = np.degrees(np.abs((th1 - th2 + 0.5 * np.pi) % np.pi - 0.5 * np.pi))
        ii, _ = np.nonzero(hit)
        pts = a[ii, 0, :] + t[hit][:, None] * rr[ii, 0, :]
        out.append(np.concatenate([pts, ang[hit][:, None]], axis=1))
    if not out:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(out, axis=0)


def _junction_count(points: np.ndarray) -> int:
    """Group crossing points into junctions.

    Greedy single-pass clustering with a fixed linkage radius, over points
    sorted lexicographically by (x, y) so the result does not depend on the
    order polylines happen to appear in the JSON.  One wide signalised
    intersection produces conflict points spread over ~30 m, so a raw grid-cell
    count would report it as several junctions; the linkage radius collapses it
    to one.
    """
    if points.shape[0] == 0:
        return 0
    pts = points[np.lexsort((points[:, 1], points[:, 0])), :2]
    centres: list[np.ndarray] = []
    sizes: list[int] = []
    for p in pts:
        best, best_d = -1, np.inf
        for i, c in enumerate(centres):
            d = float(np.hypot(p[0] - c[0], p[1] - c[1]))
            if d < best_d:
                best, best_d = i, d
        if best >= 0 and best_d <= JUNCTION_LINKAGE_M:
            n = sizes[best]
            centres[best] = (centres[best] * n + p) / (n + 1)
            sizes[best] = n + 1
        else:
            centres.append(p.astype(np.float64).copy())
            sizes.append(1)
    return len(centres)


def _heading_stats(headings: np.ndarray, weights: np.ndarray) -> tuple[int, float, float]:
    """(n_modes, dominant_heading_deg mod 180, length-weighted circular std deg)."""
    if headings.size == 0 or weights.sum() <= 0:
        return 0, 0.0, 0.0
    deg = np.degrees(headings) % 180.0
    hist, _ = np.histogram(deg, bins=36, range=(0.0, 180.0), weights=weights)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    kernel /= kernel.sum()
    padded = np.concatenate([hist[-2:], hist, hist[:2]])
    smooth = np.convolve(padded, kernel, mode="valid")
    peak_floor = MODE_PEAK_FRAC * float(smooth.max())
    candidates = [
        i
        for i in range(36)
        if smooth[i] >= smooth[(i - 1) % 36] and smooth[i] >= smooth[(i + 1) % 36] and smooth[i] >= peak_floor
    ]
    candidates.sort(key=lambda i: -smooth[i])
    picked: list[int] = []
    for i in candidates:
        sep_ok = all(min(abs(i - j), 36 - abs(i - j)) * 5.0 >= MODE_MIN_SEP_DEG for j in picked)
        if sep_ok:
            picked.append(i)
    dominant = (picked[0] * 5.0 + 2.5) if picked else float(deg[np.argmax(weights)])
    # circular std on the doubled angle (undirected data has period 180 deg)
    two = np.radians(2.0 * deg)
    w = weights / weights.sum()
    resultant = math.hypot(float((w * np.cos(two)).sum()), float((w * np.sin(two)).sum()))
    resultant = min(max(resultant, 1e-9), 1.0)
    spread_deg = 0.5 * math.degrees(math.sqrt(-2.0 * math.log(resultant)))
    return len(picked), dominant, spread_deg


def _is_circular(xy: np.ndarray) -> tuple[bool, float]:
    """(is_circular, radial coefficient of variation)."""
    if xy.shape[0] < 6:
        return False, 0.0
    d = np.diff(xy, axis=0)
    length = float(np.linalg.norm(d, axis=1).sum())
    if length > CIRC_MAX_LEN_M:
        return False, 0.0
    th = np.arctan2(d[:, 1], d[:, 0])
    dth = (np.diff(th) + np.pi) % (2.0 * np.pi) - np.pi
    if abs(math.degrees(float(dth.sum()))) < CIRC_TURN_DEG:
        return False, 0.0
    centre = xy.mean(axis=0)
    radii = np.linalg.norm(xy - centre, axis=1)
    mean_r = float(radii.mean())
    if not (CIRC_RADIUS_RANGE_M[0] <= mean_r <= CIRC_RADIUS_RANGE_M[1]):
        return False, 0.0
    cv = float(radii.std() / max(mean_r, 1e-6))
    return cv <= CIRC_CV_MAX, cv


# ------------------------------------------------------------------- topology --
def classify_topology(lane_runs: Sequence[tuple[int, np.ndarray]]) -> dict[str, Any]:
    """Implements the heuristic documented in the module docstring."""
    runs = [(pid, xy) for pid, xy in lane_runs if _polyline_length(xy) >= MIN_RUN_LEN_M]
    lane_len = sum(_polyline_length(xy) for _, xy in runs)
    feats: dict[str, Any] = {
        "lane_len_cropped_m": lane_len,
        "n_lane_runs": len(runs),
        "n_lane_polylines": len({pid for pid, _ in runs}),
        "conflict_crossings": 0,
        "conflict_junctions": 0,
        "shallow_crossings": 0,
        "shallow_junctions": 0,
        "heading_modes": 0,
        "heading_spread_deg": 0.0,
        "circular_runs": 0,
        "parallel_lanes": 0,
        "topology": "sparse",
        "topology_ambiguous": True,
    }
    if lane_len < SPARSE_LANE_LEN_M or not runs:
        return feats

    # heading statistics over every kept lane-centre segment
    head_list, wt_list = [], []
    per_poly: dict[int, list[tuple[float, float]]] = {}
    for pid, xy in runs:
        d = np.diff(xy, axis=0)
        seg_len = np.linalg.norm(d, axis=1)
        th = np.arctan2(d[:, 1], d[:, 0])
        head_list.append(th)
        wt_list.append(seg_len)
        per_poly.setdefault(pid, []).append((float(seg_len.sum()), float(np.degrees(_circ_mean_180(th, seg_len)))))
    headings = np.concatenate(head_list)
    weights = np.concatenate(wt_list)
    n_modes, dominant_deg, spread_deg = _heading_stats(headings, weights)
    feats["heading_modes"] = int(n_modes)
    feats["heading_spread_deg"] = float(spread_deg)

    parallel = 0
    for pid, entries in per_poly.items():
        total = sum(e[0] for e in entries)
        if total < PARALLEL_MIN_LEN_M:
            continue
        mean_deg = max(entries, key=lambda e: e[0])[1] % 180.0
        delta = abs(mean_deg - dominant_deg) % 180.0
        delta = min(delta, 180.0 - delta)
        if delta <= PARALLEL_TOL_DEG:
            parallel += 1
    feats["parallel_lanes"] = int(parallel)

    circ_cvs = [cv for ok, cv in (_is_circular(xy) for _, xy in runs) if ok]
    feats["circular_runs"] = len(circ_cvs)

    xs = _crossings(runs)
    conflict = xs[xs[:, 2] >= CONFLICT_ANGLE_DEG] if xs.shape[0] else xs
    shallow = xs[(xs[:, 2] >= SHALLOW_ANGLE_MIN_DEG) & (xs[:, 2] < CONFLICT_ANGLE_DEG)] if xs.shape[0] else xs
    feats["conflict_crossings"] = int(conflict.shape[0])
    feats["conflict_junctions"] = _junction_count(conflict)
    feats["shallow_crossings"] = int(shallow.shape[0])
    feats["shallow_junctions"] = _junction_count(shallow)

    if feats["circular_runs"] >= 1:
        label = "roundabout"
    elif feats["conflict_junctions"] >= 4:
        label = "urban_grid"
    elif feats["conflict_junctions"] >= 2:
        label = "multi_junction"
    elif feats["conflict_junctions"] >= 1:
        label = "single_junction"
    elif feats["shallow_junctions"] >= 1:
        label = "merge_fork"
    elif spread_deg >= CURVED_SPREAD_DEG or n_modes >= 2:
        label = "curved"
    elif parallel >= 2:
        label = "straight_multilane"
    else:
        label = "straight_single"
    feats["topology"] = label

    ambiguous = lane_len < THIN_LANE_LEN_M
    if label in ("urban_grid", "multi_junction", "single_junction") and feats["conflict_crossings"] <= 3:
        ambiguous = True
    if label == "merge_fork" and feats["shallow_crossings"] <= 2:
        ambiguous = True
    if label in ("curved", "straight_multilane", "straight_single") and 15.0 <= spread_deg < 25.0:
        ambiguous = True
    if label == "roundabout" and len(circ_cvs) == 1 and circ_cvs[0] > 0.25:
        ambiguous = True
    feats["topology_ambiguous"] = bool(ambiguous)
    return feats


def _circ_mean_180(headings: np.ndarray, weights: np.ndarray) -> float:
    two = 2.0 * headings
    w = weights / max(float(weights.sum()), 1e-9)
    ang = math.atan2(float((w * np.sin(two)).sum()), float((w * np.cos(two)).sum()))
    return 0.5 * ang


# ---------------------------------------------------------------- spawn filter --
def _attribute_spawn_filters(scene: dict[str, Any], params: SpawnFilterParams) -> dict[str, int]:
    """Re-implementation of extract_vehicle_spawns_from_scene_cfg's filter chain.

    Only used to attribute rejections to a cause; the surviving count is
    cross-checked against the imported extractor for every scene.
    """
    centre = _scene_center_xy(scene, origin_mode=params.origin_mode, origin_center_mode=params.origin_center_mode)
    items = (scene.get("agents", {}) or {}).get("items", []) or []
    half = 0.5 * params.bounds_size_m
    thresh = params.start_goal_thresh_m if params.start_goal_thresh_m is not None else params.goal_radius_m
    out = {"missing_coords": 0, "start_out_of_bounds": 0, "goal_out_of_bounds": 0, "start_in_goal": 0, "surviving": 0}
    for agent in items:
        start = agent.get("start", {}) or {}
        end = agent.get("end", {}) or {}
        try:
            sx, sy = float(start["x"]), float(start["y"])
            gx, gy = float(end["x"]), float(end["y"])
        except (KeyError, TypeError, ValueError):
            out["missing_coords"] += 1
            continue
        sz = float(start.get("z") or 0.0)
        gz = float(end.get("z") or 0.0)
        slx, sly = sx - centre[0], sy - centre[1]
        glx, gly = gx - centre[0], gy - centre[1]
        if not (abs(slx) <= half and abs(sly) <= half):
            out["start_out_of_bounds"] += 1
            continue
        if params.require_goal_in_bounds and not (abs(glx) <= half and abs(gly) <= half):
            out["goal_out_of_bounds"] += 1
            continue
        dist = math.sqrt((glx - slx) ** 2 + (gly - sly) ** 2 + (gz - sz) ** 2)
        if params.skip_if_start_in_goal and dist <= thresh:
            out["start_in_goal"] += 1
            continue
        out["surviving"] += 1
    return out


# ------------------------------------------------------------------- analysis --
def analyse_scene(path: Path, params: SpawnFilterParams) -> SceneRecord:
    scene = json.loads(path.read_text())
    centre = _scene_center_xy(scene, origin_mode=params.origin_mode, origin_center_mode=params.origin_center_mode)
    half = 0.5 * params.bounds_size_m
    polylines = (scene.get("road", {}) or {}).get("polylines", []) or []

    count_raw = Counter()
    count_cropped = Counter()
    len_raw = Counter()
    len_cropped = Counter()
    raw_pts: list[np.ndarray] = []
    crop_pts: list[np.ndarray] = []
    lane_runs: list[tuple[int, np.ndarray]] = []
    n_cropped_polylines = 0

    for poly in polylines:
        xy_raw = _polyline_xy(poly)
        if xy_raw is None:
            continue
        code = int(poly.get("type", -1))
        cls = _TYPE_TO_CLASS.get(code, OTHER_CLASS)
        count_raw[cls] += 1
        len_raw[cls] += _polyline_length(xy_raw)
        raw_pts.append(xy_raw)

        xy = xy_raw - centre
        runs = _kept_runs(xy, half=half, jump_break_m=params.jump_break_m)
        if runs:
            n_cropped_polylines += 1
            count_cropped[cls] += 1
            len_cropped[cls] += sum(_polyline_length(r) for r in runs)
            crop_pts.extend(runs)
            if code in LANE_CENTER_TYPES:
                pid = int(poly.get("id", -1))
                lane_runs.extend((pid, r) for r in runs)

    def _extent(chunks: list[np.ndarray]) -> tuple[float, float]:
        if not chunks:
            return (0.0, 0.0)
        pts = np.concatenate(chunks, axis=0)
        span = pts.max(axis=0) - pts.min(axis=0)
        return (float(span[0]), float(span[1]))

    agents = scene.get("agents", {}) or {}
    attribution = _attribute_spawn_filters(scene, params)
    spawns = extract_vehicle_spawns_from_json(
        path,
        bounds_size_m=params.bounds_size_m,
        origin_mode=params.origin_mode,
        origin_center_mode=params.origin_center_mode,
        max_controllable=UNCAPPED,
        require_goal_in_bounds=params.require_goal_in_bounds,
        skip_if_start_in_goal=params.skip_if_start_in_goal,
        goal_radius_m=params.goal_radius_m,
        start_goal_thresh_m=params.start_goal_thresh_m,
    )
    topo = classify_topology(lane_runs)

    return SceneRecord(
        scene=path.name,
        n_polylines_raw=len(polylines),
        n_polylines_cropped=n_cropped_polylines,
        poly_count_raw={c: int(count_raw.get(c, 0)) for c in ALL_CLASSES},
        poly_count_cropped={c: int(count_cropped.get(c, 0)) for c in ALL_CLASSES},
        length_raw_m={c: float(len_raw.get(c, 0.0)) for c in ALL_CLASSES},
        length_cropped_m={c: float(len_cropped.get(c, 0.0)) for c in ALL_CLASSES},
        bbox_raw_m=_extent(raw_pts),
        bbox_cropped_m=_extent(crop_pts),
        agents_count_valid=int(agents.get("count_valid", 0) or 0),
        agents_n_items=len(agents.get("items", []) or []),
        spawns_qualifying=len(spawns),
        rej_missing_coords=attribution["missing_coords"],
        rej_start_out_of_bounds=attribution["start_out_of_bounds"],
        rej_goal_out_of_bounds=attribution["goal_out_of_bounds"],
        rej_start_in_goal=attribution["start_in_goal"],
        spawn_crosscheck_ok=bool(attribution["surviving"] == len(spawns)),
        **{k: v for k, v in topo.items()},
    )


# ----------------------------------------------------------------- pool config --
def resolve_pool(pool_path: Path, goal_radius_default: float) -> tuple[list[Any], SpawnFilterParams, dict[str, Any]]:
    cfg = yaml.safe_load(pool_path.read_text()) or {}
    world_cfg = dict(cfg.get("world", {}) or {})
    vehicles_cfg = dict(cfg.get("vehicles", {}) or {})
    road_cfg = dict(cfg.get("road", {}) or {})
    specs = prepare_stage_world_specs(cfg)
    params = SpawnFilterParams(
        bounds_size_m=float(world_cfg.get("bounds_size_m", DEFAULT_BOUNDS_SIZE_M)),
        origin_mode=str(world_cfg.get("origin_mode", "center")),
        origin_center_mode=str(world_cfg.get("origin_center_mode", "mean")),
        require_goal_in_bounds=bool(vehicles_cfg.get("require_goal_in_bounds", True)),
        skip_if_start_in_goal=bool(vehicles_cfg.get("skip_if_start_in_goal", True)),
        goal_radius_m=float(vehicles_cfg.get("goal_radius_m", goal_radius_default)),
        start_goal_thresh_m=(
            float(vehicles_cfg["start_goal_thresh_m"]) if vehicles_cfg.get("start_goal_thresh_m") is not None else None
        ),
        jump_break_m=float(road_cfg.get("jump_break_m", DEFAULT_JUMP_BREAK_M)),
    )
    meta = {
        "pool_yaml": str(pool_path),
        "scene_json_dir": str((cfg.get("io", {}) or {}).get("scene_json_dir", "")),
        "world_count_declared": world_cfg.get("world_count"),
        "assignment_fill_mode": world_cfg.get("assignment_fill_mode"),
        "assignment_entries": len(world_cfg.get("assignments", []) or []),
        "vehicles_block_present": bool(vehicles_cfg),
    }
    return specs, params, meta


# --------------------------------------------------------------------- output --
def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    keys = [("min", 0), ("p05", 5), ("p25", 25), ("median", 50), ("p75", 75), ("p95", 95), ("max", 100)]
    out = {name: float(np.percentile(arr, q)) for name, q in keys}
    out["mean"] = float(arr.mean())
    return out


def write_csv(records: list[SceneRecord], path: Path) -> None:
    rows = []
    for r in records:
        row: dict[str, Any] = {
            "scene": r.scene,
            "topology": r.topology,
            "topology_ambiguous": int(r.topology_ambiguous),
            "n_polylines_raw": r.n_polylines_raw,
            "n_polylines_cropped": r.n_polylines_cropped,
            "bbox_raw_x_m": round(r.bbox_raw_m[0], 2),
            "bbox_raw_y_m": round(r.bbox_raw_m[1], 2),
            "bbox_cropped_x_m": round(r.bbox_cropped_m[0], 2),
            "bbox_cropped_y_m": round(r.bbox_cropped_m[1], 2),
            "agents_count_valid": r.agents_count_valid,
            "agents_n_items": r.agents_n_items,
            "spawns_qualifying": r.spawns_qualifying,
            "rej_missing_coords": r.rej_missing_coords,
            "rej_start_out_of_bounds": r.rej_start_out_of_bounds,
            "rej_goal_out_of_bounds": r.rej_goal_out_of_bounds,
            "rej_start_in_goal": r.rej_start_in_goal,
            "spawn_crosscheck_ok": int(r.spawn_crosscheck_ok),
            "lane_len_cropped_m": round(r.lane_len_cropped_m, 1),
            "n_lane_polylines": r.n_lane_polylines,
            "n_lane_runs": r.n_lane_runs,
            "conflict_crossings": r.conflict_crossings,
            "conflict_junctions": r.conflict_junctions,
            "shallow_crossings": r.shallow_crossings,
            "shallow_junctions": r.shallow_junctions,
            "heading_modes": r.heading_modes,
            "heading_spread_deg": round(r.heading_spread_deg, 2),
            "circular_runs": r.circular_runs,
            "parallel_lanes": r.parallel_lanes,
        }
        for cls in ALL_CLASSES:
            row[f"n_{cls}_raw"] = r.poly_count_raw[cls]
            row[f"n_{cls}_cropped"] = r.poly_count_cropped[cls]
            row[f"len_{cls}_raw_m"] = round(r.length_raw_m[cls], 1)
            row[f"len_{cls}_cropped_m"] = round(r.length_cropped_m[cls], 1)
        rows.append(row)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    records: list[SceneRecord],
    *,
    params: SpawnFilterParams,
    meta: dict[str, Any],
    world_scene_names: list[str] | None,
) -> dict[str, Any]:
    n = len(records)
    spawns = [r.spawns_qualifying for r in records]
    topo_counts = Counter(r.topology for r in records)
    ambiguous = Counter(r.topology for r in records if r.topology_ambiguous)

    world_block: dict[str, Any] = {"unique_scenes": n}
    if world_scene_names is not None:
        per_scene = Counter(world_scene_names)
        world_block.update(
            {
                "total_worlds": len(world_scene_names),
                "worlds_per_unique_scene": {
                    "min": int(min(per_scene.values())),
                    "max": int(max(per_scene.values())),
                    "mean": round(len(world_scene_names) / max(n, 1), 3),
                },
                "scene_is_replicated": bool(max(per_scene.values()) > 1),
            }
        )

    rej_total = {
        "missing_coords": sum(r.rej_missing_coords for r in records),
        "start_out_of_bounds": sum(r.rej_start_out_of_bounds for r in records),
        "goal_out_of_bounds": sum(r.rej_goal_out_of_bounds for r in records),
        "start_in_goal": sum(r.rej_start_in_goal for r in records),
    }
    raw_agents = sum(r.agents_n_items for r in records)

    return {
        "pool": meta,
        "spawn_filter_params": asdict(params),
        "composition": world_block,
        "agents": {
            "raw_items_total": raw_agents,
            "count_valid_total": sum(r.agents_count_valid for r in records),
            "count_valid_equals_len_items_scenes": sum(
                1 for r in records if r.agents_count_valid == r.agents_n_items
            ),
            "per_scene_raw_items": _percentiles([r.agents_n_items for r in records]),
        },
        "spawn_yield": {
            "per_scene": _percentiles(spawns),
            "total_qualifying": int(sum(spawns)),
            "histogram_bin_edges": [0, 1, 2, 4, 8, 16, 32, 64, "inf"],
            "histogram_counts": np.histogram(spawns, bins=[0, 1, 2, 4, 8, 16, 32, 64, 10**6])[0].tolist(),
            "fraction_at_least": {
                str(k): round(float(np.mean([s >= k for s in spawns])), 4) for k in SPAWN_CAPS
            },
            "scenes_at_least": {str(k): int(sum(s >= k for s in spawns)) for k in SPAWN_CAPS},
            "scenes_with_zero": int(sum(s == 0 for s in spawns)),
        },
        "spawn_attrition": {
            "raw_agent_items": raw_agents,
            "rejected": rej_total,
            "surviving": int(sum(spawns)),
            "rejected_fraction": {
                k: (round(v / raw_agents, 4) if raw_agents else 0.0) for k, v in rej_total.items()
            },
        },
        "spawn_crosscheck": {
            "scenes_matching_env_extractor": int(sum(1 for r in records if r.spawn_crosscheck_ok)),
            "scenes_total": n,
            "mismatched_scenes": [r.scene for r in records if not r.spawn_crosscheck_ok],
        },
        "topology_heuristic": {
            "counts": {k: int(topo_counts.get(k, 0)) for k in TOPOLOGY_ORDER},
            "fractions": {k: round(topo_counts.get(k, 0) / max(n, 1), 4) for k in TOPOLOGY_ORDER},
            "ambiguous_counts": {k: int(ambiguous.get(k, 0)) for k in TOPOLOGY_ORDER},
            "ambiguous_total": int(sum(ambiguous.values())),
            "conflict_junctions_per_scene": _percentiles([r.conflict_junctions for r in records]),
        },
        "road_length_cropped_m": {
            cls: _percentiles([r.length_cropped_m[cls] for r in records]) for cls in ALL_CLASSES
        },
        "road_length_raw_m": {cls: _percentiles([r.length_raw_m[cls] for r in records]) for cls in ALL_CLASSES},
        "crop_survival_fraction_lane_center": _percentiles(
            [
                (r.length_cropped_m["lane_center"] / r.length_raw_m["lane_center"])
                for r in records
                if r.length_raw_m["lane_center"] > 0
            ]
        ),
        "polylines_per_scene_raw": _percentiles([r.n_polylines_raw for r in records]),
    }


# -------------------------------------------------------------------- figures --
def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_topology(records: list[SceneRecord], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    counts = Counter(r.topology for r in records)
    amb = Counter(r.topology for r in records if r.topology_ambiguous)
    labels = [k for k in TOPOLOGY_ORDER if counts.get(k, 0) > 0]
    total = max(len(records), 1)
    conf = [counts[k] - amb.get(k, 0) for k in labels]
    ambi = [amb.get(k, 0) for k in labels]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.2, 0.52 * len(labels) + 1.9), facecolor=SURFACE)
    ax.barh(y, conf, height=0.62, color=PALETTE[0], label="confident")
    ax.barh(y, ambi, height=0.62, left=conf, color="#c8d8ef", label="flagged ambiguous")
    for i, k in enumerate(labels):
        ax.text(
            counts[k] + total * 0.012,
            i,
            f"{counts[k]}  ({100.0 * counts[k] / total:.1f}%)",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_yticks(y, [k.replace("_", " ") for k in labels], color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("scenes", color=INK_2, fontsize=10)
    ax.set_xlim(0, max(counts.values()) * 1.28)
    _style_axes(ax)
    ax.grid(True, axis="x", color="#e6e5e0", linewidth=0.8)
    ax.grid(False, axis="y")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    ax.set_title(f"{title} — topology class (heuristic, n={len(records)})", color=INK, fontsize=11, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_spawn_yield(records: list[SceneRecord], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    spawns = np.asarray([r.spawns_qualifying for r in records], dtype=float)
    raw = np.asarray([r.agents_n_items for r in records], dtype=float)
    top = int(max(spawns.max(), raw.max())) + 2

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=SURFACE)
    ax = axes[0]
    bins = np.arange(0, top + 4, 4)
    ax.hist(raw, bins=bins, color="#d7d6d0", label="raw agents in JSON")
    ax.hist(spawns, bins=bins, color=PALETTE[0], label="qualifying spawns")
    ax.set_xlabel("agents per scene", color=INK_2, fontsize=10)
    ax.set_ylabel("scenes", color=INK_2, fontsize=10)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    ax.set_title("distribution", color=INK, fontsize=10, loc="left")

    ax = axes[1]
    ks = np.arange(0, top)
    frac = np.asarray([float((spawns >= k).mean()) for k in ks])
    ax.step(ks, 100.0 * frac, where="post", color=PALETTE[0], linewidth=2.0)
    for cap in SPAWN_CAPS:
        f = 100.0 * float((spawns >= cap).mean())
        ax.plot([cap], [f], "o", color=PALETTE[1], markersize=8, zorder=3)
        ax.annotate(
            f"≥{cap}: {f:.1f}%",
            xy=(cap, f),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9,
            color=INK,
        )
    ax.set_xlabel("agent cap k", color=INK_2, fontsize=10)
    ax.set_ylabel("scenes with ≥ k qualifying spawns (%)", color=INK_2, fontsize=10)
    ax.set_ylim(0, 105)
    _style_axes(ax)
    ax.set_title("spawn-yield survival curve", color=INK, fontsize=10, loc="left")

    fig.suptitle(f"{title} — spawn yield (exact, n={len(records)} scenes)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_road_length(records: list[SceneRecord], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    shown = ["lane_center", "road_edge", "road_line"]
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor=SURFACE)

    ax = axes[0]
    for cls, colour in zip(shown, colors):
        vals = np.sort(np.asarray([r.length_cropped_m[cls] for r in records], dtype=float))
        y = 100.0 * np.arange(1, vals.size + 1) / vals.size
        ax.step(vals, y, where="post", color=colour, linewidth=2.0, label=cls.replace("_", " "))
    ax.set_xlabel("length inside the crop per scene (m)", color=INK_2, fontsize=10)
    ax.set_ylabel("scenes (%)", color=INK_2, fontsize=10)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK_2)
    ax.set_title("cropped road length, ECDF", color=INK, fontsize=10, loc="left")

    ax = axes[1]
    surv = []
    for cls in shown:
        vals = [r.length_cropped_m[cls] / r.length_raw_m[cls] for r in records if r.length_raw_m[cls] > 0]
        surv.append(np.asarray(vals, dtype=float) * 100.0)
    parts = ax.boxplot(
        surv,
        vert=True,
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color=INK, linewidth=1.6),
        flierprops=dict(marker="o", markersize=3, markerfacecolor=INK_MUTED, markeredgecolor="none", alpha=0.5),
        whiskerprops=dict(color=INK_MUTED),
        capprops=dict(color=INK_MUTED),
    )
    for patch, colour in zip(parts["boxes"], colors):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
        patch.set_edgecolor("none")
    ax.set_xticks(range(1, len(shown) + 1), [c.replace("_", " ") for c in shown], color=INK)
    ax.set_ylabel("length surviving the crop (%)", color=INK_2, fontsize=10)
    ax.set_ylim(0, 100)
    _style_axes(ax)
    ax.set_title("fraction of each scene's road kept by the 200 m crop", color=INK, fontsize=10, loc="left")

    fig.suptitle(f"{title} — road length by class (exact)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_attrition(records: list[SceneRecord], out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    stages = [
        ("surviving", sum(r.spawns_qualifying for r in records), PALETTE[0]),
        ("start out of crop", sum(r.rej_start_out_of_bounds for r in records), PALETTE[1]),
        ("goal out of crop", sum(r.rej_goal_out_of_bounds for r in records), PALETTE[2]),
        ("start already in goal", sum(r.rej_start_in_goal for r in records), PALETTE[4]),
        ("missing coordinates", sum(r.rej_missing_coords for r in records), "#c9c8c2"),
    ]
    total = sum(v for _, v, _ in stages)
    fig, ax = plt.subplots(figsize=(10.0, 2.9), facecolor=SURFACE)
    left = 0.0
    for name, value, colour in stages:
        if value <= 0:
            continue
        ax.barh([0], [value], left=left, height=0.5, color=colour)
        pct = 100.0 * value / max(total, 1)
        if pct >= 4.0:
            ax.text(left + value / 2.0, 0, f"{pct:.0f}%", ha="center", va="center", fontsize=9, color="#ffffff")
        left += value
    ax.set_xlim(0, total)
    ax.set_ylim(-1.15, 0.4)
    ax.set_yticks([])
    _style_axes(ax)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, v, c in stages if v > 0]
    labels = [f"{n} ({100.0 * v / max(total, 1):.1f}%)" for n, v, _ in stages if v > 0]
    ax.legend(handles, labels, frameon=False, fontsize=9, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 0.30), labelcolor=INK_2, handlelength=1.1, columnspacing=1.6)
    ax.set_title(
        f"{title} — where raw agents are lost (exact); "
        f"n={total} raw agent records in the pool's scene JSONs",
        color=INK,
        fontsize=10.5,
        loc="left",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_gallery(
    records: list[SceneRecord],
    scene_paths: dict[str, Path],
    params: SpawnFilterParams,
    out: Path,
    title: str,
    per_class: int,
) -> None:
    import matplotlib.pyplot as plt

    classes = [k for k in TOPOLOGY_ORDER if any(r.topology == k for r in records)]
    if not classes:
        return
    fig, axes = plt.subplots(
        len(classes),
        per_class,
        figsize=(2.35 * per_class, 2.55 * len(classes)),
        facecolor=SURFACE,
        squeeze=False,
    )
    half = 0.5 * params.bounds_size_m
    for row, cls in enumerate(classes):
        members = sorted((r for r in records if r.topology == cls), key=lambda r: r.scene)
        # deterministic, evenly spaced through the class so the row is not all
        # near-duplicates from one part of the pool
        if len(members) <= per_class:
            picks = members
        else:
            idx = np.linspace(0, len(members) - 1, per_class).round().astype(int)
            picks = [members[i] for i in idx]
        for col in range(per_class):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color("#dedcd6")
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)
            ax.set_aspect("equal")
            if col >= len(picks):
                ax.axis("off")
                continue
            rec = picks[col]
            scene = json.loads(scene_paths[rec.scene].read_text())
            centre = _scene_center_xy(
                scene, origin_mode=params.origin_mode, origin_center_mode=params.origin_center_mode
            )
            for poly in (scene.get("road", {}) or {}).get("polylines", []) or []:
                code = int(poly.get("type", -1))
                if code not in LANE_CENTER_TYPES and code not in (15, 16):
                    continue
                xy = _polyline_xy(poly)
                if xy is None:
                    continue
                lane = code in LANE_CENTER_TYPES
                for run in _kept_runs(xy - centre, half=half, jump_break_m=params.jump_break_m):
                    ax.plot(
                        run[:, 0],
                        run[:, 1],
                        linewidth=1.0 if lane else 0.6,
                        color=PALETTE[0] if lane else "#cfcec8",
                        solid_capstyle="round",
                        zorder=2 if lane else 1,
                    )
            spawns = extract_vehicle_spawns_from_json(
                scene_paths[rec.scene],
                bounds_size_m=params.bounds_size_m,
                origin_mode=params.origin_mode,
                origin_center_mode=params.origin_center_mode,
                max_controllable=UNCAPPED,
                require_goal_in_bounds=params.require_goal_in_bounds,
                skip_if_start_in_goal=params.skip_if_start_in_goal,
                goal_radius_m=params.goal_radius_m,
                start_goal_thresh_m=params.start_goal_thresh_m,
            )
            if spawns:
                pts = np.asarray([s.start_local_xyz[:2] for s in spawns], dtype=float)
                ax.plot(pts[:, 0], pts[:, 1], "o", markersize=2.6, color=PALETTE[1], zorder=3)
            flag = "*" if rec.topology_ambiguous else ""
            ax.set_title(
                f"{rec.scene.replace('scene_', '').replace('.json', '')}{flag}  "
                f"{rec.spawns_qualifying} sp",
                fontsize=7.5,
                color=INK_2,
                pad=3,
            )
            if col == 0:
                ax.set_ylabel(cls.replace("_", "\n"), fontsize=8.5, color=INK, rotation=0, ha="right", va="center",
                              labelpad=14)
    fig.suptitle(
        f"{title} — representative scenes per topology class "
        "(blue: lane centres kept by the crop; grey: road edges; orange: qualifying spawn starts; * = ambiguous)",
        color=INK,
        fontsize=9.5,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------ main --
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pool", type=str, help="scene-pool YAML (io.scene_json_dir + world.assignments)")
    src.add_argument("--scene-dir", type=str, help="directory of scene_*.json, analysed with default filter params")
    ap.add_argument("--out", type=str, required=True, help="output directory")
    ap.add_argument("--goal-radius-m", type=float, default=DEFAULT_GOAL_RADIUS_M,
                    help="fallback goal radius when the pool YAML has no vehicles block "
                         "(default 3.0 = env.goal_reached_threshold_m in configs/scene_factory)")
    ap.add_argument("--bounds-size-m", type=float, default=None, help="override world.bounds_size_m")
    ap.add_argument("--gallery-per-class", type=int, default=5)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--label", type=str, default="", help="title shown on the figures (default: --out basename)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or out_dir.name

    world_scene_names: list[str] | None = None
    if args.pool:
        pool_path = Path(args.pool)
        specs, params, meta = resolve_pool(pool_path, args.goal_radius_m)
        world_scene_names = [s.scene_json_name for s in specs]
        paths = {s.scene_json_name: Path(s.scene_json_path) for s in specs}
    else:
        scene_dir = Path(args.scene_dir)
        files = sorted(scene_dir.glob("*.json"))
        if not files:
            raise SystemExit(f"no scene JSONs under {scene_dir}")
        paths = {f.name: f for f in files}
        params = SpawnFilterParams(
            bounds_size_m=DEFAULT_BOUNDS_SIZE_M,
            origin_mode="center",
            origin_center_mode="mean",
            require_goal_in_bounds=True,
            skip_if_start_in_goal=True,
            goal_radius_m=args.goal_radius_m,
            start_goal_thresh_m=None,
            jump_break_m=DEFAULT_JUMP_BREAK_M,
        )
        meta = {"scene_dir": str(scene_dir), "pool_yaml": None, "note": "no pool YAML; env default filter params used"}

    if args.bounds_size_m is not None:
        params.bounds_size_m = float(args.bounds_size_m)

    names = sorted(paths)
    print(f"[analyze] {label}: {len(names)} unique scenes"
          + (f" across {len(world_scene_names)} worlds" if world_scene_names else ""))
    print(f"[analyze] filter params: {asdict(params)}")

    records: list[SceneRecord] = []
    for i, name in enumerate(names, 1):
        records.append(analyse_scene(paths[name], params))
        if i % 25 == 0 or i == len(names):
            print(f"  ... {i}/{len(names)}")

    summary = build_summary(records, params=params, meta=meta, world_scene_names=world_scene_names)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(records, out_dir / "per_scene.csv")

    bad = summary["spawn_crosscheck"]["mismatched_scenes"]
    if bad:
        print(f"[WARNING] spawn attribution disagreed with the env extractor on {len(bad)} scenes: {bad[:5]}")

    if not args.no_figures:
        import matplotlib

        matplotlib.use("Agg")
        fig_topology(records, out_dir / "topology_distribution.png", label)
        fig_spawn_yield(records, out_dir / "spawn_yield.png", label)
        fig_road_length(records, out_dir / "road_length_by_class.png", label)
        fig_attrition(records, out_dir / "spawn_attrition.png", label)
        fig_gallery(records, paths, params, out_dir / "scene_gallery.png", label, max(1, args.gallery_per_class))

    topo = summary["topology_heuristic"]["counts"]
    print("\n=== topology (heuristic) ===")
    for k in TOPOLOGY_ORDER:
        if topo[k]:
            amb = summary["topology_heuristic"]["ambiguous_counts"][k]
            print(f"  {k:20s} {topo[k]:4d}  ({100.0 * topo[k] / len(records):5.1f}%)   ambiguous {amb}")
    sy = summary["spawn_yield"]
    print("\n=== qualifying spawns per scene (exact) ===")
    print("  " + "  ".join(f"{k}={v:.1f}" for k, v in sy["per_scene"].items()))
    print("  fraction of scenes with >= k spawns: "
          + "  ".join(f"{k}:{100.0 * v:.1f}%" for k, v in sy["fraction_at_least"].items()))
    at = summary["spawn_attrition"]
    print(f"\n=== attrition (exact) === raw={at['raw_agent_items']} surviving={at['surviving']}")
    for k, v in at["rejected"].items():
        print(f"  rejected {k:22s} {v:6d}  ({100.0 * at['rejected_fraction'][k]:.1f}%)")
    print(f"\n[out] {out_dir}")


if __name__ == "__main__":
    main()
