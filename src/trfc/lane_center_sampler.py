"""Lane-centerline start/goal sampling for evaluation worlds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import math
import numpy as np


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return int(default)


def _coerce_xyz(points: Any) -> np.ndarray | None:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return None
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
    return arr[:, :3].astype(np.float32, copy=False)


def compute_scene_center_from_road(scene_cfg: Dict[str, Any]) -> np.ndarray:
    polylines = list((scene_cfg.get("road", {}) or {}).get("polylines", []) or [])
    chunks: List[np.ndarray] = []
    for pl in polylines:
        pts = _coerce_xyz(pl.get("xyz", None))
        if pts is not None:
            chunks.append(pts)
    if not chunks:
        return np.zeros((3,), dtype=np.float32)
    all_pts = np.concatenate(chunks, axis=0)
    return all_pts.mean(axis=0).astype(np.float32, copy=False)


def _is_in_local_bounds(
    *,
    world_xy: np.ndarray,
    scene_center_xy: np.ndarray,
    bounds_size_m: float,
    origin_mode: str,
) -> bool:
    if str(origin_mode).lower() == "center":
        local_xy = world_xy - scene_center_xy
    else:
        local_xy = world_xy
    half = 0.5 * float(bounds_size_m)
    return (abs(float(local_xy[0])) <= half) and (abs(float(local_xy[1])) <= half)


@dataclass(frozen=True)
class LanePolyline:
    polyline_idx: int
    road_type: int
    points_xyz: np.ndarray
    cumulative_s_m: np.ndarray
    length_m: float

    def sample(self, s_m: float) -> Tuple[np.ndarray, float]:
        if self.points_xyz.shape[0] < 2:
            raise ValueError("LanePolyline must have at least two points")
        s = float(np.clip(s_m, 0.0, self.length_m))
        cum = self.cumulative_s_m
        idx = int(np.searchsorted(cum, s, side="right") - 1)
        idx = max(0, min(idx, self.points_xyz.shape[0] - 2))
        seg_len = float(cum[idx + 1] - cum[idx])
        alpha = 0.0 if seg_len <= 1e-6 else float((s - float(cum[idx])) / seg_len)

        p0 = self.points_xyz[idx]
        p1 = self.points_xyz[idx + 1]
        point = (1.0 - alpha) * p0 + alpha * p1

        tangent = p1[:2] - p0[:2]
        tan_norm = float(np.linalg.norm(tangent))
        if tan_norm <= 1e-6:
            # Robust fallback for degenerate segment.
            if idx + 2 < self.points_xyz.shape[0]:
                tangent = self.points_xyz[idx + 2, :2] - p0[:2]
                tan_norm = float(np.linalg.norm(tangent))
            if tan_norm <= 1e-6 and idx > 0:
                tangent = p1[:2] - self.points_xyz[idx - 1, :2]
                tan_norm = float(np.linalg.norm(tangent))
        if tan_norm <= 1e-6:
            yaw = 0.0
        else:
            yaw = float(math.atan2(float(tangent[1]), float(tangent[0])))
        return point.astype(np.float32, copy=False), yaw


@dataclass(frozen=True)
class LaneStartGoalSample:
    sample_idx: int
    polyline_idx: int
    road_type: int
    travel_distance_m: float
    start_xyz: Tuple[float, float, float]
    start_yaw_rad: float
    goal_xyz: Tuple[float, float, float]
    goal_yaw_rad: float

    def to_scene_agent_item(
        self,
        *,
        agent_id: int,
        track_idx: int,
        agent_type: int = 1,
    ) -> Dict[str, Any]:
        return {
            "track_idx": int(track_idx),
            "is_sdc": False,
            "agent_type": int(agent_type),
            "agent_id": int(agent_id),
            "start": {
                "x": float(self.start_xyz[0]),
                "y": float(self.start_xyz[1]),
                "z": float(self.start_xyz[2]),
                "yaw": float(self.start_yaw_rad),
            },
            "end": {
                "x": float(self.goal_xyz[0]),
                "y": float(self.goal_xyz[1]),
                "z": float(self.goal_xyz[2]),
                "yaw": float(self.goal_yaw_rad),
            },
        }


def extract_lane_polylines(
    scene_cfg: Dict[str, Any],
    *,
    lane_types: Sequence[int] = (1, 2),
    min_polyline_length_m: float = 5.0,
    max_segment_gap_m: float | None = None,
    respect_passable: bool = True,
) -> List[LanePolyline]:
    road = scene_cfg.get("road", {}) or {}
    polylines = list(road.get("polylines", []) or [])
    lane_type_set = {int(x) for x in lane_types}

    out: List[LanePolyline] = []
    for idx, pl in enumerate(polylines):
        if respect_passable and not pl.get("passable", True):
            continue
        road_type = _safe_int(pl.get("type", -1), -1)
        if lane_type_set and road_type not in lane_type_set:
            continue
        pts = _coerce_xyz(pl.get("xyz", None))
        if pts is None:
            continue
        seg = pts[1:, :2] - pts[:-1, :2]
        seg_len = np.linalg.norm(seg, axis=1).astype(np.float32)
        if seg_len.size == 0:
            continue
        if max_segment_gap_m is not None:
            gap_thr = float(max_segment_gap_m)
            if gap_thr > 0.0 and bool(np.any(seg_len > gap_thr)):
                # Reject broken polylines with discontinuous point jumps.
                continue
        cum = np.concatenate(
            [np.zeros((1,), dtype=np.float32), np.cumsum(seg_len, dtype=np.float32)],
            axis=0,
        )
        length = float(cum[-1])
        if length < float(min_polyline_length_m):
            continue
        out.append(
            LanePolyline(
                polyline_idx=int(idx),
                road_type=int(road_type),
                points_xyz=pts,
                cumulative_s_m=cum,
                length_m=length,
            )
        )
    return out


def _any_too_close(xy: np.ndarray, others: Iterable[np.ndarray], threshold_m: float) -> bool:
    threshold2 = float(threshold_m) * float(threshold_m)
    for other in others:
        d = xy - other
        if float(d[0] * d[0] + d[1] * d[1]) < threshold2:
            return True
    return False


def sample_lane_center_start_goal_pairs(
    scene_cfg: Dict[str, Any],
    *,
    num_agents: int,
    bounds_size_m: float,
    origin_mode: str = "center",
    lane_types: Sequence[int] = (1, 2),
    min_travel_distance_m: float = 20.0,
    max_travel_distance_m: float = 60.0,
    min_start_gap_m: float = 8.0,
    min_goal_gap_m: float = 6.0,
    endpoint_margin_m: float = 2.0,
    min_polyline_length_m: float = 5.0,
    max_segment_gap_m: float | None = None,
    seed: int = 42,
    max_attempts: int | None = None,
) -> List[LaneStartGoalSample]:
    if int(num_agents) <= 0:
        raise ValueError(f"num_agents must be > 0, got {num_agents}")
    if float(min_travel_distance_m) <= 0:
        raise ValueError("min_travel_distance_m must be > 0")
    if float(max_travel_distance_m) < float(min_travel_distance_m):
        raise ValueError("max_travel_distance_m must be >= min_travel_distance_m")

    lanes = extract_lane_polylines(
        scene_cfg,
        lane_types=lane_types,
        min_polyline_length_m=min_polyline_length_m,
        max_segment_gap_m=max_segment_gap_m,
    )
    if not lanes:
        raise RuntimeError(
            "No usable lane polylines found for requested lane_types. "
            f"lane_types={list(lane_types)}"
        )

    scene_center = compute_scene_center_from_road(scene_cfg)
    scene_center_xy = scene_center[:2].astype(np.float32, copy=False)

    max_route = float(max_travel_distance_m)
    min_route = float(min_travel_distance_m)
    margin = float(max(0.0, endpoint_margin_m))

    viable_lanes: List[LanePolyline] = []
    lane_weights: List[float] = []
    for lane in lanes:
        if lane.length_m < (min_route + 2.0 * margin):
            continue
        viable_lanes.append(lane)
        lane_weights.append(float(lane.length_m))
    if not viable_lanes:
        raise RuntimeError(
            "No lane polyline long enough for requested travel distance. "
            f"min_travel_distance_m={min_route}"
        )
    weights = np.asarray(lane_weights, dtype=np.float64)
    weights = weights / np.clip(weights.sum(), 1e-9, None)

    rng = np.random.default_rng(int(seed))
    picked_starts_xy: List[np.ndarray] = []
    picked_goals_xy: List[np.ndarray] = []
    samples: List[LaneStartGoalSample] = []

    attempts_limit = int(max_attempts) if max_attempts is not None else int(300 * int(num_agents))
    attempts = 0
    while len(samples) < int(num_agents) and attempts < attempts_limit:
        attempts += 1
        lane_idx = int(rng.choice(len(viable_lanes), p=weights))
        lane = viable_lanes[lane_idx]

        travel = float(rng.uniform(min_route, max_route))
        if lane.length_m <= (travel + 2.0 * margin):
            continue
        s_goal_min = travel + margin
        s_goal_max = lane.length_m - margin
        if s_goal_max <= s_goal_min:
            continue
        s_goal = float(rng.uniform(s_goal_min, s_goal_max))
        s_start = s_goal - travel

        start_xyz, start_yaw = lane.sample(s_start)
        goal_xyz, goal_yaw = lane.sample(s_goal)
        start_xy = start_xyz[:2]
        goal_xy = goal_xyz[:2]

        if not _is_in_local_bounds(
            world_xy=start_xy,
            scene_center_xy=scene_center_xy,
            bounds_size_m=float(bounds_size_m),
            origin_mode=origin_mode,
        ):
            continue
        if not _is_in_local_bounds(
            world_xy=goal_xy,
            scene_center_xy=scene_center_xy,
            bounds_size_m=float(bounds_size_m),
            origin_mode=origin_mode,
        ):
            continue

        if _any_too_close(start_xy, picked_starts_xy, float(min_start_gap_m)):
            continue
        if _any_too_close(goal_xy, picked_goals_xy, float(min_goal_gap_m)):
            continue

        sample = LaneStartGoalSample(
            sample_idx=len(samples),
            polyline_idx=int(lane.polyline_idx),
            road_type=int(lane.road_type),
            travel_distance_m=float(travel),
            start_xyz=(float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2])),
            start_yaw_rad=float(start_yaw),
            goal_xyz=(float(goal_xyz[0]), float(goal_xyz[1]), float(goal_xyz[2])),
            goal_yaw_rad=float(goal_yaw),
        )
        samples.append(sample)
        picked_starts_xy.append(start_xy.astype(np.float32, copy=True))
        picked_goals_xy.append(goal_xy.astype(np.float32, copy=True))

    if len(samples) < int(num_agents):
        raise RuntimeError(
            "Could not sample enough lane-center start/goal pairs. "
            f"requested={num_agents} sampled={len(samples)} attempts={attempts} "
            f"lane_types={list(lane_types)} min_route={min_route} max_route={max_route} "
            f"bounds_size_m={bounds_size_m} origin_mode={origin_mode}"
        )
    return samples


def build_scene_with_sampled_agents(
    source_scene_cfg: Dict[str, Any],
    samples: Sequence[LaneStartGoalSample],
    *,
    agent_id_start: int = 10000,
    agent_type: int = 1,
) -> Dict[str, Any]:
    scene = {
        "meta": dict((source_scene_cfg.get("meta", {}) or {})),
        "road": dict((source_scene_cfg.get("road", {}) or {})),
        "agents": {"items": []},
    }
    items: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        items.append(
            sample.to_scene_agent_item(
                agent_id=int(agent_id_start) + int(idx),
                track_idx=int(idx),
                agent_type=int(agent_type),
            )
        )
    scene["agents"]["items"] = items
    return scene


def _build_host_lane_polyline(
    scene_cfg: Dict[str, Any], host_polyline_id: int
) -> LanePolyline:
    """Build a LanePolyline for the scene polyline whose JSON `id` == host_polyline_id.

    `extract_lane_polylines` keys on enumeration index, but a workzone box names
    its host lane by the JSON `id` field, so it is resolved directly here.
    """
    road = scene_cfg.get("road", {}) or {}
    polylines = list(road.get("polylines", []) or [])
    for idx, pl in enumerate(polylines):
        if _safe_int(pl.get("id", -1), -1) != int(host_polyline_id):
            continue
        pts = _coerce_xyz(pl.get("xyz", None))
        if pts is None or pts.shape[0] < 2:
            break
        seg_len = np.linalg.norm(pts[1:, :2] - pts[:-1, :2], axis=1).astype(np.float32)
        cum = np.concatenate(
            [np.zeros((1,), dtype=np.float32), np.cumsum(seg_len, dtype=np.float32)], axis=0
        )
        return LanePolyline(
            polyline_idx=int(idx),
            road_type=_safe_int(pl.get("type", -1), -1),
            points_xyz=pts.astype(np.float32, copy=False),
            cumulative_s_m=cum,
            length_m=float(cum[-1]),
        )
    raise RuntimeError(f"host polyline id={host_polyline_id} not found or too short")


def _nearest_s_on_lane(lane: LanePolyline, xy: np.ndarray) -> float:
    """Arc-length of the point on `lane` closest to `xy` (segment projection).

    Used to project the closed-lane box center onto a neighbour open lane so the
    open-lane convoy shares the same longitudinal workzone reference.
    """
    pts = lane.points_xyz[:, :2].astype(np.float32, copy=False)
    cum = lane.cumulative_s_m
    q = np.asarray(xy, dtype=np.float32).reshape(2)
    best_s = 0.0
    best_d2 = float("inf")
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        seg = p1 - p0
        seg_len2 = float(seg[0] * seg[0] + seg[1] * seg[1])
        if seg_len2 <= 1e-12:
            t = 0.0
            seg_len = 0.0
        else:
            t = float(((q[0] - p0[0]) * seg[0] + (q[1] - p0[1]) * seg[1]) / seg_len2)
            t = max(0.0, min(1.0, t))
            seg_len = math.sqrt(seg_len2)
        proj = p0 + t * seg
        d = q - proj
        d2 = float(d[0] * d[0] + d[1] * d[1])
        if d2 < best_d2:
            best_d2 = d2
            best_s = float(cum[i]) + t * seg_len
    return best_s


def _upstream_capacity_for_lane(
    lane: LanePolyline,
    box_entry_s: float,
    *,
    scene_center_xy: np.ndarray,
    bounds_size_m: float,
    spawn_spacing_m: float,
    origin_mode: str = "center",
    step_m: float = 0.5,
) -> int:
    """floor(upstream_length / spacing) for one lane's holding room.

    ``box_entry_s`` is the arc-length on ``lane`` where the workzone box begins
    (the closed lane's box edge, or the projected box centre on an open lane).
    Walk s DOWNWARD (upstream, against travel) from there in ``step_m`` steps and
    take the smallest CONTIGUOUS in-crop arc-length as the usable upstream start;
    the walk stops at the first out-of-bounds sample (the crop exit). Downstream
    of the box is never counted. ``upstream_length = box_entry_s - upstream_start``
    (raw — no approach-gap subtraction; the sampler drops any overshoot safely).
    """
    spacing = float(spawn_spacing_m)
    if spacing <= 0.0:
        return 0
    entry_s = float(np.clip(box_entry_s, 0.0, lane.length_m))
    # The box entry itself must be in-crop, else there is no upstream room here.
    entry_pt, _ = lane.sample(entry_s)
    if not _is_in_local_bounds(
        world_xy=entry_pt[:2],
        scene_center_xy=scene_center_xy,
        bounds_size_m=float(bounds_size_m),
        origin_mode=origin_mode,
    ):
        return 0
    upstream_start_s = entry_s
    s = entry_s
    step = max(1e-3, float(step_m))
    while s > 0.0:
        s = max(0.0, s - step)
        pt, _ = lane.sample(s)
        if _is_in_local_bounds(
            world_xy=pt[:2],
            scene_center_xy=scene_center_xy,
            bounds_size_m=float(bounds_size_m),
            origin_mode=origin_mode,
        ):
            upstream_start_s = s
        else:
            break
    upstream_len = max(0.0, entry_s - upstream_start_s)
    return int(math.floor(upstream_len / spacing))


def compute_multilane_capacity(
    scene_cfg: Dict[str, Any],
    *,
    spawn_spacing_m: float = 8.0,
    bounds_size_m: float = 200.0,
    origin_mode: str = "center",
) -> Dict[str, Any]:
    """Per-map traffic capacity from upstream holding length (pure numpy).

    ``capacity_map = Sum over involved lanes L of floor(upstream_length(L)/spacing)``
    where the involved lanes are the merge contract's ``closed_lane_id`` plus
    ``open_lane_ids`` (from ``scene_cfg["workzone"]["forbidden_boxes"][0]``), and
    ``upstream_length(L)`` is the in-crop arc length from the workzone box entry
    BACKWARD (upstream) to the crop edge on that lane. The box entry is
    ``center_s_m - half_len`` on the closed lane and the projected box centre
    (``_nearest_s_on_lane``) on each open lane. Constant ``1/spawn_spacing_m`` is
    applied PER-LANE (``c_L``) so a future lane-type-dependent constant is a
    one-line change; all lanes use ``1/spawn_spacing_m`` today.

    This is the SINGLE source of truth: the generator bakes the result into the
    scene JSON and the env uses it as a fallback when no baked value is present.
    Returns ``{"capacity": int, "per_lane": {lane_id: int}, "spawn_spacing_m":
    float}``; returns capacity 0 (no raise) on a missing/invalid merge contract.
    """
    spacing = float(spawn_spacing_m)
    empty = {"capacity": 0, "per_lane": {}, "spawn_spacing_m": spacing}
    wz = scene_cfg.get("workzone", {}) or {}
    boxes = wz.get("forbidden_boxes", []) or []
    if not boxes:
        return empty
    box = boxes[0]

    closed_id = _safe_int(box.get("closed_lane_id", -1), -1)
    box_s = _safe_float(box.get("center_s_m", -1.0), -1.0)
    half_len = max(0.0, _safe_float(box.get("half_len", 0.0), 0.0))
    if closed_id < 0 or box_s < 0.0:
        return empty

    open_ids: List[int] = []
    for x in box.get("open_lane_ids", None) or []:
        oid = _safe_int(x, -1)
        if oid >= 0 and oid not in open_ids:
            open_ids.append(oid)

    try:
        closed_lane = _build_host_lane_polyline(scene_cfg, closed_id)
        open_lanes = [(oid, _build_host_lane_polyline(scene_cfg, oid)) for oid in open_ids]
    except RuntimeError:
        return empty

    scene_center = compute_scene_center_from_road(scene_cfg)
    scene_center_xy = scene_center[:2].astype(np.float32, copy=False)

    per_lane: Dict[int, int] = {}
    # Closed lane: box entry is the upstream box edge (center_s - half_len).
    closed_entry_s = float(box_s) - float(half_len)
    per_lane[int(closed_id)] = _upstream_capacity_for_lane(
        closed_lane,
        closed_entry_s,
        scene_center_xy=scene_center_xy,
        bounds_size_m=bounds_size_m,
        spawn_spacing_m=spacing,
        origin_mode=origin_mode,
    )

    # Open lanes: box entry is the projected box centre on that lane.
    box_center_xy = closed_lane.sample(float(box_s))[0][:2]
    for oid, lane in open_lanes:
        ref_s = _nearest_s_on_lane(lane, box_center_xy)
        per_lane[int(oid)] = _upstream_capacity_for_lane(
            lane,
            ref_s,
            scene_center_xy=scene_center_xy,
            bounds_size_m=bounds_size_m,
            spawn_spacing_m=spacing,
            origin_mode=origin_mode,
        )

    capacity = int(sum(per_lane.values()))
    return {"capacity": capacity, "per_lane": per_lane, "spawn_spacing_m": spacing}


def sample_multilane_merge_start_goal_pairs(
    scene_cfg: Dict[str, Any],
    *,
    num_agents: int,
    bounds_size_m: float,
    origin_mode: str = "center",
    approach_gap_m: float = 33.0,
    spawn_spacing_m: float = 8.0,
    spacing_jitter_m: float = 1.5,
    lateral_jitter_m: float = 0.5,
    goal_clearance_m: float = 15.0,
    return_goal_clearance_m: float = 60.0,
    goal_spacing_m: float = 8.0,
    min_upstream_margin_m: float = 2.0,
    merge_return_frac: float = 1.0,
    lane_distribution_mode: str = "uniform",
    seed: int = 42,
) -> List[LaneStartGoalSample]:
    """Place `num_agents` across the REAL closed + open lanes of a merge workzone.

    Reads the merge contract from `scene_cfg["workzone"]["forbidden_boxes"][0]`:
    `closed_lane_id`, `open_lane_ids` (list, nearest first), `center_s_m`,
    `half_len`, and optional `parallel_s_window_m`.

    Group lanes are `[closed] + open_lanes` (group index 0 = closed). With
    `lane_distribution_mode="uniform"` (the only mode) each of the `num_agents`
    cars is drawn INDEPENDENTLY and UNIFORMLY to one group lane, so the lane
    counts are random/imbalanced BY DESIGN — the caller's density knob sets how
    many cars appear, not a forced even split. `num_agents` is already the scaled
    car count N (the env applies the density scale); the sampler does not scale.
    The lane draw is seeded off `seed` (a dedicated sub-stream, independent of the
    spawn-jitter stream) so it is byte-reproducible across cone candidates sharing
    the OD fixed seed.

    - CLOSED-lane cars start UPSTREAM and, with `merge_return_frac=1.0`, finish
      DOWNSTREAM on the SAME closed lane (they return to their lane after the
      box). The out-and-back merge maneuver is emergent driver behaviour; OD
      only sets endpoints. For `merge_return_frac < 1.0`, a fraction instead get
      their goal projected onto the nearest open lane.
      Their downstream goal uses `return_goal_clearance_m` (default 60 m) so it
      lands BEYOND the exit taper plus merge-back room: the exit taper is the
      optimizer's design variable (max ~40 m), and OD must stay fixed across
      designs, so this must exceed the WORST-CASE exit taper (~40 m) + merge-back
      distance (~20 m) rather than track any specific taper.
    - OPEN-lane cars drive STRAIGHT through on their own open lane. The closed
      box center is projected onto each open lane to give that lane's
      longitudinal reference so its convoy also clears the workzone region.

    There is exactly ONE shared goal per lane: every car assigned to a lane
    targets the SAME goal_xyz (clamped to `lane_len - margin` so it never runs
    off the end). Only SPAWNS are per-car staggered/unique. This is safe because
    agents park/vanish on `goal_reached`, so staggered arrivals clear the single
    spot one at a time; it also stops short-lane cars from being dropped for a
    goal chain overflowing the lane. `goal_spacing_m` is retained for API
    compatibility but no longer used (goals no longer stagger).

    Clearances are measured from the box EDGE (`center_s +/- half_len`),
    consistent with `sample_workzone_start_goal_pairs`. A spawn (or the shared
    goal) failing the scene-crop in-bounds check is dropped ("place however many
    fit"). Returns `List[LaneStartGoalSample]` (unchanged env OD machinery).
    Raises `RuntimeError` only on a missing/invalid contract or zero in-bounds
    spawns.
    """
    wz = scene_cfg.get("workzone", {}) or {}
    boxes = wz.get("forbidden_boxes", []) or []
    if not boxes:
        raise RuntimeError("scene has no workzone.forbidden_boxes for multilane merge OD")
    box = boxes[0]

    closed_id = _safe_int(box.get("closed_lane_id", -1), -1)
    box_s = _safe_float(box.get("center_s_m", -1.0), -1.0)
    half_len = max(0.0, _safe_float(box.get("half_len", 0.0), 0.0))
    if closed_id < 0 or box_s < 0.0:
        raise RuntimeError("merge contract lacks closed_lane_id / center_s_m")

    open_ids_raw = box.get("open_lane_ids", None)
    if not open_ids_raw:
        raise RuntimeError("merge contract missing open_lane_ids")
    open_ids: List[int] = []
    for x in open_ids_raw:
        oid = _safe_int(x, -1)
        if oid >= 0 and oid not in open_ids:
            open_ids.append(oid)
    if not open_ids:
        raise RuntimeError("merge contract open_lane_ids has no valid lane id")

    closed_lane = _build_host_lane_polyline(scene_cfg, closed_id)
    open_lanes = [_build_host_lane_polyline(scene_cfg, oid) for oid in open_ids]
    group = [closed_lane] + open_lanes
    n_lanes = len(group)

    if lane_distribution_mode != "uniform":
        raise ValueError(
            f"unknown lane_distribution_mode {lane_distribution_mode!r} "
            "(only 'uniform' is implemented)"
        )

    scene_center = compute_scene_center_from_road(scene_cfg)
    scene_center_xy = scene_center[:2].astype(np.float32, copy=False)
    rng = np.random.default_rng(int(seed))
    # Per-car lane assignment: each of the N cars is drawn INDEPENDENTLY and
    # UNIFORMLY to a group lane (index 0 = closed). Counts are random/imbalanced
    # by design — do NOT force a 50/50 split. Drawn from a DEDICATED sub-stream
    # seeded off `seed` so it (a) is byte-reproducible across cone candidates that
    # share the OD fixed seed, and (b) does not perturb the spawn-jitter stream
    # (`rng`), which stays byte-identical to the pre-density behaviour.
    lane_rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(1)[0])
    n_cars = max(0, int(num_agents))
    lane_choices = lane_rng.integers(0, n_lanes, size=n_cars)
    counts = [int(np.count_nonzero(lane_choices == li)) for li in range(n_lanes)]

    # Longitudinal reference per lane: box_s on the closed lane, projected box
    # center on each open lane.
    box_center_xy = closed_lane.sample(float(box_s))[0][:2]
    open_refs = [_nearest_s_on_lane(lane, box_center_xy) for lane in open_lanes]

    def _normal_at(lane: LanePolyline, s: float) -> np.ndarray:
        _, yaw = lane.sample(s)
        return np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)

    samples: List[LaneStartGoalSample] = []

    def _emit(
        start_lane: LanePolyline,
        goal_lane: LanePolyline,
        count: int,
        start_ref_s: float,
        shared_goal_s: float,
        start_slot_offset: int = 0,
    ) -> None:
        # ONE shared goal per lane: every car routed here targets the SAME
        # goal_xyz (computed once, no lateral jitter, so it is bit-identical).
        # This is safe because agents park/vanish on goal_reached, so staggered
        # arrivals clear the single spot one at a time; and it means only ONE
        # goal per lane must fit, so short-lane cars are no longer dropped for a
        # goal running off the lane end.  SPAWNS stay per-car staggered/unique.
        g_pt, g_yaw = goal_lane.sample(float(shared_goal_s))
        goal_xyz = (float(g_pt[0]), float(g_pt[1]), float(g_pt[2]))
        goal_in_bounds = _is_in_local_bounds(
            world_xy=np.array(goal_xyz[:2], dtype=np.float32),
            scene_center_xy=scene_center_xy,
            bounds_size_m=float(bounds_size_m),
            origin_mode=origin_mode,
        )
        for j in range(int(count)):
            i = int(start_slot_offset) + j
            start_s = (
                float(start_ref_s)
                - float(half_len)
                - float(approach_gap_m)
                - float(i) * float(spawn_spacing_m)
                - float(rng.uniform(0.0, spacing_jitter_m))
            )
            if start_s < float(min_upstream_margin_m):
                continue
            if start_s > start_lane.length_m - float(min_upstream_margin_m):
                continue
            if not goal_in_bounds:
                continue

            s_pt, s_yaw = start_lane.sample(start_s)
            lat = float(rng.uniform(-lateral_jitter_m, lateral_jitter_m))
            s_off = s_pt[:2] + lat * _normal_at(start_lane, start_s)
            start_xyz = (float(s_off[0]), float(s_off[1]), float(s_pt[2]))

            if not _is_in_local_bounds(
                world_xy=np.array(start_xyz[:2], dtype=np.float32),
                scene_center_xy=scene_center_xy,
                bounds_size_m=float(bounds_size_m),
                origin_mode=origin_mode,
            ):
                continue

            samples.append(
                LaneStartGoalSample(
                    sample_idx=len(samples),
                    polyline_idx=int(start_lane.polyline_idx),
                    road_type=int(start_lane.road_type),
                    travel_distance_m=float(shared_goal_s - start_s),
                    start_xyz=start_xyz,
                    start_yaw_rad=float(s_yaw),
                    goal_xyz=goal_xyz,
                    goal_yaw_rad=float(g_yaw),
                )
            )

    # One shared goal arc-length per lane, clamped to the lane so it never runs
    # off the end (`min_upstream_margin_m` doubles as the downstream end margin).
    # Returners use the larger `return_goal_clearance_m` so their goal clears the
    # exit taper (a design variable) plus merge-back room; open lanes use the
    # ordinary `goal_clearance_m`.
    end_margin = float(min_upstream_margin_m)
    closed_goal_s = min(
        float(box_s) + float(half_len) + float(return_goal_clearance_m),
        closed_lane.length_m - end_margin,
    )
    open_goal_s = [
        min(
            float(ref_s) + float(half_len) + float(goal_clearance_m),
            lane.length_m - end_margin,
        )
        for lane, ref_s in zip(open_lanes, open_refs)
    ]

    # Closed-lane convoy: split into returners (goal on closed lane) and, when
    # merge_return_frac < 1.0, mergers (goal projected onto the nearest open lane).
    n_closed = int(counts[0])
    frac = float(min(1.0, max(0.0, merge_return_frac)))
    n_return = int(round(n_closed * frac))
    n_merge = n_closed - n_return
    # Returners keep start+goal on the closed lane; mergers are placed on the
    # upstream (larger i) start slots so the two sub-convoys don't overlap, and
    # share the nearest open lane's goal.
    _emit(closed_lane, closed_lane, n_return, float(box_s), closed_goal_s)
    if n_merge > 0:
        _emit(
            closed_lane,
            open_lanes[0],
            n_merge,
            float(box_s),
            open_goal_s[0],
            start_slot_offset=n_return,
        )

    # Open-lane convoys: straight-through on their own lane.
    for lane, ref_s, goal_s, cnt in zip(open_lanes, open_refs, open_goal_s, counts[1:]):
        _emit(lane, lane, int(cnt), float(ref_s), goal_s)

    if not samples:
        raise RuntimeError("multilane merge OD produced no in-bounds spawns")
    return samples


def sample_workzone_start_goal_pairs(
    scene_cfg: Dict[str, Any],
    *,
    num_agents: int,
    bounds_size_m: float,
    origin_mode: str = "center",
    approach_gap_m: float = 8.0,
    spawn_spacing_m: float = 8.0,
    spacing_jitter_m: float = 1.5,
    lateral_jitter_m: float = 0.5,
    goal_clearance_m: float = 15.0,
    goal_spacing_m: float = 8.0,
    min_upstream_margin_m: float = 2.0,
    open_lane_count: int = 0,
    open_lane_offset_m: float = -3.5,
    open_lane_stagger_m: float = 4.0,
    merge_goal_lane_offset_m: float | None = None,
    seed: int = 42,
) -> List[LaneStartGoalSample]:
    """Place `num_agents` vehicles UPSTREAM of a workzone closure.

    The closure is `scene_cfg["workzone"]["forbidden_boxes"][0]`, which names the
    host lane (`host_polyline_id`) and the closure arc-length (`center_s_m`).

    Single-lane (open_lane_count == 0): all agents ride the host (closing) lane;
    starts descend upstream of the box at randomized spacing, goals ascend
    downstream, so convoy order is preserved and paths do not cross.

    Two-lane (open_lane_count > 0): the last `open_lane_count` agents ride the
    ADJACENT OPEN lane instead — the host centerline shifted laterally by
    `open_lane_offset_m` (toward the side the taper opens toward), staggered
    longitudinally by `open_lane_stagger_m` — and drive STRAIGHT through (goal
    keeps the lateral offset). The closing-lane convoy starts on the host lane
    and targets that same open lane downstream. By default,
    `merge_goal_lane_offset_m` uses `open_lane_offset_m`.

    Returns raw-scene-frame `LaneStartGoalSample`s, the same type as
    `sample_lane_center_start_goal_pairs`, so the env's OD machinery is unchanged.
    Fewer than `num_agents` may be returned if the lane runs out of room.
    """
    wz = scene_cfg.get("workzone", {}) or {}
    boxes = wz.get("forbidden_boxes", []) or []
    if not boxes:
        raise RuntimeError("scene has no workzone.forbidden_boxes for workzone OD")
    box = boxes[0]
    host_id = _safe_int(box.get("host_polyline_id", -1), -1)
    box_s = _safe_float(box.get("center_s_m", -1.0), -1.0)
    # Half length of the box BODY along the lane. The convoy must clear the box
    # body (plus taper) on the entry side and clear the body (plus exit) on the
    # goal side, so both offsets start from the box EDGE (box_s +/- half_len),
    # not the box center. Ignoring this put spawns inside the entry taper for
    # large-half_len boxes.
    box_half_len = max(0.0, _safe_float(box.get("half_len", 0.0), 0.0))
    if host_id < 0 or box_s < 0.0:
        raise RuntimeError("workzone box lacks host_polyline_id / center_s_m")

    lane = _build_host_lane_polyline(scene_cfg, host_id)
    scene_center = compute_scene_center_from_road(scene_cfg)
    scene_center_xy = scene_center[:2].astype(np.float32, copy=False)
    rng = np.random.default_rng(int(seed))

    def _normal_at(s: float) -> np.ndarray:
        # Left-hand unit normal from the lane tangent (yaw) at arc-length s.
        _, yaw = lane.sample(s)
        return np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)

    n_open = max(0, min(int(open_lane_count), int(num_agents)))
    n_host = int(num_agents) - n_open
    merge_goal_offset = (
        float(open_lane_offset_m)
        if merge_goal_lane_offset_m is None
        else float(merge_goal_lane_offset_m)
    )

    samples: List[LaneStartGoalSample] = []

    def _build_convoy(
        count: int,
        start_lane_offset_m: float,
        goal_lane_offset_m: float,
        s_shift_m: float,
    ) -> None:
        # A convoy of `count` cars: starts descend upstream from the entry side
        # of the box body, goals ascend downstream from the exit side.  The cars
        # start and finish at their respective offsets from the host centerline.
        # `approach_gap_m` / `goal_clearance_m` are clearances measured from the
        # box EDGE (box_s +/- box_half_len), NOT the box center, so the first
        # spawn clears both the box body and the entry taper.
        start_s = float(box_s) - float(box_half_len) - float(approach_gap_m) - float(s_shift_m)
        goal_s = float(box_s) + float(box_half_len) + float(goal_clearance_m) + float(s_shift_m)
        for i in range(int(count)):
            if i > 0:
                start_s -= float(spawn_spacing_m) + float(rng.uniform(0.0, spacing_jitter_m))
                goal_s += float(goal_spacing_m)
            if start_s < float(min_upstream_margin_m) or goal_s > lane.length_m - float(min_upstream_margin_m):
                break  # lane exhausted; place however many fit
            s_pt, s_yaw = lane.sample(start_s)
            g_pt, g_yaw = lane.sample(goal_s)
            # Small lateral jitter so a convoy is not perfectly single-file.
            lat = float(start_lane_offset_m) + float(rng.uniform(-lateral_jitter_m, lateral_jitter_m))
            s_off = s_pt[:2] + lat * _normal_at(start_s)
            g_off = g_pt[:2] + float(goal_lane_offset_m) * _normal_at(goal_s)
            start_xyz = (float(s_off[0]), float(s_off[1]), float(s_pt[2]))
            goal_xyz = (float(g_off[0]), float(g_off[1]), float(g_pt[2]))
            if not _is_in_local_bounds(
                world_xy=np.array(start_xyz[:2], dtype=np.float32),
                scene_center_xy=scene_center_xy,
                bounds_size_m=float(bounds_size_m),
                origin_mode=origin_mode,
            ):
                continue
            samples.append(
                LaneStartGoalSample(
                    sample_idx=len(samples),
                    polyline_idx=int(lane.polyline_idx),
                    road_type=int(lane.road_type),
                    travel_distance_m=float(goal_s - start_s),
                    start_xyz=start_xyz,
                    start_yaw_rad=float(s_yaw),
                    goal_xyz=goal_xyz,
                    goal_yaw_rad=float(g_yaw),
                )
            )

    # Closing-lane convoy: starts on the host lane and must finish in the open lane.
    _build_convoy(n_host, 0.0, merge_goal_offset, 0.0)
    # Open-lane convoy (drives straight): offset to the adjacent lane, staggered
    # half a spacing upstream so the closing cars have gaps to merge into.
    if n_open > 0:
        _build_convoy(
            n_open,
            float(open_lane_offset_m),
            float(open_lane_offset_m),
            float(open_lane_stagger_m),
        )

    if not samples:
        raise RuntimeError("workzone OD produced no in-bounds spawns on the host lane")
    return samples
