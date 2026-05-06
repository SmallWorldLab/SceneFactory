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
) -> List[LanePolyline]:
    road = scene_cfg.get("road", {}) or {}
    polylines = list(road.get("polylines", []) or [])
    lane_type_set = {int(x) for x in lane_types}

    out: List[LanePolyline] = []
    for idx, pl in enumerate(polylines):
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
