from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths
from src.trfc.lane_center_sampler import compute_scene_center_from_road

ensure_isaaclab_source_paths()

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp_cache")

from src.student_vehicle_sysid import StudentTunableConfig
from src.trfc import prepare_stage_world_specs


@dataclass(frozen=True)
class ScenarioVehicleSpawn:
    agent_id: int
    start_local_xyz: tuple[float, float, float]
    start_yaw_rad: float
    goal_local_xyz: tuple[float, float, float]
    start_in_goal: bool
    # Scene-JSON polyline id of the lane this agent's route lives on
    # (`agents.items[i].route.host_polyline_id`); -1 when the scene does not
    # declare one.  Workzone-approach scenes put start, goal and the forbidden
    # box all on this single lane.
    host_polyline_id: int = -1


def _build_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description=(
            "Build a clean SceneFactory Isaac Lab scene with multiple mini-worlds using the Chocolate "
            "road builder and spawn the current student vehicles on scenario start poses."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/scene_factory/multiworld_scene.yaml",
        help="YAML config for the multi-world scene.",
    )
    parser.add_argument(
        "--sim_steps",
        type=int,
        default=-1,
        help="Number of simulation steps to run. Negative uses the config value. Zero runs until the app closes.",
    )
    parser.add_argument(
        "--world_count",
        type=int,
        default=-1,
        help="Optional override for world.world_count.",
    )
    parser.add_argument(
        "--max_controllable_per_world",
        type=int,
        default=-1,
        help="Optional override for vehicles.max_controllable_per_world.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        default=False,
        help="Pause simulation after step 10 but keep the GUI alive for inspection/screenshots.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/scene_factory/multiworld_scene",
        help="Directory for the scene manifest and optional stage export.",
    )
    parser.add_argument(
        "--save_stage_usd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export the built stage to output_dir/stage.usda after construction.",
    )
    parser.add_argument(
        "--camera_path",
        type=str,
        default="",
        help=(
            "Path to a camera trajectory JSON file. When provided, an off-screen camera follows the "
            "trajectory and each frame is saved to --capture_dir. "
            "Format: list of {\"t\": <seconds>, \"eye\": [x,y,z], \"lookat\": [x,y,z]} keyframes. "
            "Positions are interpolated linearly between keyframes."
        ),
    )
    parser.add_argument(
        "--capture_dir",
        type=str,
        default="",
        help="Directory where captured frames (PNG) and optional video are saved. Defaults to output_dir/capture.",
    )
    parser.add_argument(
        "--capture_width",
        type=int,
        default=1920,
        help="Width of captured frames in pixels.",
    )
    parser.add_argument(
        "--capture_height",
        type=int,
        default=1080,
        help="Height of captured frames in pixels.",
    )
    parser.add_argument(
        "--capture_fps",
        type=int,
        default=30,
        help="Frames per second for the captured video. Controls how many sim steps map to 1 second of video.",
    )
    parser.add_argument(
        "--capture_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assemble captured frames into an MP4 at capture_dir/render.mp4 after the run.",
    )
    parser.add_argument(
        "--workzone_viz",
        action="store_true",
        default=False,
        help=(
            "Spawn workzone visualization overlays: a red semi-transparent ground slab "
            "representing the closed lane, and orange traffic cones arranged in a "
            "trapezoid taper pattern. Overlays are placed under <world_root>/WorkzoneVizOverlay/."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(payload).__name__}")
    return payload


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_scene_center_from_cfg(scene_cfg: Mapping[str, Any], *, center_mode: str = "mean") -> np.ndarray:
    center_mode = str(center_mode).strip().lower()
    if center_mode != "bbox":
        return compute_scene_center_from_road(dict(scene_cfg))

    road = scene_cfg.get("road", {}) or {}
    polylines = road.get("polylines", []) or []
    all_points: list[np.ndarray] = []
    for polyline in polylines:
        xyz = polyline.get("xyz")
        if not xyz:
            continue
        points = np.asarray(xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
            continue
        if points.shape[1] == 2:
            points = np.concatenate([points, np.zeros((points.shape[0], 1), dtype=np.float32)], axis=1)
        all_points.append(points[:, :3])
    if not all_points:
        return np.zeros((3,), dtype=np.float32)
    points = np.concatenate(all_points, axis=0)
    if center_mode == "bbox":
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        return 0.5 * (mins + maxs)
    return compute_scene_center_from_road(dict(scene_cfg))


def _in_bounds_xy(x: float, y: float, bounds_size_m: float) -> bool:
    half_extent = 0.5 * float(bounds_size_m)
    return (-half_extent <= float(x) <= half_extent) and (-half_extent <= float(y) <= half_extent)


def _load_scene_cfg(json_path: str | Path) -> dict[str, Any]:
    with Path(json_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Scene JSON root must be a mapping, got {type(payload).__name__}")
    return payload


def extract_vehicle_spawns_from_scene_cfg(
    scene_cfg: Mapping[str, Any],
    *,
    bounds_size_m: float,
    origin_mode: str,
    origin_center_mode: str,
    max_controllable: int,
    require_goal_in_bounds: bool,
    skip_if_start_in_goal: bool,
    goal_radius_m: float,
    start_goal_thresh_m: float | None,
) -> list[ScenarioVehicleSpawn]:
    scene_center = (
        _compute_scene_center_from_cfg(scene_cfg, center_mode=origin_center_mode)
        if str(origin_mode).strip().lower() == "center"
        else np.zeros((3,), dtype=np.float32)
    )
    agents = (scene_cfg.get("agents", {}) or {}).get("items", []) or []
    spawns: list[ScenarioVehicleSpawn] = []
    threshold = float(start_goal_thresh_m) if start_goal_thresh_m is not None else float(goal_radius_m)

    for fallback_idx, agent in enumerate(agents):
        if len(spawns) >= int(max_controllable):
            break

        start_cfg = agent.get("start", {}) or {}
        goal_cfg = agent.get("end", {}) or {}
        sx = _coerce_float(start_cfg.get("x"))
        sy = _coerce_float(start_cfg.get("y"))
        sz = _coerce_float(start_cfg.get("z"), 0.0)
        syaw = _coerce_float(start_cfg.get("yaw"), 0.0)
        gx = _coerce_float(goal_cfg.get("x"))
        gy = _coerce_float(goal_cfg.get("y"))
        gz = _coerce_float(goal_cfg.get("z"), 0.0)

        if sx is None or sy is None or gx is None or gy is None:
            continue

        start_local = np.asarray([sx, sy, float(sz or 0.0)], dtype=np.float32) - scene_center
        goal_local = np.asarray([gx, gy, float(gz or 0.0)], dtype=np.float32) - scene_center

        if not _in_bounds_xy(float(start_local[0]), float(start_local[1]), bounds_size_m):
            continue
        if require_goal_in_bounds and not _in_bounds_xy(float(goal_local[0]), float(goal_local[1]), bounds_size_m):
            continue

        distance_to_goal = float(np.linalg.norm(goal_local - start_local))
        start_in_goal = distance_to_goal <= float(threshold)
        if skip_if_start_in_goal and start_in_goal:
            continue

        agent_id = int(agent.get("agent_id", fallback_idx))
        route_cfg = agent.get("route", {}) or {}
        host_pid = route_cfg.get("host_polyline_id", -1)
        spawns.append(
            ScenarioVehicleSpawn(
                agent_id=agent_id,
                start_local_xyz=(float(start_local[0]), float(start_local[1]), float(start_local[2])),
                start_yaw_rad=float(syaw or 0.0),
                goal_local_xyz=(float(goal_local[0]), float(goal_local[1]), float(goal_local[2])),
                start_in_goal=bool(start_in_goal),
                host_polyline_id=int(host_pid) if host_pid is not None else -1,
            )
        )

    return spawns


def extract_vehicle_spawns_from_json(
    json_path: str | Path,
    *,
    bounds_size_m: float,
    origin_mode: str,
    origin_center_mode: str,
    max_controllable: int,
    require_goal_in_bounds: bool,
    skip_if_start_in_goal: bool,
    goal_radius_m: float,
    start_goal_thresh_m: float | None,
) -> list[ScenarioVehicleSpawn]:
    return extract_vehicle_spawns_from_scene_cfg(
        _load_scene_cfg(json_path),
        bounds_size_m=bounds_size_m,
        origin_mode=origin_mode,
        origin_center_mode=origin_center_mode,
        max_controllable=max_controllable,
        require_goal_in_bounds=require_goal_in_bounds,
        skip_if_start_in_goal=skip_if_start_in_goal,
        goal_radius_m=goal_radius_m,
        start_goal_thresh_m=start_goal_thresh_m,
    )


def _quat_wxyz_from_yaw(yaw_rad: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * float(yaw_rad)
    return (float(math.cos(half_yaw)), 0.0, 0.0, float(math.sin(half_yaw)))


def _world_root_path(root_container: str, world_index: int) -> str:
    return f"{str(root_container).rstrip('/')}/world_{int(world_index):03d}"


def _spawn_goal_marker(
    prim_path: str,
    *,
    goal_local_xyz: tuple[float, float, float],
    radius_m: float,
    height_m: float,
) -> None:
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    root = UsdGeom.Xform.Define(sim_utils.get_current_stage(), prim_path)
    pole_cfg = sim_utils.CylinderCfg(
        radius=max(0.18, 0.35 * float(radius_m)),
        height=max(2.0, 9.0 * float(height_m)),
        axis="Z",
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        rigid_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.48, 0.08),
            emissive_color=(0.35, 0.12, 0.02),
            roughness=0.18,
            metallic=0.0,
        ),
    )
    pole_height = max(2.0, 9.0 * float(height_m))
    pole_cfg.func(
        f"{prim_path}/Pole",
        pole_cfg,
        translation=(
            float(goal_local_xyz[0]),
            float(goal_local_xyz[1]),
            float(goal_local_xyz[2]) + 0.5 * pole_height,
        ),
    )
    cap_cfg = sim_utils.SphereCfg(
        radius=max(0.55, 1.2 * float(radius_m)),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        rigid_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.12, 1.0, 0.28),
            emissive_color=(0.05, 0.42, 0.10),
            roughness=0.10,
            metallic=0.0,
        ),
    )
    cap_cfg.func(
        f"{prim_path}/Cap",
        cap_cfg,
        translation=(
            float(goal_local_xyz[0]),
            float(goal_local_xyz[1]),
            float(goal_local_xyz[2]) + pole_height + max(0.35, 0.6 * float(radius_m)),
        ),
    )


def _spawn_workzone_viz_overlays(
    stage: Any,
    world_root: str = "/World/SceneFactoryWorlds/world_000",
    *,
    taper_factor: float = 1.5,
    obstacle_spacing: float = 3.0,
    label: str = "",
) -> "tuple[float, float, float, float] | None":
    """Spawn workzone visualization geometry that follows an actual lane polyline.

    Returns:
        (cx, cy, fwd_dx, fwd_dy) — world-space centre of the workzone and the
        forward lane direction at that point, for camera placement.  Returns None
        if the overlay could not be placed.

    Args:
        taper_factor: multiplier on lane_w for taper length  (e.g. 0.8=short, 3.0=long).
        obstacle_spacing: meters between consecutive cones; lower = denser.
        label: printed in the log line so designs can be identified.

    Strategy:
      1. Read all lane-center segments (type=1) from USD custom metadata.
      2. Detect contiguous polylines by checking endpoint continuity between segments.
      3. Pick a polyline long enough to fit ~80 m of workzone.
      4. For each segment in the chosen run, place one red slab tile oriented to
         that segment — so the overlay bends with the road.
      5. Place orange cones at sampled arc-length positions along the polyline edges:
           - Left taper wall  : every ~7 m along the closed-lane left edge, 0→50 m
           - Right taper edge : same arc positions but right offset tapers 0→lane_w
           - Buffer zone      : every 10 m along left edge, 55→95 m
    """
    import math
    import numpy as np
    from pxr import Gf, UsdGeom
    from src.student_vehicle_multiagent_goal_env import _load_scene_factory_lane_touch_metadata

    # ── 1. Read segments from USD, pick best available lane type ─────────────
    points_xy, dirs_xy, half_lengths_m, half_widths_m, types = _load_scene_factory_lane_touch_metadata(
        stage, world_root=world_root
    )
    if len(types) == 0:
        print("[WARN][WorkzoneViz] No road segments found in USD metadata — skipping overlay.", flush=True)
        return

    # Prefer lane centers (1), then left boundaries (2), then right boundaries (3),
    # then whatever is most numerous — so the overlay works on any scene.
    unique, counts = np.unique(types, return_counts=True)
    preferred = [t for t in (1, 2, 3) if t in unique]
    chosen_type = int(preferred[0]) if preferred else int(unique[np.argmax(counts)])
    lane_mask = types == chosen_type
    print(
        f"[INFO][WorkzoneViz] Available types: {dict(zip(unique.tolist(), counts.tolist()))}. "
        f"Using type {chosen_type} ({int(lane_mask.sum())} segments).",
        flush=True,
    )
    if lane_mask.sum() < 3:
        print("[WARN][WorkzoneViz] Too few segments for chosen type — skipping overlay.", flush=True)
        return

    lc_pts = points_xy[lane_mask].astype(float)     # (N, 2)
    lc_dirs = dirs_xy[lane_mask].astype(float)      # (N, 2)  normalized
    lc_hl = half_lengths_m[lane_mask].astype(float) # (N,)
    lc_hw = half_widths_m[lane_mask].astype(float)  # (N,)
    n = len(lc_pts)

    # ── 2. Group into contiguous polylines ───────────────────────────────────
    # Endpoint of seg i = midpoint + half_length * direction
    # Startpoint of seg i+1 = midpoint - half_length * direction
    # If they are close (< 0.5 m gap) they belong to the same polyline.
    endpoints = lc_pts + lc_hl[:, None] * lc_dirs          # (N, 2) forward endpoint
    startpoints = lc_pts - lc_hl[:, None] * lc_dirs        # (N, 2) backward startpoint

    polylines: list[list[int]] = []
    current: list[int] = [0]
    for i in range(1, n):
        gap = float(np.linalg.norm(endpoints[i - 1] - startpoints[i]))
        if gap < 1.0:
            current.append(i)
        else:
            polylines.append(current)
            current = [i]
    polylines.append(current)

    # ── 3. Choose a polyline with enough arc length ──────────────────────────
    target_len = 100.0  # need at least 100 m of lane
    def poly_arc_length(indices: list[int]) -> float:
        return float(sum(2.0 * lc_hl[i] for i in indices))

    long_polys = [p for p in polylines if poly_arc_length(p) >= target_len]
    if not long_polys:
        long_polys = sorted(polylines, key=poly_arc_length, reverse=True)[:1]

    # Pick the longest; start overlay ~30% along it so we're mid-lane
    chosen = max(long_polys, key=poly_arc_length)
    arc_len = poly_arc_length(chosen)
    skip_len = arc_len * 0.30
    cumul = 0.0
    start_i = 0
    for si, idx in enumerate(chosen):
        cumul += 2.0 * lc_hl[idx]
        if cumul >= skip_len:
            start_i = si
            break

    # Collect segments for the workzone run (up to 25 m arc — small workzone)
    run: list[int] = []
    run_len = 0.0
    for idx in chosen[start_i:]:
        seg_len = 2.0 * lc_hl[idx]
        if run_len + seg_len > 25.0:
            break
        run.append(idx)
        run_len += seg_len

    if not run:
        print("[WARN][WorkzoneViz] Could not find a valid lane run.", flush=True)
        return

    lane_w = float(np.median(lc_hw[run])) * 2.0  # full lane width from median half-width
    lane_w = max(2.5, lane_w)

    print(
        f"[INFO][WorkzoneViz] Using polyline with {len(run)} segments, "
        f"arc_len={run_len:.1f} m, lane_w={lane_w:.2f} m, "
        f"start=({lc_pts[run[0], 0]:.1f},{lc_pts[run[0], 1]:.1f})",
        flush=True,
    )

    # Place overlay under the world's own prim so each world gets independent geometry.
    overlay_root = f"{world_root}/WorkzoneVizOverlay"
    UsdGeom.Xform.Define(stage, overlay_root)

    red = Gf.Vec3f(1.0, 0.1, 0.1)
    orange = Gf.Vec3f(1.0, 0.45, 0.0)
    obstacle_height = 3.0   # ~10 ft — visible from aerial, proportional to lane width

    def _set_color(prim_obj: Any, color: Gf.Vec3f) -> None:
        try:
            UsdGeom.Gprim(prim_obj.GetPrim()).CreateDisplayColorAttr([color])
        except Exception:
            pass

    # ── 4. Red slab: one tile per segment, oriented to segment direction ──────
    for tile_i, seg_idx in enumerate(run):
        cx, cy = float(lc_pts[seg_idx, 0]), float(lc_pts[seg_idx, 1])
        dx, dy = float(lc_dirs[seg_idx, 0]), float(lc_dirs[seg_idx, 1])
        seg_len = 2.0 * float(lc_hl[seg_idx])
        yaw_deg = float(math.degrees(math.atan2(dy, dx)))
        rx, ry = -dy, dx   # right-hand normal
        tile_cx = cx + rx * (lane_w / 2)
        tile_cy = cy + ry * (lane_w / 2)
        tile_path = f"{overlay_root}/RedTile_{tile_i:03d}"
        cube = UsdGeom.Cube.Define(stage, tile_path)
        cube.GetSizeAttr().Set(1.0)
        api = UsdGeom.XformCommonAPI(cube)
        api.SetTranslate(Gf.Vec3d(tile_cx, tile_cy, 0.001))
        api.SetScale(Gf.Vec3f(seg_len + 0.05, lane_w, 0.002))
        api.SetRotate(Gf.Vec3f(0.0, 0.0, yaw_deg))
        _set_color(cube, red)

    # ── 5. Arc-length interpolation along the run ─────────────────────────────
    cumul_s: list[float] = [0.0]
    for seg_idx in run:
        cumul_s.append(cumul_s[-1] + 2.0 * float(lc_hl[seg_idx]))

    def sample_lane_edge(s: float, right_offset: float) -> tuple[float, float]:
        """World XY at arc distance s along the run, offset right_offset m to the right.
        Negative s extrapolates backwards from the first segment (taper buffer region)."""
        if s < 0.0:
            # Extrapolate backwards from the start of the run
            si = run[0]
            # startpoint of first segment = midpoint - half_length * dir
            px = lc_pts[si, 0] - lc_hl[si] * lc_dirs[si, 0] + s * lc_dirs[si, 0]
            py = lc_pts[si, 1] - lc_hl[si] * lc_dirs[si, 1] + s * lc_dirs[si, 1]
            return float(px - lc_dirs[si, 1] * right_offset), float(py + lc_dirs[si, 0] * right_offset)
        s = float(np.clip(s, 0.0, cumul_s[-1]))
        for k in range(len(run)):
            if s <= cumul_s[k + 1] + 1e-6:
                frac = (s - cumul_s[k]) / max(1e-6, cumul_s[k + 1] - cumul_s[k])
                si = run[k]
                px = lc_pts[si, 0] + (frac - 0.5) * 2.0 * lc_hl[si] * lc_dirs[si, 0]
                py = lc_pts[si, 1] + (frac - 0.5) * 2.0 * lc_hl[si] * lc_dirs[si, 1]
                return float(px - lc_dirs[si, 1] * right_offset), float(py + lc_dirs[si, 0] * right_offset)
        si = run[-1]
        return float(lc_pts[si, 0] - lc_dirs[si, 1] * right_offset), float(lc_pts[si, 1] + lc_dirs[si, 0] * right_offset)

    def _place_obstacle(path: str, wx: float, wy: float) -> None:
        try:
            prim = UsdGeom.Cone.Define(stage, path)
            prim.GetRadiusAttr().Set(0.6)
            prim.GetHeightAttr().Set(obstacle_height)
            prim.GetAxisAttr().Set("Z")
            UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(wx, wy, obstacle_height / 2))
            _set_color(prim, orange)
        except Exception:
            cube = UsdGeom.Cube.Define(stage, path)
            cube.GetSizeAttr().Set(1.0)
            xapi = UsdGeom.XformCommonAPI(cube)
            xapi.SetTranslate(Gf.Vec3d(wx, wy, obstacle_height / 2))
            xapi.SetScale(Gf.Vec3f(1.2, 1.2, obstacle_height))
            _set_color(cube, orange)

    # ── 6. Cones wrapping the workzone perimeter ──────────────────────────────
    # Three solid edges (left, right, back cap) + diagonal taper as fourth.
    # obstacle_spacing comes from the function parameter.
    obstacle_idx = 0

    # Left edge (right_offset=0): s from 0 to run_len
    s = 0.0
    while s <= run_len + 0.01:
        wx, wy = sample_lane_edge(s, 0.0)
        _place_obstacle(f"{overlay_root}/Cone_{obstacle_idx:02d}", wx, wy)
        obstacle_idx += 1
        s += obstacle_spacing

    # Right edge (right_offset=lane_w): s from 0 to run_len
    s = 0.0
    while s <= run_len + 0.01:
        wx, wy = sample_lane_edge(s, lane_w)
        _place_obstacle(f"{overlay_root}/Cone_{obstacle_idx:02d}", wx, wy)
        obstacle_idx += 1
        s += obstacle_spacing

    # Back cap (s=run_len): right_offset from obstacle_spacing to lane_w-obstacle_spacing
    r = obstacle_spacing
    while r < lane_w - 0.1:
        wx, wy = sample_lane_edge(run_len, r)
        _place_obstacle(f"{overlay_root}/Cone_{obstacle_idx:02d}", wx, wy)
        obstacle_idx += 1
        r += obstacle_spacing

    # ── 7. Merge taper buffer (upstream of workzone, s < 0) ──────────────────
    # The workzone + taper together form a TRUE TRAPEZOID (4 sides):
    #   Side A — left wall  : r=0 from s=-taper_len to s=run_len  (long, continuous)
    #   Side B — back cap   : s=run_len, r=0..lane_w              (short, perpendicular)
    #   Side C — right wall : r=lane_w, s=run_len..s=0            (medium, parallel to A)
    #   Side D — diagonal   : (s=0,r=lane_w) → (s=-taper_len,r=0) (angled taper face)
    # The front cap at s=0 is intentionally omitted — the diagonal replaces it.
    taper_len = max(15.0, lane_w * taper_factor)
    n_taper_steps = max(5, int(taper_len / obstacle_spacing))

    # Side A extension: left wall continuing upstream past workzone entry
    for i in range(1, n_taper_steps + 1):
        s = -obstacle_spacing * i
        wx, wy = sample_lane_edge(s, 0.0)
        _place_obstacle(f"{overlay_root}/Cone_{obstacle_idx:02d}", wx, wy)
        obstacle_idx += 1

    # Side D: diagonal from workzone right-front corner (s=0, r=lane_w)
    # converging to the left wall (s=-taper_len, r=0). i=0 places the corner
    # cone at (s=0, r=lane_w) which connects flush with the workzone right wall.
    for i in range(0, n_taper_steps + 1):
        t = i / n_taper_steps
        s = -obstacle_spacing * i
        right_off = lane_w * (1.0 - t)     # lane_w at s=0; 0 at s=-taper_len
        wx, wy = sample_lane_edge(s, right_off)
        _place_obstacle(f"{overlay_root}/Cone_{obstacle_idx:02d}", wx, wy)
        obstacle_idx += 1

    tag = f" [{label}]" if label else ""
    print(
        f"[INFO][WorkzoneViz]{tag} {world_root}: "
        f"{len(run)} red tiles ({run_len:.1f} m), {obstacle_idx} cones, "
        f"taper_factor={taper_factor:.1f}, obstacle_spacing={obstacle_spacing:.1f} m",
        flush=True,
    )

    # ── 8. Return workzone centre + forward direction for camera placement ────
    s_mid = run_len / 2.0
    cx_wz, cy_wz = sample_lane_edge(s_mid, lane_w / 2.0)

    # sample_lane_edge works in scene-local coordinates (road data is stored
    # relative to the scene centre, without the world grid offset).  The grid
    # offset lives on the world root prim's Xform, so we must add it here so
    # the caller can position a world-space camera correctly.
    try:
        _world_prim = stage.GetPrimAtPath(world_root)
        _txl = UsdGeom.Xformable(_world_prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
        cx_wz += float(_txl[0])
        cy_wz += float(_txl[1])
    except Exception:
        pass  # keep scene-local if transform is unavailable

    # Look up the segment nearest s_mid to get the lane direction there.
    fwd_dx, fwd_dy = float(lc_dirs[run[0], 0]), float(lc_dirs[run[0], 1])
    for k in range(len(run)):
        if s_mid <= cumul_s[k + 1] + 1e-6:
            fwd_dx = float(lc_dirs[run[k], 0])
            fwd_dy = float(lc_dirs[run[k], 1])
            break
    return float(cx_wz), float(cy_wz), fwd_dx, fwd_dy


def _build_roads_only(
    *,
    stage: Any,
    cfg: Mapping[str, Any],
    json_paths: Sequence[Path],
) -> Any:
    from src.chocolate_waymo_builder import ChocolateBarConstructor, GridLayout

    world_cfg = dict(cfg.get("world", {}) or {})
    road_cfg = dict(cfg.get("road", {}) or {})
    layout = GridLayout(
        world_size_m=tuple(map(float, world_cfg.get("world_size_m", (200.0, 200.0)))),
        padding_m=float(world_cfg.get("padding_m", 20.0)),
        grid_cols=int(world_cfg.get("grid_cols", 2)),
        base_z_m=float(world_cfg.get("base_z_m", 0.0)),
    )
    constructor = ChocolateBarConstructor(
        stage=stage,
        root_container=str(world_cfg.get("root_container", "/World/SceneFactoryWorlds")),
        layout=layout,
        origin_mode=str(world_cfg.get("origin_mode", "center")),
        origin_center_mode=str(world_cfg.get("origin_center_mode", "mean")),
    )
    constructor.clear_all()
    constructor.build(
        json_paths=json_paths,
        world_count=int(world_cfg["world_count"]),
        bounds_size_m=float(world_cfg.get("bounds_size_m", layout.world_size_m[0])),
        max_agents_per_world=0,
        jump_break_m=float(road_cfg.get("jump_break_m", 3.0)),
        seg_width=float(road_cfg.get("seg_width", 0.10)),
        seg_height=float(road_cfg.get("seg_height", 0.10)),
        z_lift=float(road_cfg.get("z_lift", 0.02)),
        flatten_road_z=bool(road_cfg.get("flatten_road_z", road_cfg.get("flatten_road", True))),
        road_z_m=float(road_cfg.get("road_z_m", 0.0)),
        polyline_reduction_area=float(road_cfg.get("polyline_reduction_area", 0.0)),
        min_points_for_reduction=int(road_cfg.get("min_points_for_reduction", 10)),
        enable_segment_collision=bool(road_cfg.get("enable_segment_collision", False)),
        trigger_enable=bool(road_cfg.get("trigger_enable", False)),
        trigger_height_m=float(road_cfg.get("trigger_height_m", 1.0)),
        trigger_width_scale=float(road_cfg.get("trigger_width_scale", 1.0)),
        trigger_offset_z_m=float(road_cfg.get("trigger_offset_z_m", 0.5)),
        trigger_match_segment=bool(road_cfg.get("trigger_match_segment", True)),
        trigger_script_enable=bool(road_cfg.get("trigger_script_enable", True)),
        allowed_road_types=road_cfg.get("allowed_types"),
        road_render_mode=str(road_cfg.get("render_mode", "point_instancer")),
        spawn_z_m=1.0,
        goal_radius_m=3.0,
        parked_if_start_in_goal=False,
        skip_if_start_in_goal=True,
    )
    return constructor


def _build_single_world_roads_only(
    *,
    stage: Any,
    cfg: Mapping[str, Any],
    json_path: str | Path,
    world_root: str,
    video_obstacle_markers: bool = False,
) -> None:
    from src.chocolate_waymo_builder import LocalBounds, WaymoJsonMiniWorldBuilder

    world_cfg = dict(cfg.get("world", {}) or {})
    road_cfg = dict(cfg.get("road", {}) or {})

    stage.RemovePrim(str(world_root))
    builder = WaymoJsonMiniWorldBuilder(
        stage=stage,
        world_root=str(world_root),
        bounds=LocalBounds(
            width_m=float(world_cfg.get("bounds_size_m", world_cfg.get("world_size_m", [200.0, 200.0])[0])),
            length_m=float(world_cfg.get("bounds_size_m", world_cfg.get("world_size_m", [200.0, 200.0])[1])),
            origin_xy=(0.0, 0.0),
        ),
        origin_mode=str(world_cfg.get("origin_mode", "center")),
        origin_center_mode=str(world_cfg.get("origin_center_mode", "mean")),
    )
    builder.build_from_json(
        str(Path(json_path).expanduser().resolve()),
        max_agents=0,
        polyline_reduction_area=float(road_cfg.get("polyline_reduction_area", 0.0)),
        min_points_for_reduction=int(road_cfg.get("min_points_for_reduction", 10)),
        jump_break_m=float(road_cfg.get("jump_break_m", 3.0)),
        seg_width=float(road_cfg.get("seg_width", 0.10)),
        seg_height=float(road_cfg.get("seg_height", 0.10)),
        z_lift=float(road_cfg.get("z_lift", 0.02)),
        flatten_road_z=bool(road_cfg.get("flatten_road_z", road_cfg.get("flatten_road", True))),
        road_z_m=float(road_cfg.get("road_z_m", 0.0)),
        enable_segment_collision=bool(road_cfg.get("enable_segment_collision", False)),
        trigger_enable=bool(road_cfg.get("trigger_enable", False)),
        trigger_height_m=float(road_cfg.get("trigger_height_m", 1.0)),
        trigger_width_scale=float(road_cfg.get("trigger_width_scale", 1.0)),
        trigger_offset_z_m=float(road_cfg.get("trigger_offset_z_m", 0.5)),
        trigger_match_segment=bool(road_cfg.get("trigger_match_segment", True)),
        trigger_script_enable=bool(road_cfg.get("trigger_script_enable", True)),
        allowed_road_types=road_cfg.get("allowed_types"),
        road_render_mode=str(road_cfg.get("render_mode", "point_instancer")),
        spawn_z_m=1.0,
        goal_radius_m=3.0,
        parked_if_start_in_goal=False,
        skip_if_start_in_goal=True,
        video_obstacle_markers=video_obstacle_markers,
    )


def _apply_view(stage: Any, cfg: Mapping[str, Any], sim: Any) -> None:
    import isaaclab.sim as sim_utils

    viewer_cfg = dict(cfg.get("viewer", {}) or {})
    eye = tuple(map(float, viewer_cfg.get("eye", (260.0, 160.0, 200.0))))
    lookat = tuple(map(float, viewer_cfg.get("lookat", (120.0, 120.0, 0.0))))
    sim.set_camera_view(eye=eye, target=lookat)

    light_cfg = sim_utils.DomeLightCfg(
        intensity=float(viewer_cfg.get("light_intensity", 3000.0)),
        color=tuple(map(float, viewer_cfg.get("light_color", (0.78, 0.78, 0.78)))),
    )
    light_cfg.func(str(viewer_cfg.get("light_path", "/World/Light")), light_cfg)


def _spawn_controllable_vehicles(
    *,
    stage: Any,
    cfg: Mapping[str, Any],
    world_specs: Sequence[Any],
    tunable_config: StudentTunableConfig,
) -> dict[str, Any]:
    from pxr import UsdGeom

    from src.student_vehicle_goal_env import DEFAULT_STUDENT_VEHICLE_USD, build_student_vehicle_articulation_cfg
    from src.student_vehicle_sysid import _apply_runtime_student_dynamics

    world_cfg = dict(cfg.get("world", {}) or {})
    vehicles_cfg = dict(cfg.get("vehicles", {}) or {})
    root_container = str(world_cfg.get("root_container", "/World/SceneFactoryWorlds"))
    bounds_size_m = float(world_cfg.get("bounds_size_m", 200.0))
    origin_mode = str(world_cfg.get("origin_mode", "center"))
    origin_center_mode = str(world_cfg.get("origin_center_mode", "mean"))

    student_usd_path = str(
        Path(vehicles_cfg.get("student_usd", DEFAULT_STUDENT_VEHICLE_USD)).expanduser().resolve()
    )
    spawn_height_m = float(vehicles_cfg.get("spawn_height_m", 1.2))
    max_controllable = int(vehicles_cfg.get("max_controllable_per_world", 8))
    require_goal_in_bounds = bool(vehicles_cfg.get("require_goal_in_bounds", True))
    skip_if_start_in_goal = bool(vehicles_cfg.get("skip_if_start_in_goal", True))
    goal_radius_m = float(vehicles_cfg.get("goal_radius_m", 3.0))
    start_goal_thresh_m = _coerce_float(vehicles_cfg.get("start_goal_thresh_m"), None)
    goal_marker_enable = bool((cfg.get("goal_markers", {}) or {}).get("enable", True))
    goal_marker_radius_m = float((cfg.get("goal_markers", {}) or {}).get("radius_m", 1.0))
    goal_marker_height_m = float((cfg.get("goal_markers", {}) or {}).get("height_m", 0.12))

    vehicle_spawn_cfg = build_student_vehicle_articulation_cfg(
        student_usd_path,
        spawn_height_m=spawn_height_m,
        prim_path="/World/__unused_vehicle_path",
        max_depenetration_velocity=1.0,  # defence-in-depth: redundant once bodies are kinematic, but harmless
    ).spawn

    manifest: dict[str, Any] = {"worlds": []}

    for world_spec in world_specs:
        world_index = int(world_spec.world_index)
        world_root = _world_root_path(root_container, world_index)
        world_vehicles_root = f"{world_root}/ControllableVehicles"
        world_goals_root = f"{world_root}/ControllableGoals"
        UsdGeom.Xform.Define(stage, world_vehicles_root)
        UsdGeom.Xform.Define(stage, world_goals_root)

        spawns = extract_vehicle_spawns_from_json(
            world_spec.scene_json_path,
            bounds_size_m=bounds_size_m,
            origin_mode=origin_mode,
            origin_center_mode=origin_center_mode,
            max_controllable=max_controllable,
            require_goal_in_bounds=require_goal_in_bounds,
            skip_if_start_in_goal=skip_if_start_in_goal,
            goal_radius_m=goal_radius_m,
            start_goal_thresh_m=start_goal_thresh_m,
        )

        world_record = {
            "world_index": world_index,
            "scene_json_path": str(world_spec.scene_json_path),
            "world_root": world_root,
            "spawned_vehicle_count": len(spawns),
            "vehicles": [],
        }

        for vehicle_index, spawn in enumerate(spawns):
            vehicle_root = f"{world_vehicles_root}/Vehicle_{vehicle_index:03d}_id{int(spawn.agent_id)}"
            vehicle_spawn_cfg.func(
                vehicle_root,
                vehicle_spawn_cfg,
                translation=(
                    float(spawn.start_local_xyz[0]),
                    float(spawn.start_local_xyz[1]),
                    float(spawn_height_m),
                ),
                orientation=_quat_wxyz_from_yaw(spawn.start_yaw_rad),
            )
            _apply_runtime_student_dynamics(stage=stage, student_root_path=vehicle_root, config=tunable_config)

            vehicle_prim = stage.GetPrimAtPath(vehicle_root)
            if vehicle_prim.IsValid():
                vehicle_prim.SetCustomDataByKey("world_index", world_index)
                vehicle_prim.SetCustomDataByKey("agent_id", int(spawn.agent_id))
                vehicle_prim.SetCustomDataByKey("goal_local_m", tuple(map(float, spawn.goal_local_xyz)))
                vehicle_prim.SetCustomDataByKey("spawn_local_m", tuple(map(float, spawn.start_local_xyz)))

            if goal_marker_enable:
                _spawn_goal_marker(
                    f"{world_goals_root}/Goal_{vehicle_index:03d}_id{int(spawn.agent_id)}",
                    goal_local_xyz=spawn.goal_local_xyz,
                    radius_m=goal_marker_radius_m,
                    height_m=goal_marker_height_m,
                )

            world_record["vehicles"].append(
                {
                    "vehicle_root": vehicle_root,
                    "agent_id": int(spawn.agent_id),
                    "spawn_local_xyz": [float(v) for v in spawn.start_local_xyz],
                    "spawn_yaw_rad": float(spawn.start_yaw_rad),
                    "goal_local_xyz": [float(v) for v in spawn.goal_local_xyz],
                }
            )

        manifest["worlds"].append(world_record)

    return manifest


def _write_manifest(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scene_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _make_scene_vehicles_kinematic(stage: Any, root_container: str) -> None:
    """Anchor every vehicle articulation root to the world so vehicles act as static display props.

    Root cause of the wheel-wiggle bug
    -----------------------------------
    Setting ``physics:kinematicEnabled=True`` on the *RigidBodyAPI* of an articulation root
    link does NOT freeze the articulation in PhysX 5.  PhysX distinguishes between:

    * **Standalone rigid bodies** — ``kinematicEnabled`` works correctly.
    * **Articulation root links** — the kinematic flag on the RigidBodyAPI is ignored; the
      articulation is still fully simulated.  The correct API to freeze an articulation root
      is to create a ``PhysicsFixedJoint`` between the world frame and the root link (with the
      ArticulationRootAPI moved to the parent Xform).  Isaac Lab exposes this as
      ``ArticulationRootPropertiesCfg(fix_root_link=True)``.

    Previous behaviour: the chassis appeared static for a moment but the child links (wheels,
    suspension links) were still being simulated by the articulation solver.  Because no
    kinematic constraint actually held the chassis, gravity began accelerating it downward.
    The articulation solver then drove the joints to compensate, producing the
    "wheels fly outward / wiggle" artefact.

    Fix
    ---
    For each articulation found under *root_container* we call
    ``modify_articulation_root_properties`` with ``fix_root_link=True``.  Internally that
    function:

    1. Creates a ``PhysicsFixedJoint`` from the world frame to ``base_link``.
    2. Moves the ``ArticulationRootAPI`` / ``PhysxArticulationAPI`` from ``base_link`` to its
       parent Xform (required by the PhysX parser to recognise a fixed-base articulation).
    3. Removes the APIs from ``base_link`` so there is no duplicate.

    The vehicle is now a *fixed-base articulation*: the root is world-anchored and the
    internal joint DOFs (suspension travel, wheel spin, steer angle) are still solved by the
    articulation solver.  The display USD's baked drives (stiffness=2000, damping=200,
    targetPosition=0, type=acceleration) hold all DOFs at the neutral pose.

    Training is not affected — training uses ``student_fwd_vehicle.usd`` (not the display USD)
    and never calls this function.
    """
    from pxr import Usd, UsdPhysics

    import isaaclab.sim as sim_utils
    from isaaclab.sim.schemas import modify_articulation_root_properties

    root_prim = stage.GetPrimAtPath(root_container)
    if not root_prim.IsValid():
        return

    # Collect all articulation root prims first, then fix them.
    # We must not modify the prim tree while iterating over it.
    artic_prims: list[Any] = []
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            artic_prims.append(prim)

    fixed_count = 0
    for prim in artic_prims:
        prim_path = str(prim.GetPath())
        # fix_root_link=True creates a PhysicsFixedJoint from the world to this link
        # and moves the ArticulationRootAPI to the parent Xform — the correct way to
        # anchor an articulation root in PhysX 5.
        try:
            modify_articulation_root_properties(
                prim_path,
                sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True),
                stage=stage,
            )
            fixed_count += 1
        except Exception as exc:  # pragma: no cover — only fires on malformed USD
            print(f"[visualizer] Warning: could not fix articulation root at {prim_path}: {exc}")

    if fixed_count:
        print(f"[visualizer] Fixed {fixed_count} articulation root(s) to world frame under {root_container}")


def _build_scene(args_cli: argparse.Namespace, cfg: dict[str, Any], *, output_dir: Path) -> tuple[Any, dict[str, Any]]:
    import omni.usd
    from pxr import Sdf

    import torch
    from isaaclab.sim import SimulationCfg, SimulationContext

    from src.student_vehicle_goal_env import _dry_ground_material_cfg, _spawn_ground
    from src.student_vehicle_sysid import load_tunable_config

    sim_cfg = dict(cfg.get("sim", {}) or {})
    requested_device = str(args_cli.device or sim_cfg.get("device") or "").strip()
    if not requested_device:
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested_device.lower().startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    sim_device = requested_device
    sim = SimulationContext(
        SimulationCfg(
            dt=float(sim_cfg.get("dt", 1.0 / 120.0)),
            device=sim_device,
        )
    )

    stage = omni.usd.get_context().get_stage()
    _apply_view(stage, cfg, sim)

    tunable_config_json = str((cfg.get("vehicles", {}) or {}).get("tunable_config_json", "")).strip()
    tunable_config = (
        load_tunable_config(tunable_config_json) if tunable_config_json else StudentTunableConfig()
    )

    # Standalone visualizer — single non-cloned stage, shared ground is acceptable here.
    # Multi-env training uses per-env ground inside /World/envs/env_N/Ground instead.
    _spawn_ground("/World/ground", _dry_ground_material_cfg(tunable_config), mode="cuboid")

    world_specs = prepare_stage_world_specs(cfg)
    _build_roads_only(stage=stage, cfg=cfg, json_paths=[spec.scene_json_path for spec in world_specs])
    manifest = _spawn_controllable_vehicles(stage=stage, cfg=cfg, world_specs=world_specs, tunable_config=tunable_config)

    # Make every vehicle rigid body kinematic — vehicles are display props in the
    # visualizer, not driven agents. This prevents PhysX from launching them at
    # spawn due to depenetration impulses or gravity accumulation.
    _make_scene_vehicles_kinematic(stage, root_container=str(
        (cfg.get("world", {}) or {}).get("root_container", "/World/SceneFactoryWorlds")
    ))

    stage.SetDefaultPrim(stage.GetPrimAtPath(Sdf.Path("/World")))
    sim.reset()

    if args_cli.save_stage_usd:
        output_dir.mkdir(parents=True, exist_ok=True)
        stage.Export(str((output_dir / "stage.usda").resolve()))

    return sim, manifest


@dataclass
class _Keyframe:
    t: float
    eye: tuple[float, float, float]
    lookat: tuple[float, float, float]


def _load_camera_trajectory(path: str) -> list[_Keyframe]:
    """Load and validate a camera trajectory JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"camera_path JSON must be a non-empty list of keyframes, got: {type(raw)}")
    frames: list[_Keyframe] = []
    for entry in raw:
        frames.append(_Keyframe(
            t=float(entry["t"]),
            eye=tuple(float(v) for v in entry["eye"]),
            lookat=tuple(float(v) for v in entry["lookat"]),
        ))
    frames.sort(key=lambda k: k.t)
    return frames


def _interpolate_trajectory(frames: list[_Keyframe], t: float) -> tuple[tuple, tuple]:
    """Return (eye, lookat) linearly interpolated at time t."""
    if t <= frames[0].t:
        return frames[0].eye, frames[0].lookat
    if t >= frames[-1].t:
        return frames[-1].eye, frames[-1].lookat
    for i in range(len(frames) - 1):
        a, b = frames[i], frames[i + 1]
        if a.t <= t <= b.t:
            alpha = (t - a.t) / (b.t - a.t)
            eye = tuple(a.eye[j] + alpha * (b.eye[j] - a.eye[j]) for j in range(3))
            lookat = tuple(a.lookat[j] + alpha * (b.lookat[j] - a.lookat[j]) for j in range(3))
            return eye, lookat
    return frames[-1].eye, frames[-1].lookat


def _spawn_visualizer_camera(width: int, height: int) -> Any:
    """Create an off-screen Isaac Lab Camera at a fixed USD path."""
    from isaaclab.sensors import Camera, CameraCfg
    import isaaclab.sim as sim_utils

    cam_cfg = CameraCfg(
        prim_path="/World/VisualizerCaptureCamera",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        colorize_instance_id_segmentation=False,
        colorize_instance_segmentation=False,
        colorize_semantic_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e6),
        ),
    )
    cam = Camera(cam_cfg)
    cam.reset()
    return cam


def _camera_look_at_xform(eye: tuple, lookat: tuple, up_hint: tuple = (0, 0, 1)) -> Any:
    """Place /World/VisualizerCaptureCamera at eye looking at lookat.

    up_hint controls the camera roll.  Pass (0, 0, -1) to rotate 180° around
    the view axis (flip the image upside-down / clockwise 180°).
    """
    from pxr import Gf, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/VisualizerCaptureCamera")
    if not cam_prim.IsValid():
        return

    eye_v = Gf.Vec3d(*eye)
    lookat_v = Gf.Vec3d(*lookat)
    up = Gf.Vec3d(*up_hint)
    fwd = (lookat_v - eye_v).GetNormalized()
    if abs(Gf.Dot(fwd, up)) > 0.999:
        up = Gf.Vec3d(0, 1, 0)
    right = Gf.Cross(fwd, up).GetNormalized()
    up = Gf.Cross(right, fwd).GetNormalized()

    # USD camera looks along -Z, so we negate fwd
    mat = Gf.Matrix4d(
        right[0],  right[1],  right[2],  0,
        up[0],     up[1],     up[2],     0,
        -fwd[0],  -fwd[1],  -fwd[2],    0,
        eye[0],    eye[1],    eye[2],    1,
    )
    xform = UsdGeom.Xformable(cam_prim)
    xform.ClearXformOpOrder()
    op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform)
    op.Set(mat)


def _capture_workzone_photos(
    sim: Any,
    simulation_app: Any,
    centers: "list[tuple[float, float, float, float] | None]",
    labels: "list[str]",
    save_dir: "Path",
    *,
    cam_z: float = 255.0,
    back_dist: float = 0.0,
    cam_x_offset: float = -72.0,
    render_warmup: int = 8,
    sim_dt: float = 1.0 / 120.0,
) -> None:
    """Capture one overhead photo per workzone using the viewport renderer.

    Avoids the Isaac Lab Camera sensor (which requires --enable_cameras).
    Instead: creates a plain USD Camera prim, sets it as the active viewport
    camera, positions it with the existing look-at helper, warms up a few
    render steps, then calls omni.renderer_capture to save the swapchain frame.

    Works in GUI (non-headless) mode.  In headless mode the swapchain does not
    exist; photos are silently skipped with a warning.
    """
    from pxr import UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create a plain USD Camera (no Isaac Lab sensor needed).
    cam_path = "/World/VisualizerCaptureCamera"
    if not stage.GetPrimAtPath(cam_path).IsValid():
        UsdGeom.Camera.Define(stage, cam_path)

    # Point the active viewport at this camera.
    try:
        import omni.kit.viewport.utility as _vpu
        _viewport = _vpu.get_active_viewport()
        _viewport.camera_path = cam_path
    except Exception as _e:
        print(f"[WorkzoneViz] Could not set viewport camera: {_e}", flush=True)

    # Renderer-capture interface — writes the next rendered swapchain frame to disk.
    try:
        import omni.renderer_capture as _rc
        _capture_iface = _rc.acquire_renderer_capture_interface()
    except Exception as _e:
        print(f"[WorkzoneViz] omni.renderer_capture unavailable ({_e}) — skipping photos.", flush=True)
        return

    for wi, center in enumerate(centers):
        if center is None:
            print(f"[WorkzoneViz] World {wi}: no workzone centre — skipping photo.", flush=True)
            continue
        cx, cy, fdx, fdy = center

        # Eye: back_dist m behind the workzone along the lane, cam_z m above ground.
        eye = (cx - back_dist * fdx + cam_x_offset, cy - back_dist * fdy, cam_z)
        lookat = (cx, cy, 1.0)
        # Use the lane forward direction as the camera up-hint so the road runs
        # bottom-to-top in the image regardless of world orientation.
        # This also avoids gimbal lock when the camera is straight overhead.
        _camera_look_at_xform(eye, lookat, up_hint=(fdx, fdy, 0.0))

        # Warm up: let the renderer settle on the new camera position.
        for _ in range(render_warmup):
            sim.step(render=True)

        # Schedule capture of the very next rendered frame, then render it.
        # sim.step(render=True) is needed — simulation_app.update() alone does
        # not issue a render pass and the swapchain capture never fires.
        slug = labels[wi].strip().replace(" ", "_").replace("/", "-")
        out_path = save_dir / f"world_{wi:02d}_{slug}.png"
        _capture_iface.capture_next_frame_swapchain(str(out_path))
        sim.step(render=True)

        print(f"[WorkzoneViz] Photo → {out_path}", flush=True)


def main() -> None:
    from isaaclab.app import AppLauncher

    args_cli = _build_parser().parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    cfg = _load_yaml(args_cli.config)

    _zone_viz = bool(getattr(args_cli, "workzone_viz", False))
    if _zone_viz and int(args_cli.world_count) <= 0:
        # Default to 4 worlds for the 2×2 taper-design comparison.
        args_cli.world_count = 4
    if int(args_cli.world_count) > 0:
        cfg.setdefault("world", {})["world_count"] = int(args_cli.world_count)
        cfg["world"]["grid_cols"] = max(1, math.isqrt(int(args_cli.world_count)))
        if _zone_viz:
            # All worlds share one scene so the taper designs are directly comparable.
            cfg.setdefault("io", {})["take_first_k_scenes"] = 1
        else:
            # Normal mode: each world gets its own unique scene.
            cfg.setdefault("io", {})["take_first_k_scenes"] = int(args_cli.world_count)
    if int(args_cli.max_controllable_per_world) >= 0:
        cfg.setdefault("vehicles", {})["max_controllable_per_world"] = int(args_cli.max_controllable_per_world)

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    sim, manifest = _build_scene(args_cli, cfg, output_dir=output_dir)
    _write_manifest(
        output_dir,
        {
            "config_path": str(Path(args_cli.config).expanduser().resolve()),
            "world_count": int((cfg.get("world", {}) or {}).get("world_count", 0)),
            "manifest": manifest,
        },
    )

    if _zone_viz:
        import omni.usd as _omni_usd
        _wz_root_container = str((cfg.get("world", {}) or {}).get("root_container", "/World/SceneFactoryWorlds"))
        _wz_stage = _omni_usd.get_context().get_stage()
        # Four taper designs for direct comparison (2×2 grid of worlds, same map).
        # Axis 1: taper length (taper_factor × lane_w)  — slope of the merge zone
        # Axis 2: cone density (obstacle_spacing in metres)  — visual clarity / material cost
        _taper_designs = [
            dict(taper_factor=0.8, obstacle_spacing=2.0, label="short-steep  dense"),
            dict(taper_factor=1.5, obstacle_spacing=3.0, label="standard     dense"),
            dict(taper_factor=0.8, obstacle_spacing=6.0, label="short-steep  sparse"),
            dict(taper_factor=1.5, obstacle_spacing=6.0, label="standard     sparse"),
        ]
        _wz_centers: list = []
        for _wi, _design in enumerate(_taper_designs):
            _wz_world_root = _world_root_path(_wz_root_container, _wi)
            _center = _spawn_workzone_viz_overlays(_wz_stage, world_root=_wz_world_root, **_design)
            _wz_centers.append(_center)

        # Capture one overhead photo per workzone before entering the interactive loop.
        _capture_workzone_photos(
            sim=sim,
            simulation_app=simulation_app,
            centers=_wz_centers,
            labels=[d["label"] for d in _taper_designs],
            save_dir=output_dir / "workzone_photos",
            sim_dt=float((cfg.get("sim", {}) or {}).get("dt", 1.0 / 120.0)),
        )

    # ── Camera trajectory capture setup ────────────────────────────────────
    camera_path_str = str(getattr(args_cli, "camera_path", "") or "").strip()
    traj_frames: list[_Keyframe] = []
    capture_cam = None
    capture_dir: Path | None = None
    sim_dt = float((cfg.get("sim", {}) or {}).get("dt", 1.0 / 120.0))
    capture_fps = max(1, int(getattr(args_cli, "capture_fps", 30)))
    # How many sim steps between saved frames (round to nearest int, min 1)
    steps_per_frame = max(1, round(1.0 / (capture_fps * sim_dt)))

    if camera_path_str:
        traj_frames = _load_camera_trajectory(camera_path_str)
        capture_dir = Path(str(getattr(args_cli, "capture_dir", "") or "")).expanduser().resolve()
        if not capture_dir.name:
            capture_dir = output_dir / "capture"
        capture_dir.mkdir(parents=True, exist_ok=True)
        capture_cam = _spawn_visualizer_camera(
            width=int(getattr(args_cli, "capture_width", 1920)),
            height=int(getattr(args_cli, "capture_height", 1080)),
        )
        total_duration = traj_frames[-1].t
        total_steps = int(math.ceil(total_duration / sim_dt))
        print(
            f"[Visualizer] Camera trajectory loaded: {len(traj_frames)} keyframes, "
            f"{total_duration:.1f}s, capturing {capture_fps} fps → {capture_dir}"
        )
        # Override step budget to cover the full trajectory
        args_cli.sim_steps = total_steps + 30  # +30 warm-up steps
    # ───────────────────────────────────────────────────────────────────────

    sim_cfg_steps = int((cfg.get("sim", {}) or {}).get("steps", 0))
    step_budget = int(args_cli.sim_steps) if int(args_cli.sim_steps) >= 0 else sim_cfg_steps
    freeze = bool(getattr(args_cli, "freeze", False))
    freeze_at = 10
    step_index = 0
    frame_index = 0
    headless = bool(getattr(args_cli, "headless", False))
    captured_frames: list[np.ndarray] = []

    while simulation_app.is_running():
        if freeze and step_index >= freeze_at:
            simulation_app.update()
            continue

        # Advance simulation
        sim.step(render=not headless)
        step_index += 1

        # Camera trajectory capture
        if capture_cam is not None and traj_frames and step_index % steps_per_frame == 0:
            current_t = step_index * sim_dt
            eye, lookat = _interpolate_trajectory(traj_frames, current_t)
            _camera_look_at_xform(eye, lookat)
            capture_cam.update(sim_dt * steps_per_frame)
            rgb = capture_cam.data.output.get("rgb")
            if rgb is not None and rgb.numel() > 0:
                frame_arr = rgb[0].detach().cpu().numpy()
                png_path = capture_dir / f"frame_{frame_index:06d}.png"
                import imageio.v2 as imageio
                imageio.imwrite(str(png_path), frame_arr)
                captured_frames.append(frame_arr)
                frame_index += 1
                if frame_index % 30 == 0:
                    print(f"[Visualizer] Captured frame {frame_index} (t={current_t:.2f}s) → {png_path.name}")

        if step_budget > 0 and step_index >= step_budget:
            break

    # Assemble video if requested
    if capture_cam is not None and captured_frames and bool(getattr(args_cli, "capture_video", True)):
        import imageio.v2 as imageio
        video_path = capture_dir / "render.mp4"
        print(f"[Visualizer] Writing {len(captured_frames)} frames to {video_path} at {capture_fps} fps …")
        with imageio.get_writer(str(video_path), fps=capture_fps) as writer:
            for frame in captured_frames:
                writer.append_data(frame)
        print(f"[Visualizer] Video saved: {video_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
