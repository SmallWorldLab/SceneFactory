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
        spawns.append(
            ScenarioVehicleSpawn(
                agent_id=agent_id,
                start_local_xyz=(float(start_local[0]), float(start_local[1]), float(start_local[2])),
                start_yaw_rad=float(syaw or 0.0),
                goal_local_xyz=(float(goal_local[0]), float(goal_local[1]), float(goal_local[2])),
                start_in_goal=bool(start_in_goal),
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

    _spawn_ground("/World/ground", _dry_ground_material_cfg(tunable_config), mode="plane")

    world_specs = prepare_stage_world_specs(cfg)
    _build_roads_only(stage=stage, cfg=cfg, json_paths=[spec.scene_json_path for spec in world_specs])
    manifest = _spawn_controllable_vehicles(stage=stage, cfg=cfg, world_specs=world_specs, tunable_config=tunable_config)

    stage.SetDefaultPrim(stage.GetPrimAtPath(Sdf.Path("/World")))
    sim.reset()

    if args_cli.save_stage_usd:
        output_dir.mkdir(parents=True, exist_ok=True)
        stage.Export(str((output_dir / "stage.usda").resolve()))

    return sim, manifest


def main() -> None:
    from isaaclab.app import AppLauncher

    args_cli = _build_parser().parse_args()
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    cfg = _load_yaml(args_cli.config)

    if int(args_cli.world_count) > 0:
        cfg.setdefault("world", {})["world_count"] = int(args_cli.world_count)
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

    step_budget = int(args_cli.sim_steps) if int(args_cli.sim_steps) >= 0 else int((cfg.get("sim", {}) or {}).get("steps", 0))
    step_index = 0
    while simulation_app.is_running():
        sim.step(render=not bool(getattr(args_cli, "headless", False)))
        step_index += 1
        if step_budget > 0 and step_index >= step_budget:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
