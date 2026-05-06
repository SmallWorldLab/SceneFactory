from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths
from src.student_vehicle_goal_env import (
    DEFAULT_STUDENT_VEHICLE_USD,
    _default_tunable_config_json,
    _dry_ground_material_cfg,
    build_goal_beacon_marker,
    goal_beacon_visualization,
    _hide_ground_visuals,
    _spawn_ground,
    _source_env_vehicle_root_path,
    build_student_vehicle_articulation_cfg,
)
from src.scene_factory_multiworld_scene import (
    _build_single_world_roads_only,
    _load_scene_cfg,
    _load_yaml,
    extract_vehicle_spawns_from_json,
)
from src.student_vehicle_sysid import (
    StudentTunableConfig,
    _apply_runtime_student_dynamics,
    load_tunable_config,
    normalize_tunable_config,
)
from src.trfc import encode_weather_context, prepare_stage_world_specs, weather_context_dim

ensure_isaaclab_source_paths()

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, sample_uniform, subtract_frame_transforms


OBSERVATION_MODE_DIMS = {
    "full": 22,
    "goal_reaching": 6,
}
_COLLIDABLE_VEHICLE_BODIES = (
    "base_link",
)

# Safety metric thresholds (standard traffic-safety literature values)
_TTC_NEAR_MISS_THRESHOLD_S: float = 2.0   # TTC < 2 s → near-miss event
_DRAC_HIGH_THRESHOLD: float = 3.4          # DRAC > 3.4 m/s² → dangerous deceleration demand


def _load_scene_factory_lane_touch_metadata(stage, *, world_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(str(world_root))
    if not prim.IsValid():
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    try:
        custom_data = prim.GetCustomData()
    except Exception:
        custom_data = {}
    if not isinstance(custom_data, dict):
        custom_data = {}

    points = custom_data.get("road_points_m", None)
    dirs = custom_data.get("road_point_dirs", None)
    types = custom_data.get("road_point_types", None)
    half_lengths = custom_data.get("road_point_half_lengths_m", None)
    half_widths = custom_data.get("road_point_half_widths_m", None)
    if points is None or dirs is None or types is None:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    try:
        points_np = np.asarray(points, dtype=np.float32)
        dirs_np = np.asarray(dirs, dtype=np.float32)
        types_np = np.asarray(types, dtype=np.int64)
    except Exception:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    if points_np.ndim != 2 or points_np.shape[0] == 0 or points_np.shape[1] < 2:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    if dirs_np.ndim != 2 or dirs_np.shape[0] != points_np.shape[0] or dirs_np.shape[1] < 2:
        dirs_np = np.zeros((points_np.shape[0], 2), dtype=np.float32)
    else:
        dirs_np = dirs_np[:, :2]
    norms = np.linalg.norm(dirs_np, axis=1, keepdims=True)
    dirs_np = np.divide(dirs_np, np.maximum(norms, 1.0e-6), out=np.zeros_like(dirs_np), where=norms > 1.0e-6)

    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    if not math.isfinite(mpu) or mpu <= 0.0:
        mpu = 1.0
    points_xy = points_np[:, :2] * mpu

    if half_lengths is not None:
        try:
            half_lengths_np = np.asarray(half_lengths, dtype=np.float32).reshape(-1) * mpu
        except Exception:
            half_lengths_np = np.zeros((points_np.shape[0],), dtype=np.float32)
    else:
        half_lengths_np = np.zeros((points_np.shape[0],), dtype=np.float32)

    if half_widths is not None:
        try:
            half_widths_np = np.asarray(half_widths, dtype=np.float32).reshape(-1) * mpu
        except Exception:
            half_widths_np = np.zeros((points_np.shape[0],), dtype=np.float32)
    else:
        half_widths_np = np.zeros((points_np.shape[0],), dtype=np.float32)

    if half_lengths_np.shape[0] != points_np.shape[0]:
        half_lengths_np = np.zeros((points_np.shape[0],), dtype=np.float32)
    if half_widths_np.shape[0] != points_np.shape[0]:
        half_widths_np = np.zeros((points_np.shape[0],), dtype=np.float32)

    return points_xy, dirs_np, half_lengths_np, half_widths_np, types_np.reshape(-1)


def _load_student_vehicle_dimensions_m(usd_path: str | Path) -> tuple[float, float, float]:
    default_chassis_length_m = 4.0
    default_chassis_width_m = 2.0
    default_wheelbase_m = 2.6

    usd_path = Path(usd_path).expanduser().resolve()
    meta_path = usd_path.with_name("student_vehicle_import_meta.json")
    chassis_length_m = default_chassis_length_m
    chassis_width_m = default_chassis_width_m
    wheelbase_m = default_wheelbase_m
    if meta_path.is_file():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            spec = payload.get("spec", {}) or {}
            chassis_length_m = float(spec.get("chassis_length_m", chassis_length_m))
            chassis_width_m = float(spec.get("chassis_width_m", chassis_width_m))
            wheelbase_m = float(spec.get("wheelbase_m", wheelbase_m))
        except Exception:
            pass

    return float(chassis_length_m), float(chassis_width_m), float(wheelbase_m)


def _build_vehicle_lane_touch_circle_proxy(usd_path: str | Path) -> tuple[torch.Tensor, float]:
    chassis_length_m, chassis_width_m, wheelbase_m = _load_student_vehicle_dimensions_m(usd_path)

    half_length_m = max(0.5, 0.5 * float(chassis_length_m))
    radius_m = max(0.45, 0.55 * float(chassis_width_m))
    offset_mag_m = min(0.5 * float(wheelbase_m), max(0.0, half_length_m - 0.8 * radius_m))
    centers_b = torch.tensor(
        [
            [float(offset_mag_m), 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-float(offset_mag_m), 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return centers_b, float(radius_m)


def _scene_factory_bounds_size_from_cfg(cfg: "StudentVehicleMultiAgentGoalEnvCfg") -> float:
    scene_factory_cfg = _load_yaml(cfg.scene_factory_config_path)
    world_cfg = dict(scene_factory_cfg.get("world", {}) or {})
    return float(world_cfg.get("bounds_size_m", 200.0))


def _scene_factory_weather_context_from_spec(cfg: "StudentVehicleMultiAgentGoalEnvCfg", world_spec) -> np.ndarray:
    if world_spec is None or getattr(world_spec, "friction_estimate", None) is None:
        return np.zeros((weather_context_dim(),), dtype=np.float32)
    estimate = world_spec.friction_estimate
    return np.asarray(
        encode_weather_context(
            water_film_mm=getattr(estimate, "water_film_mm", 0.0),
            road_type=getattr(estimate, "road_type", None),
        ),
        dtype=np.float32,
    )


def _reference_road_point_feat_dim(include_dirs: bool) -> int:
    return 5 if bool(include_dirs) else 3


def _reference_vehicle_feat_dim(include_ttc: bool, include_index: bool) -> int:
    dim = 6
    if bool(include_ttc):
        dim += 1
    if bool(include_index):
        dim += 1
    return dim


_LOWPOLY_CAR_USDZ = str(
    Path(__file__).resolve().parent.parent / "artifacts" / "low_poly_car_proxy.usd"
)


def _build_vehicle_proxy_marker(
    prim_path: str,
    *,
    num_agent_prototypes: int,
    vehicle_length_m: float,
    vehicle_width_m: float,
    vehicle_height_m: float = 0.55,
) -> VisualizationMarkers:
    """Build car-shaped proxy markers using a low-poly USDZ asset.

    The USDZ has: up=Y, metersPerUnit=0.01, car long-axis along Z.
    Isaac Sim is Z-up with forward along +X.
    We apply scale=0.01 (cm→m) and a rotation to convert Y-up/Z-forward
    to Z-up/X-forward.
    """
    # scale: cm → m
    _scale = 0.01
    marker_cfg = CUBOID_MARKER_CFG.copy()
    marker_cfg.prim_path = str(prim_path)
    marker_cfg.markers = {}
    for agent_idx in range(max(1, int(num_agent_prototypes))):
        marker_cfg.markers[f"vehicle_{agent_idx}"] = sim_utils.UsdFileCfg(
            usd_path=_LOWPOLY_CAR_USDZ,
            scale=(_scale, _scale, _scale),
        )
    return VisualizationMarkers(marker_cfg)


def _reference_observation_dim(cfg: "StudentVehicleMultiAgentGoalEnvCfg") -> int:
    dim = 7
    if bool(cfg.obs_weather_context_enable):
        dim += int(weather_context_dim())
    if bool(cfg.obs_road_points_enable):
        dim += int(cfg.obs_road_points_k) * int(_reference_road_point_feat_dim(cfg.obs_road_points_include_dirs))
    if bool(cfg.obs_neighbor_enable):
        dim += int(cfg.obs_neighbor_k) * int(
            _reference_vehicle_feat_dim(cfg.obs_neighbor_include_ttc, cfg.obs_neighbor_include_index)
        )
    return int(dim)


def _wrap_pi_torch(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _world_to_ego_xy_torch(dx: torch.Tensor, dy: torch.Tensor, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    x_ego = cy * dx + sy * dy
    y_ego = -sy * dx + cy * dy
    return x_ego, y_ego


def _ego_to_world_xy_torch(x_ego: torch.Tensor, y_ego: torch.Tensor, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    dx = cy * x_ego - sy * y_ego
    dy = sy * x_ego + cy * y_ego
    return dx, dy


def multi_agent_obs_dim(observation_mode: str, cfg: "StudentVehicleMultiAgentGoalEnvCfg" | None = None) -> int:
    mode = str(observation_mode).strip().lower()
    if mode == "choco_reference":
        if cfg is None:
            raise ValueError("cfg is required to size the 'choco_reference' observation mode")
        return _reference_observation_dim(cfg)
    if mode not in OBSERVATION_MODE_DIMS:
        raise ValueError(f"Unsupported observation mode: {observation_mode!r}")
    return int(OBSERVATION_MODE_DIMS[mode])


def configure_multi_agent_spaces(cfg: "StudentVehicleMultiAgentGoalEnvCfg", num_agents_per_env: int):
    agent_ids = [f"vehicle_{idx}" for idx in range(int(num_agents_per_env))]
    obs_dim = multi_agent_obs_dim(getattr(cfg, "observation_mode", "full"), cfg=cfg)
    cfg.num_agents_per_env = int(num_agents_per_env)
    cfg.possible_agents = agent_ids
    cfg.action_spaces = {
        agent_id: gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) for agent_id in agent_ids
    }
    cfg.observation_spaces = {agent_id: obs_dim for agent_id in agent_ids}
    cfg.state_space = 0
    return cfg


def resolve_scene_factory_world_and_spawns(cfg: "StudentVehicleMultiAgentGoalEnvCfg"):
    scene_factory_cfg = _load_yaml(cfg.scene_factory_config_path)
    world_specs = prepare_stage_world_specs(scene_factory_cfg)
    if not world_specs:
        raise RuntimeError(f"No SceneFactory worlds resolved from {cfg.scene_factory_config_path}")

    vehicles_cfg = dict(scene_factory_cfg.get("vehicles", {}) or {})
    bounds_size_m = float((scene_factory_cfg.get("world", {}) or {}).get("bounds_size_m", 200.0))
    origin_mode = str((scene_factory_cfg.get("world", {}) or {}).get("origin_mode", "center"))
    origin_center_mode = str((scene_factory_cfg.get("world", {}) or {}).get("origin_center_mode", "mean"))
    requested_agents = max(int(cfg.num_agents_per_env), 1)

    requested_world_index = int(cfg.scene_factory_world_index)
    world_spec = world_specs[requested_world_index % len(world_specs)]
    spawns = extract_vehicle_spawns_from_json(
        world_spec.scene_json_path,
        bounds_size_m=bounds_size_m,
        origin_mode=origin_mode,
        origin_center_mode=origin_center_mode,
        max_controllable=requested_agents,
        require_goal_in_bounds=bool(vehicles_cfg.get("require_goal_in_bounds", True)),
        skip_if_start_in_goal=bool(vehicles_cfg.get("skip_if_start_in_goal", True)),
        goal_radius_m=float(vehicles_cfg.get("goal_radius_m", cfg.goal_reached_threshold_m)),
        start_goal_thresh_m=vehicles_cfg.get("start_goal_thresh_m"),
    )
    print(
        "[INFO] SceneFactory world selection: "
        f"world_index={world_spec.world_index} scene={world_spec.scene_json_name} spawns={len(spawns)} "
        f"crop={bounds_size_m:.1f}x{bounds_size_m:.1f}m origin_mode={origin_mode} center_mode={origin_center_mode}"
    )
    return world_spec, list(spawns)


def resolve_scene_factory_spawn_subset(cfg: "StudentVehicleMultiAgentGoalEnvCfg") -> list:
    _, spawns = resolve_scene_factory_world_and_spawns(cfg)
    return spawns


def resolve_scene_factory_env_assignments(cfg: "StudentVehicleMultiAgentGoalEnvCfg"):
    scene_factory_cfg = _load_yaml(cfg.scene_factory_config_path)
    world_specs = prepare_stage_world_specs(scene_factory_cfg)
    if not world_specs:
        raise RuntimeError(f"No SceneFactory worlds resolved from {cfg.scene_factory_config_path}")

    vehicles_cfg = dict(scene_factory_cfg.get("vehicles", {}) or {})
    bounds_size_m = float((scene_factory_cfg.get("world", {}) or {}).get("bounds_size_m", 200.0))
    origin_mode = str((scene_factory_cfg.get("world", {}) or {}).get("origin_mode", "center"))
    origin_center_mode = str((scene_factory_cfg.get("world", {}) or {}).get("origin_center_mode", "mean"))
    requested_agents = max(int(cfg.num_agents_per_env), 1)
    num_envs = max(int(cfg.scene.num_envs), 1)
    selection_mode = str(getattr(cfg, "scene_factory_world_selection_mode", "fixed")).strip().lower().replace("_", "-")
    world_seed = int(getattr(cfg, "scene_factory_random_world_seed", cfg.seed if getattr(cfg, "seed", None) is not None else 0))

    if selection_mode == "fixed":
        requested_world_index = int(cfg.scene_factory_world_index)
        selected_specs = [world_specs[requested_world_index % len(world_specs)] for _ in range(num_envs)]
    elif selection_mode == "sequential":
        selected_specs = [world_specs[i % len(world_specs)] for i in range(num_envs)]
    elif selection_mode == "random-envs":
        rng = random.Random(world_seed)
        order = list(range(len(world_specs)))
        selected_specs = []
        while len(selected_specs) < num_envs:
            rng.shuffle(order)
            for spec_idx in order:
                selected_specs.append(world_specs[spec_idx])
                if len(selected_specs) >= num_envs:
                    break
    else:
        raise ValueError(f"Unsupported scene_factory_world_selection_mode: {selection_mode!r}")

    per_env_spawns: list[list] = []
    per_env_specs: list = []
    min_available = requested_agents
    for env_index, world_spec in enumerate(selected_specs):
        spawns = extract_vehicle_spawns_from_json(
            world_spec.scene_json_path,
            bounds_size_m=bounds_size_m,
            origin_mode=origin_mode,
            origin_center_mode=origin_center_mode,
            max_controllable=requested_agents,
            require_goal_in_bounds=bool(vehicles_cfg.get("require_goal_in_bounds", True)),
            skip_if_start_in_goal=bool(vehicles_cfg.get("skip_if_start_in_goal", True)),
            goal_radius_m=float(vehicles_cfg.get("goal_radius_m", cfg.goal_reached_threshold_m)),
            start_goal_thresh_m=vehicles_cfg.get("start_goal_thresh_m"),
        )
        if len(spawns) <= 0:
            print(
                f"[WARNING] SceneFactory: {world_spec.scene_json_name} yields no controllable spawns "
                f"(env_{env_index}) — skipping this scene."
            )
            continue
        min_available = min(min_available, len(spawns))
        per_env_specs.append(world_spec)
        per_env_spawns.append(list(spawns))

    if not per_env_specs:
        raise RuntimeError(
            "SceneFactory: every selected scene yielded zero controllable spawns. "
            "Check bounds_size_m, require_goal_in_bounds, and your scene JSON files."
        )

    # If we filtered some scenes out, fill remaining env slots by cycling through valid ones
    if len(per_env_specs) < num_envs:
        n_valid = len(per_env_specs)
        print(
            f"[WARNING] SceneFactory: {num_envs - n_valid} scene(s) were skipped due to no spawns. "
            f"Cycling through {n_valid} valid scene(s) to fill {num_envs} env slots."
        )
        while len(per_env_specs) < num_envs:
            per_env_specs.append(per_env_specs[len(per_env_specs) % n_valid])
            per_env_spawns.append(per_env_spawns[len(per_env_spawns) % n_valid])

    if min_available < requested_agents:
        availability = ", ".join(
            f"env_{env_idx}:{spec.scene_json_name}={len(spawns)}"
            for env_idx, (spec, spawns) in enumerate(zip(per_env_specs, per_env_spawns))
        )
        print(
            "[INFO] SceneFactory selected worlds provide fewer controllable spawns than requested for some envs: "
            f"requested={requested_agents}, available=[{availability}]. "
            "Missing agent slots will stay inactive for those envs."
        )

    num_agents = requested_agents
    trimmed_spawns = [list(spawns[:num_agents]) for spawns in per_env_spawns]
    scenes = ", ".join(f"env_{env_idx}:{spec.scene_json_name}" for env_idx, spec in enumerate(per_env_specs))
    print(
        "[INFO] SceneFactory env assignments: "
        f"selection_mode={selection_mode} num_envs={num_envs} agents_requested={num_agents} scenes=[{scenes}]"
    )
    return per_env_specs, trimmed_spawns


def _scene_factory_road_render_mode(cfg: "StudentVehicleMultiAgentGoalEnvCfg") -> str:
    scene_factory_cfg = _load_yaml(cfg.scene_factory_config_path)
    road_cfg = dict(scene_factory_cfg.get("road", {}) or {})
    return str(road_cfg.get("render_mode", "point_instancer")).strip().lower()


@configclass
class StudentVehicleMultiAgentGoalEnvCfg(DirectMARLEnvCfg):
    episode_length_s = 15.0
    decimation = 4
    debug_vis = True
    ui_window_class_type = None

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="min",
            restitution_combine_mode="min",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            # Default 5*2**15=163840 is insufficient when suspension stiffness
            # is non-zero (sysid v4): all wheels generate contact patches at
            # spawn simultaneously. 2**22=4194304 gives ~25x headroom.
            gpu_max_rigid_patch_count=2**22,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=12.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    num_agents_per_env: int = 2
    possible_agents = ["vehicle_0", "vehicle_1"]
    action_spaces = {
        "vehicle_0": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        "vehicle_1": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
    }
    observation_mode: str = "full"
    observation_spaces = {
        "vehicle_0": OBSERVATION_MODE_DIMS["full"],
        "vehicle_1": OBSERVATION_MODE_DIMS["full"],
    }
    state_space = 0

    student_usd_path: str = DEFAULT_STUDENT_VEHICLE_USD
    tunable_config_json: str = _default_tunable_config_json()

    spawn_height_m: float = 1.6
    ground_mode: str = "plane"
    use_scene_factory_roads: bool = False
    scene_factory_config_path: str = "configs/scene_factory/multiworld_scene.yaml"
    scene_factory_world_index: int = 0
    scene_factory_world_selection_mode: str = "fixed"
    scene_factory_random_world_seed: int = 42
    reset_mode: str = "isaac_reset"
    start_radius_m: float = 0.5
    agent_spawn_circle_radius_m: float = 3.5
    agent_spawn_jitter_m: float = 0.12
    randomize_spawn_phase: bool = True
    spawn_yaw_noise_rad: float = 0.5
    goal_heading_noise_rad: float = 0.75
    apply_runtime_external_wrench: bool = True
    goal_radius_min_m: float = 5.0
    goal_radius_max_m: float = 8.0
    goal_height_m: float = 0.05
    goal_reached_threshold_m: float = 0.85
    fall_height_threshold_m: float = 0.18
    bad_tilt_gravity_threshold: float = -0.15
    max_distance_from_origin_m: float = 14.0
    agent_neighbor_obs_scale_m: float = 12.0
    agent_safe_distance_m: float = 2.0
    agent_collision_distance_m: float = 1.1
    agent_collision_force_threshold_n: float = 25.0
    agent_collision_warmup_steps: int = 24
    lane_touch_enabled: bool = True
    lane_touch_margin_m: float = 0.40
    reward_lane_center_enable: bool = True
    reward_lane_center_types: tuple[int, ...] = (1, 2)
    reward_lane_center_per_step: float = 0.05
    reward_lane_forbidden_enable: bool = True
    reward_lane_forbidden_types: tuple[int, ...] = (15, 16)
    reward_lane_forbidden_penalty: float = -30.0
    reward_mode: str = "scene_factory_default"
    reward_choco_offroad_penalty: float = -0.5
    reward_choco_idle_penalty_enable: bool = True
    reward_choco_idle_penalty_per_step: float = 0.03
    reward_choco_idle_speed_threshold_mps: float = 0.5
    reward_choco_speed_bonus_enable: bool = False
    reward_choco_speed_bonus_per_step: float = 0.02
    reward_choco_speed_bonus_max_mps: float = 10.0
    reward_choco_geom_lane_enable: bool = True
    reward_choco_geom_lane_per_step: float = 0.12
    reward_choco_geom_lane_tolerance_m: float = 1.75
    reward_choco_geom_lane_heading_weight: float = 0.8
    reward_choco_geom_lane_min_alignment: float = 0.35
    reward_choco_geom_route_progress_weight: float = 0.0
    reward_choco_geom_offroad_enable: bool = True
    reward_choco_geom_offroad_lateral_threshold_m: float = 3.25
    reward_choco_geom_offroad_distance_threshold_m: float = 6.0
    reward_choco_geom_lane_types: tuple[int, ...] = (1, 2)
    reward_choco_geom_road_edge_types: tuple[int, ...] = (15, 16)
    reward_choco_ttc_penalty_enable: bool = True
    reward_choco_ttc_penalty_alpha: float = 0.15
    reward_choco_ttc_penalty_max: float = 0.50
    reward_choco_ttc_penalty_min_ttc: float = 0.5
    reward_choco_road_edge_ttc_penalty_enable: bool = True
    reward_choco_road_edge_ttc_penalty_alpha: float = 0.10
    reward_choco_road_edge_ttc_penalty_max: float = 0.60
    reward_choco_road_edge_ttc_penalty_min_ttc: float = 0.5
    reward_choco_road_edge_ttc_hard_min_ttc: float = 0.5
    reward_choco_road_edge_ttc_radius_m: float = 40.0
    obs_weather_context_enable: bool = True
    obs_weather_context_blind: bool = False  # If True, feed all-zeros weather context regardless of actual surface (for OOD eval of dry-trained models)
    obs_road_points_enable: bool = True
    obs_road_points_k: int = 200
    obs_road_points_radius_m: float = 35.0
    obs_road_points_type_norm: float = 20.0
    obs_road_points_mode: str = "road-running"
    obs_road_points_include_dirs: bool = False
    obs_neighbor_enable: bool = True
    obs_neighbor_k: int = 63
    obs_neighbor_include_ttc: bool = False
    obs_neighbor_include_index: bool = False
    obs_neighbor_ttc_max_s: float = 10.0
    obs_timing_print_enable: bool = False
    obs_timing_print_every_n: int = 32
    step_timing_log_enable: bool = False
    step_timing_print_enable: bool = False
    step_timing_print_every_n: int = 128
    step_timing_cuda_sync_enable: bool = False

    test_mode: str = "none"
    collision_test_half_distance_m: float = 12.0
    collision_test_goal_distance_m: float = 40.0
    collision_test_settle_steps: int = 24
    collision_test_drive_steps: int = 360
    collision_test_post_collision_steps: int = 120
    collision_test_throttle: float = 0.85
    collision_test_steering: float = 0.0
    collision_test_brake: float = 0.0
    collision_test_post_collision_throttle: float = 0.0
    collision_test_post_collision_steering: float = 0.0
    collision_test_post_collision_brake: float = 1.0
    collision_test_debug_markers: bool = False
    random_steer_test_settle_steps: int = 24
    random_steer_test_drive_steps: int = 600
    random_steer_test_throttle: float = 1.0
    random_steer_test_brake: float = 0.0
    random_steer_test_steering_min: float = -1.0
    random_steer_test_steering_max: float = 1.0
    random_steer_test_steering_hold_steps: int = 12
    random_steer_test_seed: int = 123
    invincible: bool = False

    # Dynamics backend: "physx" uses the full articulated rigid-body simulation;
    # "bicycle" replaces PhysX with a GPU-batched kinematic bicycle model and
    # writes root poses directly each step (useful for physics-gap ablations).
    dynamics_mode: str = "physx"
    bicycle_wheelbase_m: float = 2.6       # distance between front and rear axles
    bicycle_lr_ratio: float = 0.45         # L_r / wheelbase  (rear-axle to CoM fraction)
    bicycle_max_speed_mps: float = 15.0    # hard clamp on longitudinal speed
    bicycle_accel_scale: float = 6.0       # throttle → m/s² gain
    bicycle_steer_limit_rad: float = 0.52  # max front wheel angle (≈30 deg)

    friction_ruler_mode: bool = False
    friction_ruler_mu_values: str = ""  # comma-separated per-env μ, e.g. "1.1,0.6,0.3,0.1"
    friction_ruler_labels: str = ""  # comma-separated per-env labels for video overlay
    fixed_action: str = ""  # e.g. "0.8,0.0,0.0" → [throttle, steer, brake]
    action_schedule: str = ""  # e.g. "0:1.0,0.0,0.0 60:1.0,1.0,0.0" step:t,s,b entries
    random_od: bool = False
    random_od_min_travel_m: float = 20.0
    random_od_max_travel_m: float = 60.0
    random_od_lane_types: tuple[int, ...] = (1, 2)

    reward_scale_alive: float = 0.05
    reward_scale_progress: float = 10.0
    reward_scale_goal_shaping: float = 1.5
    reward_scale_heading: float = 0.35
    reward_scale_speed_to_goal: float = 0.20
    reward_scale_lateral_velocity: float = -0.08
    reward_scale_yaw_rate: float = -0.03
    reward_scale_action_rate: float = -0.02
    reward_scale_action_magnitude: float = -0.002
    reward_scale_throttle_brake_conflict: float = -0.10
    reward_scale_neighbor_proximity: float = -0.20
    reward_goal_bonus: float = 20.0
    reward_collision_penalty: float = -15.0
    reward_crash_penalty: float = -10.0

    capture_camera_enabled: bool = False
    capture_camera_width: int = 1280
    capture_camera_height: int = 720
    capture_camera_focal_length: float = 24.0
    capture_camera_horizontal_aperture: float = 20.955
    capture_camera_padding_scale: float = 1.35
    capture_camera_height_scale: float = 1.6
    capture_camera_view_mode: str = "whole_grid"
    capture_camera_env_index: int = 0
    capture_camera_pose_mode: str = "top_down"  # "top_down" or "traffic_cam"
    capture_camera_traffic_cam_height_m: float = 7.0
    capture_camera_traffic_cam_distance_m: float = 25.0
    capture_camera_traffic_cam_look_height_m: float = 0.5
    capture_camera_traffic_cam_azimuth_deg: float = 0.0  # 0=south, positive=counterclockwise (left from cam POV)
    capture_camera_traffic_cam_lateral_offset_m: float = 0.0  # shift camera left (positive) or right (negative)
    # Flyover camera: cinematic reveal — surveillance → tilt to horizon → gentle zoom-out
    capture_camera_flyover_start_height_m: float = 8.0
    capture_camera_flyover_end_height_m: float = 200.0
    capture_camera_flyover_surveillance_frames: int = 120  # phase 1: hold as surveillance cam
    capture_camera_flyover_tilt_frames: int = 180  # phase 2: tilt up to reveal neighbors
    capture_camera_flyover_zoomout_frames: int = 300  # phase 3: gentle rise
    capture_camera_flyover_start_env_index: int = 0
    capture_camera_flyover_start_tilt_deg: float = 25.0  # low surveillance angle from horizontal
    capture_camera_flyover_end_tilt_deg: float = 75.0  # nearly top-down at end
    capture_camera_flyover_start_distance_m: float = 25.0  # horizontal offset at start
    capture_camera_flyover_azimuth_deg: float = 0.0  # viewing direction
    capture_camera_flyover_lookaway_frames: int = 0  # phase 4: tilt head up + pan left (0 = disabled)
    capture_camera_flyover_lookaway_pitch_deg: float = 45.0  # how far to tilt up from final look direction
    capture_camera_flyover_lookaway_yaw_deg: float = 45.0  # how far to pan left
    # Flyover-drift: rise → lateral drift left (slow start) → pan right + pitch up
    capture_camera_drift_rise_frames: int = 300  # phase 1: rise to overview height
    capture_camera_drift_lateral_frames: int = 300  # phase 2: drift left
    capture_camera_drift_pan_frames: int = 300  # phase 3: pan right + pitch up
    capture_camera_drift_hold_frames: int = 0  # phase 4: hold final pose
    capture_camera_drift_start_height_m: float = 8.0
    capture_camera_drift_rise_height_m: float = 400.0  # height at end of rise
    capture_camera_drift_lateral_distance_m: float = 600.0  # how far to drift left
    capture_camera_drift_pan_yaw_deg: float = 90.0  # how far to pan right
    capture_camera_drift_pitch_up_deg: float = 30.0  # how far to tilt up at end
    capture_camera_drift_start_tilt_deg: float = 25.0  # initial surveillance tilt
    capture_camera_drift_rise_tilt_deg: float = 70.0  # tilt at top of rise
    capture_camera_drift_azimuth_deg: float = 0.0  # initial viewing direction
    hide_goal_markers: bool = False  # hide destination beacon visuals (for clean video capture)
    vehicle_proxy_marker_enable: bool = False
    vehicle_proxy_marker_z_offset_m: float = -0.5


class StudentVehicleMultiAgentGoalEnv(DirectMARLEnv):
    cfg: StudentVehicleMultiAgentGoalEnvCfg

    def __init__(self, cfg: StudentVehicleMultiAgentGoalEnvCfg, render_mode: str | None = None, **kwargs):
        self._capture_camera: Camera | None = None
        self._capture_cameras_per_env: list[Camera] = []  # for per_env view mode
        self._vehicle_proxy_marker: VisualizationMarkers | None = None
        self._capture_cam_center_xy: tuple[float, float] | None = None
        self._capture_cam_half_w: float = 0.0
        self._capture_cam_half_h: float = 0.0
        self._scenario_spawns: list | None = None
        self._scenario_spawns_by_env: list[list] | None = None
        self._scene_factory_spawn_start_local = None
        self._scene_factory_spawn_start_yaw = None
        self._scene_factory_spawn_goal_local = None
        self._scene_factory_spawn_valid = None
        self._scene_factory_scene_json_path: str | None = None
        self._scene_factory_scene_json_paths_by_env: list[str] | None = None
        self._scene_factory_scene_cfgs_by_env: list[dict[str, Any]] | None = None
        self._scene_factory_specs_by_env: list | None = None
        self._random_od_rng = np.random.default_rng(42)
        self._scene_factory_flatten_road_z = False
        self._scene_factory_ignore_dataset_spawn_z = False
        self._scene_factory_bounds_size_m = float(_scene_factory_bounds_size_from_cfg(cfg))
        self._last_reset_world_episode_summaries: list[dict[str, Any]] = []
        self._weather_context_np = np.zeros((weather_context_dim(),), dtype=np.float32)
        self._weather_context = torch.zeros((0, weather_context_dim()), dtype=torch.float32)
        self._obs_timing_call_count = 0
        self._obs_timing_last_ms = 0.0
        self._obs_timing_ema_ms = 0.0
        self._step_timing_call_count = 0
        self._step_timing_last_ms: dict[str, float] = {
            "pre_physics_ms": 0.0,
            "apply_action_ms": 0.0,
            "apply_action_math_ms": 0.0,
            "apply_action_target_submit_ms": 0.0,
            "apply_action_wrench_submit_ms": 0.0,
            "physics_write_ms": 0.0,
            "physics_sim_ms": 0.0,
            "physics_update_ms": 0.0,
            "obs_ms": 0.0,
            "obs_lane_ms": 0.0,
            "obs_shared_ms": 0.0,
            "obs_shared_stack_ms": 0.0,
            "obs_shared_yaw_speed_ms": 0.0,
            "obs_shared_ttc_ms": 0.0,
            "obs_goal_ms": 0.0,
            "obs_road_ms": 0.0,
            "obs_neighbor_ms": 0.0,
            "obs_finalize_ms": 0.0,
            "reward_ms": 0.0,
            "reward_lane_ms": 0.0,
            "reward_shared_ms": 0.0,
            "reward_goal_ms": 0.0,
            "reward_geom_ms": 0.0,
            "reward_route_progress_ms": 0.0,
            "reward_ttc_ms": 0.0,
            "reward_road_edge_ttc_ms": 0.0,
            "reward_finalize_ms": 0.0,
            "done_ms": 0.0,
            "done_lane_ms": 0.0,
            "done_collision_ms": 0.0,
            "done_state_ms": 0.0,
            "done_finalize_ms": 0.0,
            "reset_ms": 0.0,
            "reset_metrics_prep_ms": 0.0,
            "reset_log_ms": 0.0,
            "reset_backend_ms": 0.0,
            "reset_spawn_prep_ms": 0.0,
            "reset_state_clear_ms": 0.0,
            "reset_pose_build_ms": 0.0,
            "reset_pose_park_ms": 0.0,
            "reset_pose_spawn_ms": 0.0,
            "reset_pose_finalize_ms": 0.0,
            "reset_pose_quat_ms": 0.0,
            "reset_pose_goal_ms": 0.0,
            "reset_pose_inactive_ms": 0.0,
            "reset_pose_joint_defaults_ms": 0.0,
            "reset_write_ms": 0.0,
            "step_total_ms": 0.0,
            "step_bookkeeping_ms": 0.0,
            "step_event_ms": 0.0,
            "step_obs_noise_ms": 0.0,
            "step_other_ms": 0.0,
        }
        self._step_timing_ema_ms: dict[str, float] = dict(self._step_timing_last_ms)
        self._teleport_only_reset_initialized = False
        self._teleport_only_reset_announced = False
        self._lane_touch_mask_cache_valid = False
        self._collision_sensor_names_by_agent: list[str] = []
        self._collision_sensors_by_agent: list[ContactSensor | None] = []
        self._collision_force_cache_valid = False
        self._collision_force_cache = torch.zeros((0, 0), dtype=torch.float32)
        self._lane_touch_points_xy_m = torch.zeros((0, 0, 2), dtype=torch.float32)
        self._lane_touch_dirs_xy = torch.zeros((0, 0, 2), dtype=torch.float32)
        self._lane_touch_half_lengths_m = torch.zeros((0, 0), dtype=torch.float32)
        self._lane_touch_half_widths_m = torch.zeros((0, 0), dtype=torch.float32)
        self._lane_touch_types = torch.zeros((0, 0), dtype=torch.long)
        self._lane_touch_valid = torch.zeros((0, 0), dtype=torch.bool)
        self._lane_touch_type_one_hot = torch.zeros((0, 0, 1), dtype=torch.bool)
        self._lane_touch_type_dim = 1
        self._lane_touch_circle_centers_b = torch.zeros((3, 3), dtype=torch.float32)
        self._lane_touch_circle_centers_xy_b = torch.zeros((3, 2), dtype=torch.float32)
        self._lane_touch_circle_radius_m = 1.0
        self._lane_touch_mask = torch.zeros((0, 0, 1), dtype=torch.bool)
        if bool(cfg.use_scene_factory_roads) and str(cfg.test_mode).strip().lower() != "collision_test":
            scene_factory_cfg = _load_yaml(cfg.scene_factory_config_path)
            road_cfg = dict(scene_factory_cfg.get("road", {}) or {})
            self._scene_factory_flatten_road_z = bool(
                road_cfg.get("flatten_road_z", road_cfg.get("flatten_road", True))
            )
            self._scene_factory_ignore_dataset_spawn_z = bool(self._scene_factory_flatten_road_z)
            resolved_specs, resolved_spawns_by_env = resolve_scene_factory_env_assignments(cfg)
            cfg.scene_factory_world_index = int(resolved_specs[0].world_index)
            self._scene_factory_specs_by_env = list(resolved_specs)
            self._scenario_spawns_by_env = [list(spawns) for spawns in resolved_spawns_by_env]
            num_envs_cfg = max(1, int(cfg.scene.num_envs))
            num_agents_cfg = max(1, int(cfg.num_agents_per_env))
            spawn_start_local_np = np.zeros((num_envs_cfg, num_agents_cfg, 3), dtype=np.float32)
            spawn_start_yaw_np = np.zeros((num_envs_cfg, num_agents_cfg), dtype=np.float32)
            spawn_goal_local_np = np.zeros((num_envs_cfg, num_agents_cfg, 3), dtype=np.float32)
            spawn_valid_np = np.zeros((num_envs_cfg, num_agents_cfg), dtype=np.bool_)
            for env_idx, spawns in enumerate(self._scenario_spawns_by_env[:num_envs_cfg]):
                active_agents = min(num_agents_cfg, len(spawns))
                for agent_idx in range(active_agents):
                    spawn = spawns[agent_idx]
                    spawn_start_local_np[env_idx, agent_idx] = np.asarray(spawn.start_local_xyz, dtype=np.float32)
                    spawn_start_yaw_np[env_idx, agent_idx] = float(spawn.start_yaw_rad)
                    spawn_goal_local_np[env_idx, agent_idx] = np.asarray(spawn.goal_local_xyz, dtype=np.float32)
                    spawn_valid_np[env_idx, agent_idx] = True
            self._scenario_spawns = list(self._scenario_spawns_by_env[0])
            self._scene_factory_scene_json_paths_by_env = [
                str(Path(spec.scene_json_path).expanduser().resolve()) for spec in resolved_specs
            ]
            self._scene_factory_scene_json_path = str(self._scene_factory_scene_json_paths_by_env[0])
            if bool(cfg.random_od):
                self._scene_factory_scene_cfgs_by_env = [
                    _load_scene_cfg(p) for p in self._scene_factory_scene_json_paths_by_env
                ]
                print(
                    f"[INFO][SceneFactory] random_od=true: cached {len(self._scene_factory_scene_cfgs_by_env)} "
                    f"scene_cfg dicts for runtime OD resampling "
                    f"(travel={cfg.random_od_min_travel_m:.0f}-{cfg.random_od_max_travel_m:.0f}m, "
                    f"lane_types={cfg.random_od_lane_types}).",
                    flush=True,
                )
            self._weather_context_np = np.stack(
                [_scene_factory_weather_context_from_spec(cfg, spec) for spec in resolved_specs],
                axis=0,
            ).astype(np.float32)
            self._scene_factory_spawn_start_local = spawn_start_local_np
            self._scene_factory_spawn_start_yaw = spawn_start_yaw_np
            self._scene_factory_spawn_goal_local = spawn_goal_local_np
            self._scene_factory_spawn_valid = spawn_valid_np
            if self._scene_factory_ignore_dataset_spawn_z:
                print(
                    "[INFO][SceneFactory] flatten_road_z=true: ignoring dataset spawn/goal z and using flat training heights.",
                    flush=True,
                )
        configure_multi_agent_spaces(cfg, cfg.num_agents_per_env)
        self._tunable_config = normalize_tunable_config(
            load_tunable_config(cfg.tunable_config_json) if str(cfg.tunable_config_json) else StudentTunableConfig()
        )
        vehicle_length_m, vehicle_width_m, _wheelbase_m = _load_student_vehicle_dimensions_m(cfg.student_usd_path)
        self._vehicle_length_m = float(vehicle_length_m)
        self._vehicle_width_m = float(vehicle_width_m)
        circle_centers_b, circle_radius_m = _build_vehicle_lane_touch_circle_proxy(cfg.student_usd_path)
        self._lane_touch_circle_centers_b = circle_centers_b
        self._lane_touch_circle_centers_xy_b = circle_centers_b[:, :2].contiguous()
        self._lane_touch_circle_radius_m = float(circle_radius_m)
        self._agent_ids = list(cfg.possible_agents)
        self._collision_sensor_names_by_agent = ["" for _ in self._agent_ids]
        self._collision_sensors_by_agent = [None for _ in self._agent_ids]
        super().__init__(cfg, render_mode, **kwargs)
        self._configure_capture_camera_pose()
        self._lane_touch_circle_centers_b = self._lane_touch_circle_centers_b.to(self.device)
        self._lane_touch_circle_centers_xy_b = self._lane_touch_circle_centers_xy_b.to(self.device)
        weather_context = torch.as_tensor(self._weather_context_np, dtype=torch.float32, device=self.device)
        if weather_context.ndim == 1:
            weather_context = weather_context.unsqueeze(0).repeat(self.num_envs, 1)
        self._weather_context = weather_context
        # Constant used when obs_weather_context_blind=True: dry AC surface (water_film=0, road_type=AC)
        # This matches the fixed context seen by weather-unaware models trained on a single dry surface.
        _dry_ac = encode_weather_context(water_film_mm=0.0, road_type="AC")
        self._weather_context_blind_const = torch.tensor(
            _dry_ac, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        if self._scene_factory_spawn_start_local is not None:
            self._scene_factory_spawn_start_local = torch.as_tensor(
                self._scene_factory_spawn_start_local, dtype=torch.float32, device=self.device
            )
            self._scene_factory_spawn_start_yaw = torch.as_tensor(
                self._scene_factory_spawn_start_yaw, dtype=torch.float32, device=self.device
            )
            self._scene_factory_spawn_goal_local = torch.as_tensor(
                self._scene_factory_spawn_goal_local, dtype=torch.float32, device=self.device
            )
            self._scene_factory_spawn_valid = torch.as_tensor(
                self._scene_factory_spawn_valid, dtype=torch.bool, device=self.device
            )

        self._num_agents = len(self._agent_ids)
        self._vehicles = [self.scene.articulations[agent_id] for agent_id in self._agent_ids]
        self._bicycle_speed_buf = torch.zeros(
            (self._num_agents, self.num_envs), dtype=torch.float32, device=self.device
        )
        self._steps_since_reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._raw_actions = torch.zeros(self._num_agents, self.num_envs, 3, device=self.device)
        self._semantic_actions = torch.zeros_like(self._raw_actions)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._goal_pos_w = torch.zeros(self._num_agents, self.num_envs, 3, device=self.device)
        self._previous_goal_distance = torch.zeros(self._num_agents, self.num_envs, device=self.device)
        self._current_goal_distance = torch.zeros(self._num_agents, self.num_envs, device=self.device)
        self._terminal_goal_distance = torch.zeros(self._num_agents, self.num_envs, device=self.device)
        self._previous_root_pos_xy = torch.zeros(self._num_agents, self.num_envs, 2, device=self.device)
        self._spawned_agent_mask = torch.ones(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._agent_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._collision_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._crash_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._crash_too_low_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._crash_too_far_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._crash_bad_tilt_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._lane_forbidden_done_mask = torch.zeros(
            self._num_agents, self.num_envs, dtype=torch.bool, device=self.device
        )
        self._pending_goal_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_collision_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_crash_done_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_crash_too_low_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_crash_too_far_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_crash_bad_tilt_mask = torch.zeros(self._num_agents, self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_lane_forbidden_done_mask = torch.zeros(
            self._num_agents, self.num_envs, dtype=torch.bool, device=self.device
        )
        self._lifetime_controlled_spawn_count = 0.0
        self._lifetime_success_count = 0.0
        self._lifetime_all_goals_reached_count = 0.0
        self._lifetime_crash_count = 0.0
        self._lifetime_crash_too_low_count = 0.0
        self._lifetime_crash_too_far_count = 0.0
        self._lifetime_crash_bad_tilt_count = 0.0
        self._lifetime_collision_count = 0.0
        self._lifetime_lane_center_touch_count = 0.0
        self._lifetime_lane_forbidden_count = 0.0

        # TTC / DRAC episode-level tracking buffers (shape: [num_agents, num_envs])
        self._episode_min_ttc_sum = torch.zeros(self._num_agents, self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_ttc_finite_steps = torch.zeros(self._num_agents, self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_near_miss_steps = torch.zeros(self._num_agents, self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_max_drac = torch.zeros(self._num_agents, self.num_envs, dtype=torch.float32, device=self.device)
        # Lifetime TTC / DRAC counters
        self._lifetime_near_miss_count = 0.0
        self._lifetime_high_drac_count = 0.0
        self._lifetime_ttc_episode_count = 0.0  # episodes with ≥1 finite TTC step

        if self._scene_factory_spawn_valid is not None:
            self._spawned_agent_mask = self._scene_factory_spawn_valid.transpose(0, 1).clone()

        self._steer_joint_ids: list[list[int]] = []
        self._drive_joint_ids: list[list[int]] = []
        self._brake_joint_ids: list[list[int]] = []
        self._wheel_joint_ids: list[list[int]] = []
        self._suspension_joint_ids: list[list[int]] = []
        self._base_body_id: list[list[int]] = []
        self._base_body_ids: list[torch.Tensor] = []
        self._joint_effort_targets: list[torch.Tensor] = []
        self._external_forces: list[torch.Tensor] = []
        self._external_torques: list[torch.Tensor] = []
        self._brake_sign_memory: list[torch.Tensor] = []
        self._default_root_pose: list[torch.Tensor] = []
        self._default_joint_pos: list[torch.Tensor] = []
        self._default_joint_vel: list[torch.Tensor] = []

        for vehicle in self._vehicles:
            steer_joint_ids, _ = vehicle.find_joints(
                ["front_left_steer_joint", "front_right_steer_joint"], preserve_order=True
            )
            drive_joint_ids, _ = vehicle.find_joints(
                ["front_left_wheel_joint", "front_right_wheel_joint"], preserve_order=True
            )
            brake_joint_ids, _ = vehicle.find_joints(
                [
                    "front_left_wheel_joint",
                    "front_right_wheel_joint",
                    "rear_left_wheel_joint",
                    "rear_right_wheel_joint",
                ],
                preserve_order=True,
            )
            suspension_joint_ids, _ = vehicle.find_joints(
                [
                    "front_left_suspension_joint",
                    "front_right_suspension_joint",
                    "rear_left_suspension_joint",
                    "rear_right_suspension_joint",
                ],
                preserve_order=True,
            )
            base_body_id, _ = vehicle.find_bodies("base_link")
            self._steer_joint_ids.append(list(steer_joint_ids))
            self._drive_joint_ids.append(list(drive_joint_ids))
            self._brake_joint_ids.append(list(brake_joint_ids))
            self._wheel_joint_ids.append(list(brake_joint_ids))
            self._suspension_joint_ids.append(list(suspension_joint_ids))
            self._base_body_id.append(list(base_body_id))
            self._base_body_ids.append(torch.tensor(base_body_id, dtype=torch.int32, device=self.device))
            self._joint_effort_targets.append(torch.zeros(self.num_envs, vehicle.num_joints, device=self.device))
            self._external_forces.append(torch.zeros(self.num_envs, len(base_body_id), 3, device=self.device))
            self._external_torques.append(torch.zeros(self.num_envs, len(base_body_id), 3, device=self.device))
            self._brake_sign_memory.append(torch.ones(self.num_envs, len(brake_joint_ids), device=self.device))
            self._default_root_pose.append(vehicle.data.default_root_state[:, :7].clone())
            self._default_joint_pos.append(vehicle.data.default_joint_pos.clone())
            self._default_joint_vel.append(vehicle.data.default_joint_vel.clone())

            vehicle.write_joint_viscous_friction_coefficient_to_sim(
                joint_viscous_friction_coeff=torch.full(
                    (self.num_envs, len(steer_joint_ids)),
                    float(self._tunable_config.steering_viscous_friction),
                    device=self.device,
                ),
                joint_ids=steer_joint_ids,
            )
            vehicle.write_joint_viscous_friction_coefficient_to_sim(
                joint_viscous_friction_coeff=torch.full(
                    (self.num_envs, len(brake_joint_ids)),
                    float(self._tunable_config.wheel_viscous_friction),
                    device=self.device,
                ),
                joint_ids=brake_joint_ids,
            )
            vehicle.write_joint_viscous_friction_coefficient_to_sim(
                joint_viscous_friction_coeff=torch.full(
                    (self.num_envs, len(suspension_joint_ids)),
                    float(self._tunable_config.suspension_viscous_friction),
                    device=self.device,
                ),
                joint_ids=suspension_joint_ids,
            )

        # --- Apply per-env tire friction from weather/friction pipeline ---
        self._apply_per_env_tire_friction()

        self._steer_joint_ids_tensor = torch.tensor(self._steer_joint_ids, dtype=torch.long, device=self.device)
        self._drive_joint_ids_tensor = torch.tensor(self._drive_joint_ids, dtype=torch.long, device=self.device)
        self._brake_joint_ids_tensor = torch.tensor(self._brake_joint_ids, dtype=torch.long, device=self.device)

        self._steer_limit = float(self._tunable_config.steering_limit_rad)
        self._dry_longitudinal_scale = float(self._tunable_config.surface_longitudinal_scale.get("dry_asphalt", 1.0))
        self._dry_lateral_scale = float(self._tunable_config.surface_lateral_scale.get("dry_asphalt", 1.0))

        reward_keys = (
            "alive",
            "progress",
            "goal_shaping",
            "heading",
            "speed_to_goal",
            "lateral_velocity",
            "yaw_rate",
            "action_rate",
            "action_magnitude",
            "throttle_brake_conflict",
            "neighbor_proximity",
            "goal_bonus",
            "collision_penalty",
            "crash_penalty",
            "lane_center_bonus",
            "lane_forbidden_penalty",
            "offroad_penalty",
            "idle_penalty",
            "speed_bonus",
            "ttc_penalty",
            "road_edge_ttc_penalty",
            "geom_lane_reward",
            "geom_route_progress",
        )
        self._episode_sums = {
            key: torch.zeros(self._num_agents, self.num_envs, dtype=torch.float32, device=self.device)
            for key in reward_keys
        }
        self._lane_touch_mask = torch.zeros(
            (self._num_agents, self.num_envs, int(self._lane_touch_type_dim)),
            dtype=torch.bool,
            device=self.device,
        )

        self.set_debug_vis(bool(self.cfg.debug_vis))

    def _setup_scene(self):
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        spawned_vehicles: dict[str, Articulation] = {}
        for agent_idx, agent_id in enumerate(self._agent_ids):
            prim_path = f"/World/envs/env_.*/Vehicle_{agent_idx}"
            vehicle_cfg = build_student_vehicle_articulation_cfg(
                self.cfg.student_usd_path,
                spawn_height_m=float(self.cfg.spawn_height_m),
                prim_path=prim_path,
            )
            vehicle = Articulation(vehicle_cfg)
            spawned_vehicles[agent_id] = vehicle
            _apply_runtime_student_dynamics(
                stage=stage,
                student_root_path=_source_env_vehicle_root_path(prim_path),
                config=self._tunable_config,
            )

        _spawn_ground("/World/ground", _dry_ground_material_cfg(self._tunable_config), mode=self.cfg.ground_mode)
        if self.cfg.use_scene_factory_roads and str(self.cfg.ground_mode).strip().lower() == "plane":
            _hide_ground_visuals("/World/ground")

        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.use_scene_factory_roads:
            print("[INFO] Building SceneFactory roads independently inside each env after cloning.")
            self._build_scene_factory_worlds(stage)
            self._initialize_lane_touch_metadata(stage)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        for agent_id, vehicle in spawned_vehicles.items():
            # Register scene entities after cloning to match Isaac Lab's direct MARL task setup.
            self.scene.articulations[agent_id] = vehicle
        self._register_vehicle_contact_sensors()

        if bool(self.cfg.friction_ruler_mode):
            self._build_friction_ruler_visuals(stage)
        # Hide USD vehicle meshes & spawn 3D proxy markers whenever proxy markers are enabled
        if bool(self.cfg.vehicle_proxy_marker_enable) or bool(self.cfg.friction_ruler_mode):
            self._hide_vehicle_visuals(stage)
        if bool(self.cfg.vehicle_proxy_marker_enable):
            if self._vehicle_proxy_marker is None:
                self._vehicle_proxy_marker = _build_vehicle_proxy_marker(
                    "/Visuals/VehicleProxyMarkers",
                    num_agent_prototypes=int(self.cfg.num_agents_per_env),
                    vehicle_length_m=float(self._vehicle_length_m),
                    vehicle_width_m=float(self._vehicle_width_m),
                )
            self._vehicle_proxy_marker.set_visibility(True)

        # Visually hide road types that shouldn't appear on camera (e.g. lane centers)
        if hasattr(self.cfg, "road_hidden_types") and self.cfg.road_hidden_types:
            self._hide_road_type_visuals(stage, self.cfg.road_hidden_types)

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._spawn_capture_camera()

    def _vehicle_collision_filter_paths(self, sensor_agent_idx: int) -> list[str]:
        filter_paths: list[str] = []
        for other_agent_idx in range(len(self._agent_ids)):
            if other_agent_idx == sensor_agent_idx:
                continue
            for body_name in _COLLIDABLE_VEHICLE_BODIES:
                filter_paths.append(f"/World/envs/env_.*/Vehicle_{other_agent_idx}/{body_name}")
        return filter_paths

    def _register_vehicle_contact_sensors(self) -> None:
        if len(self._agent_ids) <= 1:
            return
        for agent_idx, agent_id in enumerate(self._agent_ids):
            filter_paths = self._vehicle_collision_filter_paths(agent_idx)
            body_name = _COLLIDABLE_VEHICLE_BODIES[0]
            sensor_name = f"{agent_id}_contact_{body_name}"
            sensor_cfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Vehicle_{agent_idx}/{body_name}",
                update_period=0.0,
                debug_vis=False,
                filter_prim_paths_expr=filter_paths,
            )
            contact_sensor = ContactSensor(sensor_cfg)
            self.scene.sensors[sensor_name] = contact_sensor
            self._collision_sensor_names_by_agent[agent_idx] = sensor_name
            self._collision_sensors_by_agent[agent_idx] = contact_sensor

    def _spawn_capture_camera(self) -> None:
        if not bool(self.cfg.capture_camera_enabled):
            self._capture_camera = None
            return

        view_mode = str(self.cfg.capture_camera_view_mode).strip().lower()

        if view_mode == "per_env":
            # Spawn one camera per environment
            self._capture_cameras_per_env = []
            for ei in range(self.num_envs):
                cam_cfg = CameraCfg(
                    prim_path=f"/World/SceneFactoryCaptureCamera_env{ei}",
                    update_period=0.0,
                    height=int(self.cfg.capture_camera_height),
                    width=int(self.cfg.capture_camera_width),
                    data_types=["rgb"],
                    colorize_instance_id_segmentation=False,
                    colorize_instance_segmentation=False,
                    colorize_semantic_segmentation=False,
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=float(self.cfg.capture_camera_focal_length),
                        focus_distance=400.0,
                        horizontal_aperture=float(self.cfg.capture_camera_horizontal_aperture),
                        clipping_range=(0.1, 1.0e6),
                    ),
                )
                self._capture_cameras_per_env.append(Camera(cam_cfg))
            self._capture_camera = None  # not used in per_env mode
            return

        camera_cfg = CameraCfg(
            prim_path="/World/SceneFactoryCaptureCamera",
            update_period=0.0,
            height=int(self.cfg.capture_camera_height),
            width=int(self.cfg.capture_camera_width),
            data_types=["rgb"],
            colorize_instance_id_segmentation=False,
            colorize_instance_segmentation=False,
            colorize_semantic_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(self.cfg.capture_camera_focal_length),
                focus_distance=400.0,
                horizontal_aperture=float(self.cfg.capture_camera_horizontal_aperture),
                clipping_range=(0.1, 1.0e6),
            ),
        )
        self._capture_camera = Camera(camera_cfg)

    def _configure_capture_camera_pose(self) -> None:
        env_origins = self.scene.env_origins.detach()

        # --- per_env mode: position each camera over its env ---
        if self._capture_cameras_per_env:
            for ei, cam in enumerate(self._capture_cameras_per_env):
                sc = env_origins[ei]
                if bool(self.cfg.friction_ruler_mode):
                    # Friction ruler: top-down camera centered on the car's starting position
                    h = 50.0
                    eye = torch.tensor(
                        [[float(sc[0]), float(sc[1]) + 10.0, h]],
                        dtype=torch.float32, device=self.device,
                    )
                    target = torch.tensor(
                        [[float(sc[0]), float(sc[1]) + 10.0, 0.0]],
                        dtype=torch.float32, device=self.device,
                    )
                else:
                    pose_mode = str(self.cfg.capture_camera_pose_mode).strip().lower()
                    if pose_mode == "traffic_cam":
                        # Traffic-light style: offset camera behind/above, tilted down
                        import math as _math
                        cam_h = float(self.cfg.capture_camera_traffic_cam_height_m)
                        cam_dist = float(self.cfg.capture_camera_traffic_cam_distance_m)
                        look_h = float(self.cfg.capture_camera_traffic_cam_look_height_m)
                        az_rad = _math.radians(float(self.cfg.capture_camera_traffic_cam_azimuth_deg))
                        # base direction is -y (south); rotate CCW by azimuth
                        dx = cam_dist * -_math.sin(az_rad)
                        dy = cam_dist * -_math.cos(az_rad)
                        eye = torch.tensor(
                            [[float(sc[0]) + dx, float(sc[1]) + dy, cam_h]],
                            dtype=torch.float32, device=self.device,
                        )
                        target = torch.tensor(
                            [[float(sc[0]), float(sc[1]), look_h]],
                            dtype=torch.float32, device=self.device,
                        )
                    else:
                        # Default top-down
                        coverage_radius = float(self.cfg.max_distance_from_origin_m) * float(self.cfg.capture_camera_padding_scale)
                        h = max(40.0, float(self.cfg.capture_camera_height_scale) * float(max(25.0, coverage_radius)))
                        eye = torch.tensor([[float(sc[0]), float(sc[1]), h]], dtype=torch.float32, device=self.device)
                        target = torch.tensor([[float(sc[0]), float(sc[1]), 0.0]], dtype=torch.float32, device=self.device)
                cam.set_world_poses_from_view(eyes=eye, targets=target)
            return

        if self._capture_camera is None:
            return
        if str(self.cfg.test_mode).strip().lower() in {"collision_test", "scene_factory_collision_test"}:
            env_index = int(np.clip(int(self.cfg.capture_camera_env_index), 0, max(self.num_envs - 1, 0)))
            scene_center = env_origins[env_index]
            eye = torch.tensor(
                [[float(scene_center[0]), float(scene_center[1] - 34.0), 14.0]],
                dtype=torch.float32,
                device=self.device,
            )
            target = torch.tensor(
                [[float(scene_center[0]), float(scene_center[1]), 1.2]],
                dtype=torch.float32,
                device=self.device,
            )
            self._capture_camera.set_world_poses_from_view(eyes=eye, targets=target)
            return

        view_mode = str(self.cfg.capture_camera_view_mode).strip().lower()
        if view_mode == "single_env":
            env_index = int(np.clip(int(self.cfg.capture_camera_env_index), 0, max(self.num_envs - 1, 0)))
            scene_center = env_origins[env_index]
            xy_extent = 0.0
            coverage_radius = float(self.cfg.max_distance_from_origin_m) * float(self.cfg.capture_camera_padding_scale)
        else:
            scene_center = env_origins.mean(dim=0)
            if self.num_envs > 1:
                xy_extent = torch.linalg.norm(env_origins[:, :2] - scene_center[:2], dim=1).max().item()
            else:
                xy_extent = 0.0
            coverage_radius = (
                float(xy_extent)
                + float(self.cfg.max_distance_from_origin_m)
                + float(self.cfg.scene.env_spacing) * 0.25
            ) * float(self.cfg.capture_camera_padding_scale)
        pose_mode = str(getattr(self.cfg, "capture_camera_pose_mode", "top_down")).strip().lower()
        if pose_mode == "traffic_cam":
            import math as _math
            cam_h = float(getattr(self.cfg, "capture_camera_traffic_cam_height_m", 7.0))
            cam_d = float(getattr(self.cfg, "capture_camera_traffic_cam_distance_m", 25.0))
            look_h = float(getattr(self.cfg, "capture_camera_traffic_cam_look_height_m", 0.5))
            az_rad = _math.radians(float(getattr(self.cfg, "capture_camera_traffic_cam_azimuth_deg", 0.0)))
            lat_off = float(getattr(self.cfg, "capture_camera_traffic_cam_lateral_offset_m", 0.0))
            dx = cam_d * -_math.sin(az_rad)
            dy = cam_d * -_math.cos(az_rad)
            # "Left" from camera POV is 90° CCW from forward direction
            fwd_x, fwd_y = -dx, -dy
            fwd_len = max(1e-6, _math.sqrt(fwd_x**2 + fwd_y**2))
            left_x, left_y = -fwd_y / fwd_len, fwd_x / fwd_len
            cx = float(scene_center[0]) + left_x * lat_off
            cy = float(scene_center[1]) + left_y * lat_off
            eye = torch.tensor(
                [[cx + dx, cy + dy, cam_h]],
                dtype=torch.float32,
                device=self.device,
            )
            target = torch.tensor(
                [[cx, cy, look_h]],
                dtype=torch.float32,
                device=self.device,
            )
            capture_height = cam_h  # for projection params below
        else:
            capture_height = max(
                40.0,
                float(self.cfg.capture_camera_height_scale) * float(max(25.0, coverage_radius)),
            )
            eye = torch.tensor(
                [[float(scene_center[0]), float(scene_center[1]), capture_height]],
                dtype=torch.float32,
                device=self.device,
            )
            target = torch.tensor(
                [[float(scene_center[0]), float(scene_center[1]), 0.0]],
                dtype=torch.float32,
                device=self.device,
            )
        self._capture_camera.set_world_poses_from_view(eyes=eye, targets=target)
        # Store projection parameters for 2D vehicle proxy overlay
        h_aperture = float(self.cfg.capture_camera_horizontal_aperture)
        f_length = float(self.cfg.capture_camera_focal_length)
        self._capture_cam_center_xy = (float(scene_center[0]), float(scene_center[1]))
        self._capture_cam_half_w = capture_height * h_aperture / (2.0 * f_length)
        self._capture_cam_half_h = self._capture_cam_half_w * int(self.cfg.capture_camera_height) / max(1, int(self.cfg.capture_camera_width))

    def _update_vehicle_proxy_markers(self) -> None:
        """Sync 3D low-poly proxy marker positions/orientations with current vehicle state."""
        if self._vehicle_proxy_marker is None:
            return
        positions = []
        orientations = []
        indices = []
        z_offset = float(self.cfg.vehicle_proxy_marker_z_offset_m)
        for agent_idx, vehicle in enumerate(self._vehicles):
            pos_w = vehicle.data.root_pos_w.clone()
            if not hasattr(self, "_proxy_z_logged"):
                print(
                    f"[ProxyMarker] agent={agent_idx} root_z={pos_w[0, 2].item():.3f} "
                    f"z_offset={z_offset:.3f} final_z={pos_w[0, 2].item() + z_offset:.3f}",
                    flush=True,
                )
            pos_w[:, 2] += z_offset
            positions.append(pos_w)
            orientations.append(vehicle.data.root_quat_w.clone())
            indices.append(
                torch.full((self.num_envs,), agent_idx, dtype=torch.int32, device=self.device)
            )
        if not hasattr(self, "_proxy_z_logged"):
            self._proxy_z_logged = True
        self._vehicle_proxy_marker.visualize(
            translations=torch.cat(positions, dim=0),
            orientations=torch.cat(orientations, dim=0),
            marker_indices=torch.cat(indices, dim=0),
        )

    def _update_flyover_camera(self) -> None:
        """Animate flyover camera: surveillance → tilt to horizon → gentle zoom-out.

        Always stays centered on the starting world.
        Phase 1 (surveillance): low traffic-cam view of one scene.
        Phase 2 (tilt): camera tilts up to reveal neighboring worlds on the horizon.
        Phase 3 (zoom-out): gently rise to show the bigger grid picture.
        """
        if self._capture_camera is None:
            return
        pose_mode = str(self.cfg.capture_camera_pose_mode).strip().lower()
        if pose_mode != "flyover":
            return
        import math as _math

        if not hasattr(self, "_flyover_frame"):
            self._flyover_frame = 0
            env_origins = self.scene.env_origins
            start_idx = int(min(self.cfg.capture_camera_flyover_start_env_index, max(self.num_envs - 1, 0)))
            self._flyover_center_xy = (float(env_origins[start_idx, 0]), float(env_origins[start_idx, 1]))

        frame = self._flyover_frame
        p1 = max(1, int(self.cfg.capture_camera_flyover_surveillance_frames))
        p2 = max(1, int(self.cfg.capture_camera_flyover_tilt_frames))
        p3 = max(1, int(self.cfg.capture_camera_flyover_zoomout_frames))

        h_start = float(self.cfg.capture_camera_flyover_start_height_m)
        h_end = float(self.cfg.capture_camera_flyover_end_height_m)
        tilt_start = float(self.cfg.capture_camera_flyover_start_tilt_deg)
        tilt_end = float(self.cfg.capture_camera_flyover_end_tilt_deg)
        dist_start = float(self.cfg.capture_camera_flyover_start_distance_m)
        az_rad = _math.radians(float(self.cfg.capture_camera_flyover_azimuth_deg))

        cx, cy = self._flyover_center_xy

        def _ease(t: float) -> float:
            return 0.5 - 0.5 * _math.cos(min(1.0, max(0.0, t)) * _math.pi)

        p4 = max(0, int(self.cfg.capture_camera_flyover_lookaway_frames))

        if frame < p1:
            # Phase 1: hold surveillance pose
            tilt_deg = tilt_start
            cam_h = h_start
        elif frame < p1 + p2:
            # Phase 2: tilt up, slight height increase
            frac = _ease((frame - p1) / p2)
            tilt_deg = tilt_start + (tilt_end - tilt_start) * frac * 0.5  # tilt halfway
            cam_h = h_start + (h_end - h_start) * frac * 0.15  # rise a little
        elif frame < p1 + p2 + p3:
            # Phase 3: continue tilting + zoom out
            frac = _ease((frame - p1 - p2) / p3)
            tilt_deg = tilt_start + (tilt_end - tilt_start) * (0.5 + 0.5 * frac)
            cam_h = h_start + (h_end - h_start) * (0.15 + 0.85 * frac)
        else:
            # Phase 4: hold height, tilt head up + pan left
            cam_h = h_end
            tilt_deg = tilt_end  # base tilt stays at end value

        tilt_rad = _math.radians(tilt_deg)
        horiz_dist = cam_h / max(0.01, _math.tan(tilt_rad))

        eye_x = cx + horiz_dist * (-_math.sin(az_rad))
        eye_y = cy + horiz_dist * (-_math.cos(az_rad))

        eye = torch.tensor([[eye_x, eye_y, cam_h]], dtype=torch.float32, device=self.device)

        # Phase 4: smoothly move the look-target up (pitch) and left (yaw)
        if p4 > 0 and frame >= p1 + p2 + p3:
            frac4 = _ease(min(1.0, (frame - p1 - p2 - p3) / p4))
            pitch_offset_deg = float(self.cfg.capture_camera_flyover_lookaway_pitch_deg) * frac4
            yaw_offset_deg = float(self.cfg.capture_camera_flyover_lookaway_yaw_deg) * frac4
            # Move target upward (pitch up = raise target Z)
            target_z = cam_h * _math.tan(_math.radians(pitch_offset_deg))
            # Move target left (positive yaw_offset = counterclockwise = left from camera POV)
            yaw_shift_rad = _math.radians(yaw_offset_deg)
            # "Left" from camera's POV is perpendicular to the camera-to-center direction
            # Camera faces along (cx - eye_x, cy - eye_y); left is 90° counterclockwise
            fwd_x, fwd_y = cx - eye_x, cy - eye_y
            left_x, left_y = -fwd_y, fwd_x  # 90° CCW
            fwd_len = max(1e-6, _math.sqrt(fwd_x**2 + fwd_y**2))
            left_x, left_y = left_x / fwd_len, left_y / fwd_len
            pan_dist = fwd_len * _math.tan(yaw_shift_rad)
            target = torch.tensor([[cx + left_x * pan_dist, cy + left_y * pan_dist, target_z]], dtype=torch.float32, device=self.device)
        else:
            target = torch.tensor([[cx, cy, 0.0]], dtype=torch.float32, device=self.device)

        self._capture_camera.set_world_poses_from_view(eyes=eye, targets=target)
        self._flyover_frame += 1

    def _update_flyover_drift_camera(self) -> None:
        """Animate flyover-drift camera: rise over grid center → drift left → pan right + pitch up.

        Phase 1 (rise): camera rises straight up from low surveillance to overview height,
                         always looking at the grid center.
        Phase 2 (drift): camera and look-target both slide left together (starts slow).
        Phase 3 (pan): camera holds position, pans right + pitches up.
        """
        if self._capture_camera is None:
            return
        pose_mode = str(self.cfg.capture_camera_pose_mode).strip().lower()
        if pose_mode != "flyover_drift":
            return
        import math as _math

        if not hasattr(self, "_drift_frame"):
            self._drift_frame = 0
            env_origins = self.scene.env_origins
            az_rad_init = _math.radians(float(self.cfg.capture_camera_drift_azimuth_deg))
            # "Left" from camera POV
            fwd_x0, fwd_y0 = -_math.sin(az_rad_init), -_math.cos(az_rad_init)
            left_x0, left_y0 = -fwd_y0, fwd_x0

            # Find the env closest to the geometric center of the grid
            grid_mean = env_origins[:, :2].mean(dim=0)
            dists = torch.linalg.norm(env_origins[:, :2] - grid_mean.unsqueeze(0), dim=1)
            center_idx = int(dists.argmin().item())
            cx, cy = float(env_origins[center_idx, 0]), float(env_origins[center_idx, 1])

            # Project all envs onto the "left" axis relative to center env
            rel = env_origins[:, :2] - env_origins[center_idx, :2].unsqueeze(0)
            left_proj = rel[:, 0] * left_x0 + rel[:, 1] * left_y0  # signed distance along left
            # Find the env farthest in the "left" direction
            end_idx = int(left_proj.argmax().item())
            ex, ey = float(env_origins[end_idx, 0]), float(env_origins[end_idx, 1])

            # Drift distance = distance between center env and end env projected onto left axis
            self._drift_computed_distance = float(left_proj[end_idx].item())
            self._drift_grid_center = (cx, cy)
            self._drift_end_xy = (ex, ey)
            print(
                f"[FlyoverDrift] center env={center_idx} ({cx:.0f},{cy:.0f}) → "
                f"end env={end_idx} ({ex:.0f},{ey:.0f}), "
                f"drift={self._drift_computed_distance:.0f}m"
            )

        frame = self._drift_frame
        p1 = max(1, int(self.cfg.capture_camera_drift_rise_frames))
        p2 = max(1, int(self.cfg.capture_camera_drift_lateral_frames))
        p3 = max(1, int(self.cfg.capture_camera_drift_pan_frames))
        p4_hold = max(0, int(self.cfg.capture_camera_drift_hold_frames))

        h_start = float(self.cfg.capture_camera_drift_start_height_m)
        h_rise = float(self.cfg.capture_camera_drift_rise_height_m)
        tilt_start = float(self.cfg.capture_camera_drift_start_tilt_deg)
        tilt_rise = float(self.cfg.capture_camera_drift_rise_tilt_deg)
        az_rad = _math.radians(float(self.cfg.capture_camera_drift_azimuth_deg))
        # Use computed distance (center env → farthest env in left direction)
        lateral_dist = self._drift_computed_distance
        pan_yaw_deg = float(self.cfg.capture_camera_drift_pan_yaw_deg)
        pitch_up_deg = float(self.cfg.capture_camera_drift_pitch_up_deg)

        gcx, gcy = self._drift_grid_center

        def _ease(t: float) -> float:
            return 0.5 - 0.5 * _math.cos(min(1.0, max(0.0, t)) * _math.pi)

        def _ease_in(t: float) -> float:
            t = min(1.0, max(0.0, t))
            return t * t

        # Camera forward direction (from camera toward look-target)
        fwd_x, fwd_y = -_math.sin(az_rad), -_math.cos(az_rad)
        # "Left" from camera POV is 90° CCW from forward
        left_x, left_y = -fwd_y, fwd_x

        if frame < p1:
            # Phase 1: rise straight up, always looking at grid center
            frac = _ease(frame / p1)
            cam_h = h_start + (h_rise - h_start) * frac
            tilt_deg = tilt_start + (tilt_rise - tilt_start) * frac
            look_x, look_y = gcx, gcy
            lateral_offset = 0.0
            pan_frac = 0.0
        elif frame < p1 + p2:
            # Phase 2: drift left — starts slow, accelerates
            frac = _ease_in((frame - p1) / p2)
            cam_h = h_rise
            tilt_deg = tilt_rise
            lateral_offset = lateral_dist * frac
            # Shift look-target left too so the grid stays centered in frame
            look_x = gcx + left_x * lateral_offset
            look_y = gcy + left_y * lateral_offset
            pan_frac = 0.0
        elif frame < p1 + p2 + p3:
            # Phase 3: hold position, pan right + pitch up
            frac = _ease(min(1.0, (frame - p1 - p2) / p3))
            cam_h = h_rise
            tilt_deg = tilt_rise
            lateral_offset = lateral_dist
            look_x = gcx + left_x * lateral_offset
            look_y = gcy + left_y * lateral_offset
            pan_frac = frac
        else:
            # Phase 4: hold final pose
            cam_h = h_rise
            tilt_deg = tilt_rise
            lateral_offset = lateral_dist
            look_x = gcx + left_x * lateral_offset
            look_y = gcy + left_y * lateral_offset
            pan_frac = 1.0

        tilt_rad = _math.radians(tilt_deg)
        horiz_dist = cam_h / max(0.01, _math.tan(tilt_rad))

        # Eye position: behind look-target along -forward, plus lateral offset
        eye_x = look_x + horiz_dist * (-fwd_x)
        eye_y = look_y + horiz_dist * (-fwd_y)

        eye = torch.tensor([[eye_x, eye_y, cam_h]], dtype=torch.float32, device=self.device)

        # Compute look target with pan right + pitch up
        if pan_frac > 0:
            # Pan right = negative of left direction
            right_x, right_y = -left_x, -left_y
            pan_dist = horiz_dist * _math.tan(_math.radians(pan_yaw_deg * pan_frac))
            pitch_z = cam_h * _math.tan(_math.radians(pitch_up_deg * pan_frac))
            target = torch.tensor(
                [[look_x + right_x * pan_dist, look_y + right_y * pan_dist, pitch_z]],
                dtype=torch.float32, device=self.device,
            )
        else:
            target = torch.tensor([[look_x, look_y, 0.0]], dtype=torch.float32, device=self.device)

        self._capture_camera.set_world_poses_from_view(eyes=eye, targets=target)
        self._drift_frame += 1

    def capture_fixed_camera_frame(self) -> np.ndarray | None:
        if self._capture_camera is None:
            return None
        self._update_flyover_camera()
        self._update_flyover_drift_camera()
        self._update_vehicle_proxy_markers()
        self._capture_camera.update(self.step_dt)
        rgb = self._capture_camera.data.output.get("rgb")
        if rgb is None or rgb.numel() == 0:
            return None
        frame = rgb[0].detach().cpu().numpy().copy()
        if bool(self.cfg.vehicle_proxy_marker_enable):
            frame = self._overlay_vehicle_proxy_markers_2d(frame)
        return frame

    def capture_per_env_frames(self) -> list[np.ndarray | None]:
        """Capture one RGB frame per environment (for per_env video mode)."""
        if not self._capture_cameras_per_env:
            return []
        self._update_vehicle_proxy_markers()
        frames: list[np.ndarray | None] = []
        for cam in self._capture_cameras_per_env:
            cam.update(self.step_dt)
            rgb = cam.data.output.get("rgb")
            if rgb is None or rgb.numel() == 0:
                frames.append(None)
            else:
                frames.append(rgb[0].detach().cpu().numpy().copy())
        return frames

    def _overlay_vehicle_proxy_markers_2d(self, frame: np.ndarray) -> np.ndarray:
        """Draw color-coded oriented rectangles at vehicle positions on the captured frame."""
        if self._capture_cam_center_xy is None or self._capture_cam_half_w <= 0:
            return frame
        cx, cy = self._capture_cam_center_xy
        half_w = self._capture_cam_half_w
        half_h = self._capture_cam_half_h
        img_h, img_w = frame.shape[:2]
        n_channels = frame.shape[2] if frame.ndim == 3 else 1

        palette = [
            (243, 51, 51),
            (26, 209, 235),
            (245, 199, 26),
            (89, 235, 66),
            (245, 107, 31),
            (158, 82, 245),
            (242, 77, 184),
            (166, 217, 46),
        ]
        env_idx = int(np.clip(int(self.cfg.capture_camera_env_index), 0, max(self.num_envs - 1, 0)))
        vlen = float(self._vehicle_length_m)
        vwid = float(self._vehicle_width_m)
        try:
            import cv2
            _has_cv2 = True
        except ImportError:
            _has_cv2 = False

        for agent_idx, vehicle in enumerate(self._vehicles):
            # Skip un-spawned (inactive / limbo) agents
            if hasattr(self, "_spawned_agent_mask"):
                if not bool(self._spawned_agent_mask[agent_idx, env_idx].item()):
                    continue

            pos = vehicle.data.root_pos_w[env_idx]
            quat = vehicle.data.root_quat_w[env_idx]  # (w, x, y, z)
            wx, wy = float(pos[0]), float(pos[1])

            # Skip vehicles that are clearly outside the camera view (limbo / stale)
            if abs(wx - cx) > half_w * 3 or abs(wy - cy) > half_h * 3:
                continue

            qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            color_rgb = palette[agent_idx % len(palette)]
            # Pad to match frame channels (e.g. RGBA)
            color = color_rgb + (255,) * (n_channels - 3) if n_channels > 3 else color_rgb

            if _has_cv2:
                fwd = np.array([math.cos(yaw), math.sin(yaw)])
                rgt = np.array([math.sin(yaw), -math.cos(yaw)])
                corners_world = [
                    np.array([wx, wy]) + fwd * vlen / 2 + rgt * vwid / 2,
                    np.array([wx, wy]) + fwd * vlen / 2 - rgt * vwid / 2,
                    np.array([wx, wy]) - fwd * vlen / 2 - rgt * vwid / 2,
                    np.array([wx, wy]) - fwd * vlen / 2 + rgt * vwid / 2,
                ]
                corners_px = []
                for c in corners_world:
                    px_col = int(((c[0] - cx) / half_w + 1.0) * 0.5 * img_w)
                    px_row = int((1.0 - (c[1] - cy) / half_h) * 0.5 * img_h)
                    corners_px.append([px_col, px_row])
                pts = np.array(corners_px, dtype=np.int32)
                cv2.fillPoly(frame, [pts], color)
                border_color = tuple([255] * n_channels)
                cv2.polylines(frame, [pts], True, border_color, 1)
            else:
                px_col = int(((wx - cx) / half_w + 1.0) * 0.5 * img_w)
                px_row = int((1.0 - (wy - cy) / half_h) * 0.5 * img_h)
                px_half = max(3, int(vlen / (2.0 * half_w) * img_w * 0.5))
                r1, r2 = max(0, px_row - px_half), min(img_h, px_row + px_half)
                c1, c2 = max(0, px_col - px_half), min(img_w, px_col + px_half)
                if r1 < r2 and c1 < c2:
                    frame[r1:r2, c1:c2] = color

        return frame

    def _build_scene_factory_world(self, stage, *, world_root: str, env_index: int = 0) -> None:
        scene_factory_cfg = _load_yaml(self.cfg.scene_factory_config_path)
        if self._scene_factory_scene_json_paths_by_env is not None:
            scene_json_path = self._scene_factory_scene_json_paths_by_env[int(env_index)]
        elif self._scene_factory_scene_json_path:
            scene_json_path = self._scene_factory_scene_json_path
        else:
            world_spec, _ = resolve_scene_factory_world_and_spawns(self.cfg)
            scene_json_path = str(Path(world_spec.scene_json_path).expanduser().resolve())
            self._scene_factory_scene_json_path = scene_json_path
        if self._scenario_spawns is None:
            self._scenario_spawns = resolve_scene_factory_spawn_subset(self.cfg)[: int(self.cfg.num_agents_per_env)]
        build_cfg = dict(scene_factory_cfg)
        build_cfg["world"] = dict(scene_factory_cfg.get("world", {}) or {})
        _build_single_world_roads_only(
            stage=stage,
            cfg=build_cfg,
            json_path=scene_json_path,
            world_root=world_root,
        )
        self._build_scene_factory_visual_floor(stage, world_root=world_root)

    def _build_scene_factory_worlds(self, stage) -> None:
        for env_index in range(int(self.cfg.scene.num_envs)):
            world_root = f"/World/envs/env_{env_index}/SceneFactoryWorlds/world_000"
            self._build_scene_factory_world(stage, world_root=world_root, env_index=env_index)

    def _initialize_lane_touch_metadata(self, stage) -> None:
        agent_count = len(self._agent_ids)
        if not bool(self.cfg.lane_touch_enabled) or not bool(self.cfg.use_scene_factory_roads):
            self._lane_touch_points_xy_m = torch.zeros((self.num_envs, 0, 2), dtype=torch.float32, device=self.device)
            self._lane_touch_dirs_xy = torch.zeros((self.num_envs, 0, 2), dtype=torch.float32, device=self.device)
            self._lane_touch_half_lengths_m = torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
            self._lane_touch_half_widths_m = torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device)
            self._lane_touch_types = torch.zeros((self.num_envs, 0), dtype=torch.long, device=self.device)
            self._lane_touch_valid = torch.zeros((self.num_envs, 0), dtype=torch.bool, device=self.device)
            self._lane_touch_type_one_hot = torch.zeros((self.num_envs, 0, 1), dtype=torch.bool, device=self.device)
            self._lane_touch_type_dim = 1
            self._lane_touch_mask = torch.zeros((agent_count, self.num_envs, 1), dtype=torch.bool, device=self.device)
            return

        points_by_env: list[np.ndarray] = []
        dirs_by_env: list[np.ndarray] = []
        half_lengths_by_env: list[np.ndarray] = []
        half_widths_by_env: list[np.ndarray] = []
        types_by_env: list[np.ndarray] = []
        max_segments = 0
        max_type = 0
        for env_index in range(int(self.cfg.scene.num_envs)):
            world_root = f"/World/envs/env_{env_index}/SceneFactoryWorlds/world_000"
            points_xy, dirs_xy, half_lengths_m, half_widths_m, types = _load_scene_factory_lane_touch_metadata(
                stage, world_root=world_root
            )
            points_by_env.append(points_xy)
            dirs_by_env.append(dirs_xy)
            half_lengths_by_env.append(half_lengths_m)
            half_widths_by_env.append(half_widths_m)
            types_by_env.append(types)
            max_segments = max(max_segments, int(points_xy.shape[0]))
            if types.size > 0:
                max_type = max(max_type, int(np.max(types)))

        self._lane_touch_type_dim = max(1, max_type + 1)
        self._lane_touch_points_xy_m = torch.zeros(
            (self.num_envs, max_segments, 2), dtype=torch.float32, device=self.device
        )
        self._lane_touch_dirs_xy = torch.zeros(
            (self.num_envs, max_segments, 2), dtype=torch.float32, device=self.device
        )
        self._lane_touch_half_lengths_m = torch.zeros(
            (self.num_envs, max_segments), dtype=torch.float32, device=self.device
        )
        self._lane_touch_half_widths_m = torch.zeros(
            (self.num_envs, max_segments), dtype=torch.float32, device=self.device
        )
        self._lane_touch_types = torch.zeros((self.num_envs, max_segments), dtype=torch.long, device=self.device)
        self._lane_touch_valid = torch.zeros((self.num_envs, max_segments), dtype=torch.bool, device=self.device)
        for env_index in range(int(self.cfg.scene.num_envs)):
            segment_count = int(points_by_env[env_index].shape[0])
            if segment_count <= 0:
                continue
            env_slice = slice(0, segment_count)
            self._lane_touch_points_xy_m[env_index, env_slice] = torch.as_tensor(
                points_by_env[env_index], dtype=torch.float32, device=self.device
            )
            self._lane_touch_dirs_xy[env_index, env_slice] = torch.as_tensor(
                dirs_by_env[env_index], dtype=torch.float32, device=self.device
            )
            self._lane_touch_half_lengths_m[env_index, env_slice] = torch.as_tensor(
                half_lengths_by_env[env_index], dtype=torch.float32, device=self.device
            )
            self._lane_touch_half_widths_m[env_index, env_slice] = torch.as_tensor(
                half_widths_by_env[env_index], dtype=torch.float32, device=self.device
            )
            self._lane_touch_types[env_index, env_slice] = torch.as_tensor(
                types_by_env[env_index], dtype=torch.long, device=self.device
            )
            self._lane_touch_valid[env_index, env_slice] = True
        self._lane_touch_type_one_hot = torch.nn.functional.one_hot(
            self._lane_touch_types.clamp(min=0),
            num_classes=int(self._lane_touch_type_dim),
        ).to(dtype=torch.bool)
        self._lane_touch_type_one_hot &= self._lane_touch_valid.unsqueeze(-1)
        self._lane_touch_mask = torch.zeros(
            (agent_count, self.num_envs, int(self._lane_touch_type_dim)),
            dtype=torch.bool,
            device=self.device,
        )

    def _update_lane_touch_mask(self) -> None:
        if bool(self._lane_touch_mask_cache_valid):
            return
        if (
            not bool(self.cfg.lane_touch_enabled)
            or not bool(self.cfg.use_scene_factory_roads)
            or self._lane_touch_valid.numel() == 0
            or self._lane_touch_valid.shape[1] == 0
        ):
            self._lane_touch_mask.zero_()
            self._lane_touch_mask_cache_valid = True
            return

        env_origins_xy = self.scene.env_origins[:, :2]
        dirs_xy = self._lane_touch_dirs_xy
        perp_xy = torch.stack((-dirs_xy[..., 1], dirs_xy[..., 0]), dim=-1)
        half_lengths = self._lane_touch_half_lengths_m
        half_widths = self._lane_touch_half_widths_m
        valid = self._lane_touch_valid
        circle_radius = float(self._lane_touch_circle_radius_m) + float(self.cfg.lane_touch_margin_m)
        root_pos_w = torch.stack([vehicle.data.root_pos_w for vehicle in self._vehicles], dim=0)
        root_quat_w = torch.stack([vehicle.data.root_quat_w for vehicle in self._vehicles], dim=0)
        root_pos_xy = root_pos_w[..., :2] - env_origins_xy.unsqueeze(0)
        yaw_by_agent = self._compute_yaw_by_agent(root_quat_w)
        cos_yaw = torch.cos(yaw_by_agent).unsqueeze(-1)
        sin_yaw = torch.sin(yaw_by_agent).unsqueeze(-1)
        local_x = self._lane_touch_circle_centers_xy_b[:, 0].view(1, 1, -1)
        local_y = self._lane_touch_circle_centers_xy_b[:, 1].view(1, 1, -1)
        rotated_x = cos_yaw * local_x - sin_yaw * local_y
        rotated_y = sin_yaw * local_x + cos_yaw * local_y
        circle_centers_w = torch.stack(
            (
                root_pos_xy[..., 0].unsqueeze(-1) + rotated_x,
                root_pos_xy[..., 1].unsqueeze(-1) + rotated_y,
            ),
            dim=-1,
        )

        self._lane_touch_mask.zero_()
        delta = circle_centers_w.unsqueeze(3) - self._lane_touch_points_xy_m.unsqueeze(0).unsqueeze(2)
        along = torch.sum(delta * dirs_xy.unsqueeze(0).unsqueeze(2), dim=-1).abs() - half_lengths.unsqueeze(0).unsqueeze(2)
        perp = torch.sum(delta * perp_xy.unsqueeze(0).unsqueeze(2), dim=-1).abs() - half_widths.unsqueeze(0).unsqueeze(2)
        outside_dx = torch.clamp(along, min=0.0)
        outside_dy = torch.clamp(perp, min=0.0)
        distance_sq = outside_dx.square() + outside_dy.square()
        touch = valid.unsqueeze(0).unsqueeze(2) & (distance_sq <= (circle_radius * circle_radius))
        self._lane_touch_mask.copy_(
            torch.any(
                touch.unsqueeze(-1) & self._lane_touch_type_one_hot.unsqueeze(0).unsqueeze(2),
                dim=(2, 3),
            )
        )
        self._lane_touch_mask_cache_valid = True

    def lane_touch_type_mask_by_agent(self) -> dict[str, torch.Tensor]:
        return {agent_id: self._lane_touch_mask[idx].clone() for idx, agent_id in enumerate(self._agent_ids)}

    def lane_touch_types_by_agent(self) -> dict[str, list[list[int]]]:
        result: dict[str, list[list[int]]] = {}
        for agent_idx, agent_id in enumerate(self._agent_ids):
            mask = self._lane_touch_mask[agent_idx].detach().cpu()
            result[agent_id] = [torch.nonzero(mask[env_idx], as_tuple=False).view(-1).tolist() for env_idx in range(mask.shape[0])]
        return result

    def _lane_touch_any_type_mask(self, agent_idx: int, lane_types: Sequence[int]) -> torch.Tensor:
        if (
            not bool(self.cfg.lane_touch_enabled)
            or self._lane_touch_mask.numel() == 0
            or self._lane_touch_mask.shape[-1] <= 0
        ):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        valid_types = []
        type_dim = int(self._lane_touch_type_dim)
        for road_type in lane_types:
            road_type_int = int(road_type)
            if 0 <= road_type_int < type_dim:
                valid_types.append(road_type_int)
        if not valid_types:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.any(self._lane_touch_mask[agent_idx, :, valid_types], dim=1)

    def _resample_random_od_for_envs(self, env_ids: torch.Tensor) -> None:
        """Resample OD pairs on lane centerlines for the given env indices.

        Overwrites ``_scene_factory_spawn_start_local``, ``_scene_factory_spawn_start_yaw``,
        ``_scene_factory_spawn_goal_local``, and ``_scene_factory_spawn_valid`` for all agents
        in the specified environments.
        """
        from src.trfc import sample_lane_center_start_goal_pairs
        from src.trfc.lane_center_sampler import compute_scene_center_from_road

        num_agents = max(1, int(self.cfg.num_agents_per_env))
        bounds_size_m = float(self._scene_factory_bounds_size_m)
        min_travel = float(self.cfg.random_od_min_travel_m)
        max_travel = float(self.cfg.random_od_max_travel_m)
        lane_types = tuple(int(t) for t in self.cfg.random_od_lane_types)
        env_ids_cpu = env_ids.cpu().tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)

        for env_id in env_ids_cpu:
            scene_cfg = self._scene_factory_scene_cfgs_by_env[int(env_id)]
            seed = int(self._random_od_rng.integers(0, 2**31))
            try:
                samples = sample_lane_center_start_goal_pairs(
                    scene_cfg,
                    num_agents=num_agents,
                    bounds_size_m=bounds_size_m,
                    origin_mode="center",
                    lane_types=lane_types,
                    min_travel_distance_m=min_travel,
                    max_travel_distance_m=max_travel,
                    seed=seed,
                )
            except RuntimeError:
                # If sampling fails (e.g. not enough viable lanes), keep existing OD.
                continue
            # sample_lane_center_start_goal_pairs returns coordinates in the raw
            # scene JSON frame.  _scene_factory_spawn_start_local must be in the
            # scene-center-relative "local" frame (the same frame used by the
            # road builder), so subtract the scene center.
            scene_center = compute_scene_center_from_road(scene_cfg)
            for agent_idx, sample in enumerate(samples[:num_agents]):
                self._scene_factory_spawn_start_local[env_id, agent_idx, 0] = float(sample.start_xyz[0]) - float(scene_center[0])
                self._scene_factory_spawn_start_local[env_id, agent_idx, 1] = float(sample.start_xyz[1]) - float(scene_center[1])
                self._scene_factory_spawn_start_local[env_id, agent_idx, 2] = float(sample.start_xyz[2]) - float(scene_center[2])
                self._scene_factory_spawn_start_yaw[env_id, agent_idx] = float(sample.start_yaw_rad)
                self._scene_factory_spawn_goal_local[env_id, agent_idx, 0] = float(sample.goal_xyz[0]) - float(scene_center[0])
                self._scene_factory_spawn_goal_local[env_id, agent_idx, 1] = float(sample.goal_xyz[1]) - float(scene_center[1])
                self._scene_factory_spawn_goal_local[env_id, agent_idx, 2] = float(sample.goal_xyz[2]) - float(scene_center[2])
                self._scene_factory_spawn_valid[env_id, agent_idx] = True
            # Mark any remaining agent slots as invalid
            for agent_idx in range(len(samples), num_agents):
                self._scene_factory_spawn_valid[env_id, agent_idx] = False

    def _done_vehicle_root_pose(self, agent_idx: int, env_ids: torch.Tensor) -> torch.Tensor:
        root_pose = self._default_root_pose[agent_idx][env_ids].clone()
        # Park done vehicles in a row just outside the grid boundary
        if not hasattr(self, "_parking_row_x"):
            all_origins = self.scene.env_origins  # (num_envs, 3)
            spacing = float(self.cfg.scene.env_spacing)
            self._parking_row_x = float(all_origins[:, 0].max().item()) + spacing + 5.0
            self._parking_row_y_start = float(all_origins[:, 1].min().item())
            self._parking_row_z = 0.5
        root_pose[:, 0] = self._parking_row_x
        root_pose[:, 1] = self._parking_row_y_start + 3.0 * float(agent_idx)
        root_pose[:, 2] = self._parking_row_z
        return root_pose

    def _park_done_vehicle(self, agent_idx: int, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        vehicle = self._vehicles[agent_idx]
        parked_root_pose = self._done_vehicle_root_pose(agent_idx, env_ids)
        parked_root_velocity = self._default_joint_vel[agent_idx].new_zeros((len(env_ids), 6))
        parked_joint_pos = self._default_joint_pos[agent_idx][env_ids].clone()
        parked_joint_vel = self._default_joint_vel[agent_idx][env_ids].clone()
        vehicle.write_root_pose_to_sim(parked_root_pose, env_ids=env_ids)
        vehicle.write_root_velocity_to_sim(parked_root_velocity, env_ids=env_ids)
        vehicle.write_joint_state_to_sim(parked_joint_pos, parked_joint_vel, None, env_ids)

    def _set_root_pose_quat_from_yaw(self, root_pose: torch.Tensor, yaw: torch.Tensor) -> None:
        half_yaw = 0.5 * yaw
        root_pose[:, 3] = torch.cos(half_yaw)
        root_pose[:, 4] = 0.0
        root_pose[:, 5] = 0.0
        root_pose[:, 6] = torch.sin(half_yaw)

    def _reset_mode_name(self) -> str:
        return str(getattr(self.cfg, "reset_mode", "isaac_reset")).strip().lower().replace("-", "_")

    def _apply_per_env_tire_friction(self) -> None:
        """Set wheel contact-shape friction per-env from weather friction estimates.

        Because every env shares a single PhysX ground plane whose material
        cannot vary per-env, we instead set the **wheel** collision-shape
        material on each vehicle.  The ground uses
        ``friction_combine_mode="min"``, so the effective contact friction is
        ``min(ground_μ, wheel_μ)``.  As long as the wheel μ is ≤ the ground μ
        (true for any wet condition), the ``min`` picks the wheel value —
        giving us per-env friction control.

        For each env we read the Zhao et al. friction estimate (μ_static,
        μ_dynamic) and write those values directly onto the wheel shapes.

        In **friction_ruler_mode**, μ values come directly from the config
        (``friction_ruler_mu_values``) instead of Zhao estimates.
        """
        # ── Friction ruler mode: direct μ override ──
        if bool(self.cfg.friction_ruler_mode):
            mu_values = self._friction_ruler_mu_values()
            mu_static_per_env = [max(1e-3, v) for v in mu_values[:self.num_envs]]
            # Use 80% of static for dynamic (typical ratio)
            mu_dynamic_per_env = [0.8 * s for s in mu_static_per_env]
        else:
            specs = getattr(self, "_scene_factory_specs_by_env", None)
            if not specs:
                return

            # Collect per-env friction estimates; skip if none have estimates
            estimates = [getattr(s, "friction_estimate", None) for s in specs]
            if all(e is None for e in estimates):
                return

            # Build per-env target friction (num_envs,)
            # For envs without an estimate, use a high default so the ground
            # material dominates via min().
            DEFAULT_MU = 2.0  # effectively "don't limit"
            mu_static_per_env = []
            mu_dynamic_per_env = []
            any_finite = False
            for est in estimates:
                if est is None:
                    mu_static_per_env.append(DEFAULT_MU)
                    mu_dynamic_per_env.append(DEFAULT_MU)
                else:
                    s = max(1.0e-3, float(est.mu_static))
                    d = min(s, max(1.0e-3, float(est.mu_dynamic)))
                    mu_static_per_env.append(s)
                    mu_dynamic_per_env.append(d)
                    if s < DEFAULT_MU - 0.01:
                        any_finite = True

            if not any_finite:
                return

        mu_s_t = torch.tensor(mu_static_per_env, dtype=torch.float32)   # (num_envs,)
        mu_d_t = torch.tensor(mu_dynamic_per_env, dtype=torch.float32)  # (num_envs,)

        modified_count = 0
        for vehicle in self._vehicles:
            view = vehicle.root_physx_view

            # --- Discover which shape indices belong to wheel bodies ---
            wheel_body_names = [
                "front_left_wheel_link", "front_right_wheel_link",
                "rear_left_wheel_link", "rear_right_wheel_link",
            ]
            wheel_body_ids, _ = vehicle.find_bodies(wheel_body_names)

            # Build a map: body_id -> (start_shape_idx, end_shape_idx)
            num_shapes_per_body = []
            for link_path in view.link_paths[0]:
                link_view = vehicle._physics_sim_view.create_rigid_body_view(link_path)
                num_shapes_per_body.append(link_view.max_shapes)

            wheel_shape_slices = []
            for bid in wheel_body_ids:
                start = sum(num_shapes_per_body[:bid])
                end = start + num_shapes_per_body[bid]
                wheel_shape_slices.append((start, end))

            if not wheel_shape_slices:
                continue

            # --- Read current material, overwrite wheel shapes only ---
            # material_properties shape: (num_envs, max_shapes, 3)
            #   [:, :, 0] = static_friction
            #   [:, :, 1] = dynamic_friction
            #   [:, :, 2] = restitution
            materials = view.get_material_properties()

            for start, end in wheel_shape_slices:
                materials[:, start:end, 0] = mu_s_t.unsqueeze(1)
                materials[:, start:end, 1] = mu_d_t.unsqueeze(1)

            all_env_ids = torch.arange(self.num_envs, dtype=torch.int32)
            view.set_material_properties(materials, all_env_ids)
            modified_count += 1

            # ── Read-back verification ──
            readback = view.get_material_properties()
            for env_i in range(min(self.num_envs, 8)):
                for start, end in wheel_shape_slices:
                    rb_static = readback[env_i, start, 0].item()
                    rb_dynamic = readback[env_i, start, 1].item()
                    print(
                        f"  [VERIFY] env={env_i}  wheel_shape={start}  "
                        f"μ_static_SET={mu_static_per_env[env_i]:.4f}  μ_static_READ={rb_static:.4f}  "
                        f"μ_dynamic_SET={mu_dynamic_per_env[env_i]:.4f}  μ_dynamic_READ={rb_dynamic:.4f}",
                        flush=True,
                    )

        # Log summary
        unique_mu = sorted(set(f"{s:.4f}" for s in mu_static_per_env))
        print(
            f"[INFO][SceneFactory] Applied per-env wheel friction to {modified_count} vehicle slots. "
            f"Unique μ_static values: {unique_mu or ['(all dry)']}.",
            flush=True,
        )

    def _invincible_mode_enabled(self) -> bool:
        return bool(getattr(self.cfg, "invincible", False))

    def _should_use_teleport_only_reset(self) -> bool:
        return self._reset_mode_name() == "teleport_only" and bool(self._teleport_only_reset_initialized)

    def _timing_enabled(self) -> bool:
        return bool(self.cfg.step_timing_log_enable) or bool(self.cfg.step_timing_print_enable)

    def _sync_timing_device(self) -> None:
        if not self._timing_enabled():
            return
        if not bool(getattr(self.cfg, "step_timing_cuda_sync_enable", False)):
            return
        if not isinstance(self.device, torch.device) or self.device.type != "cuda":
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _apply_lightweight_reset_state(self, env_ids: torch.Tensor) -> None:
        if hasattr(self, "episode_length_buf") and isinstance(self.episode_length_buf, torch.Tensor):
            self.episode_length_buf[env_ids] = 0
        for attr_name in ("reset_buf", "reset_terminated", "reset_time_outs", "termination_manager"):
            attr = getattr(self, attr_name, None)
            if isinstance(attr, torch.Tensor) and attr.shape[0] == self.num_envs:
                if attr.dtype == torch.bool:
                    attr[env_ids] = False
                else:
                    attr[env_ids] = 0

    def _build_scene_factory_visual_floor(self, stage, *, world_root: str) -> None:
        from pxr import Gf, UsdGeom

        scene_factory_cfg = _load_yaml(self.cfg.scene_factory_config_path)
        world_cfg = dict(scene_factory_cfg.get("world", {}) or {})
        bounds_size_m = float(world_cfg.get("bounds_size_m", 200.0))
        # Keep a thin visual-only floor slightly above the hidden physics plane so it reliably
        # wins the render without intersecting the road bars.
        floor_thickness_m = 0.01
        floor_color = Gf.Vec3f(0.18, 0.18, 0.20)
        floor_path = f"{world_root}/VisualFloor"
        cube = UsdGeom.Cube.Define(stage, floor_path)
        cube.GetSizeAttr().Set(1.0)
        api = UsdGeom.XformCommonAPI(cube)
        api.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.0))
        api.SetScale(Gf.Vec3f(bounds_size_m, bounds_size_m, floor_thickness_m))
        try:
            UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([floor_color])
        except Exception:
            pass

    def _hide_vehicle_visuals(self, stage) -> None:
        """Make USD vehicle visual meshes invisible so only proxy markers show."""
        from pxr import Usd, UsdGeom
        num_envs = self.num_envs
        for env_idx in range(num_envs):
            for agent_idx in range(len(self._agent_ids)):
                vpath = f"/World/envs/env_{env_idx}/Vehicle_{agent_idx}"
                root = stage.GetPrimAtPath(vpath)
                if not root.IsValid():
                    continue
                for prim in Usd.PrimRange(root):
                    pp = str(prim.GetPath())
                    if "/visuals/" in pp:
                        try:
                            UsdGeom.Imageable(prim).MakeInvisible()
                        except Exception:
                            pass
        print(f"[INFO][FrictionRuler] Hid USD vehicle visual meshes for {num_envs} envs.")

    def _hide_road_type_visuals(self, stage, hidden_types: list[int]) -> None:
        """Make road segment prims for specific Waymo types invisible in the render.

        Road segments are under .../SceneFactoryWorlds/world_000/Road/Type_XX/.
        This hides them visually while keeping them in the scene for observations/physics.
        """
        from pxr import Usd, UsdGeom
        hidden_set = set(int(t) for t in hidden_types)
        count = 0
        for env_idx in range(self.num_envs):
            for t in hidden_set:
                type_path = f"/World/envs/env_{env_idx}/SceneFactoryWorlds/world_000/Road/Type_{t:02d}"
                root = stage.GetPrimAtPath(type_path)
                if not root.IsValid():
                    continue
                for prim in Usd.PrimRange(root):
                    try:
                        UsdGeom.Imageable(prim).MakeInvisible()
                        count += 1
                    except Exception:
                        pass
        print(f"[INFO][SceneFactory] Hid {count} road prims for types {sorted(hidden_set)}.")

    def _build_friction_ruler_visuals(self, stage) -> None:
        """Spawn ruler lines + distance/friction labels on the ground per env."""
        from pxr import Gf, UsdGeom, Sdf

        env_origins = self.scene.env_origins.cpu().numpy()  # (num_envs, 3)
        mu_values = self._friction_ruler_mu_values()
        labels_raw = str(self.cfg.friction_ruler_labels).strip()
        labels = [s.strip() for s in labels_raw.split(",")] if labels_raw else [f"μ={mu:.2f}" for mu in mu_values]

        ruler_length_m = 50.0   # how far ahead to draw ruler
        ruler_width_m = 0.15    # line width
        lane_width_m = 4.0      # visual lane width for each env
        line_height = 0.005     # just above ground

        for env_i in range(self.num_envs):
            ox, oy, oz = float(env_origins[env_i, 0]), float(env_origins[env_i, 1]), float(env_origins[env_i, 2])
            env_root = f"/World/envs/env_{env_i}/FrictionRuler"

            # --- Draw horizontal ruler lines every 1 meter ---
            for dist_m in range(0, int(ruler_length_m) + 1):
                is_major = (dist_m % 5 == 0)
                line_w = ruler_width_m * (2.0 if is_major else 1.0)
                color = Gf.Vec3f(1.0, 1.0, 1.0) if is_major else Gf.Vec3f(0.6, 0.6, 0.6)

                line_path = f"{env_root}/line_{dist_m}"
                cube = UsdGeom.Cube.Define(stage, line_path)
                cube.GetSizeAttr().Set(1.0)
                api = UsdGeom.XformCommonAPI(cube)
                # Lines are perpendicular to Y (driving direction), at local y=dist_m
                api.SetTranslate(Gf.Vec3d(0.0, float(dist_m), line_height))
                api.SetScale(Gf.Vec3f(lane_width_m, line_w, 0.002))
                try:
                    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([color])
                except Exception:
                    pass

            # --- Distance labels at every 5m ---
            for dist_m in range(0, int(ruler_length_m) + 1, 5):
                label_path = f"{env_root}/dist_label_{dist_m}"
                # Use a small colored cube as a label marker (no text rendering in PhysX)
                # Put a distinct color block at left edge
                cube = UsdGeom.Cube.Define(stage, label_path)
                cube.GetSizeAttr().Set(1.0)
                api = UsdGeom.XformCommonAPI(cube)
                api.SetTranslate(Gf.Vec3d(-lane_width_m / 2 - 0.8, float(dist_m), line_height + 0.01))
                api.SetScale(Gf.Vec3f(1.2, 0.5, 0.002))
                # Color encodes distance: gradient from green (0m) to red (50m)
                frac = dist_m / max(ruler_length_m, 1.0)
                label_color = Gf.Vec3f(frac, 1.0 - frac, 0.0)
                try:
                    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([label_color])
                except Exception:
                    pass

            # --- Friction label: colored block at start to identify env ---
            mu = mu_values[env_i] if env_i < len(mu_values) else 1.0
            label_path = f"{env_root}/friction_label"
            cube = UsdGeom.Cube.Define(stage, label_path)
            cube.GetSizeAttr().Set(1.0)
            api = UsdGeom.XformCommonAPI(cube)
            api.SetTranslate(Gf.Vec3d(lane_width_m / 2 + 1.0, 0.0, line_height + 0.01))
            api.SetScale(Gf.Vec3f(1.5, 1.5, 0.002))
            # Blue = high friction, Red = low friction
            mu_frac = min(1.0, max(0.0, mu / 1.2))
            friction_color = Gf.Vec3f(1.0 - mu_frac, 0.2, mu_frac)
            try:
                UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([friction_color])
            except Exception:
                pass

        mu_str = ", ".join(f"env{i}={mu_values[i]:.3f}" for i in range(min(self.num_envs, len(mu_values))))
        label_str = ", ".join(labels[:self.num_envs])
        print(
            f"[INFO][FrictionRuler] Built ruler visuals for {self.num_envs} envs. "
            f"μ: [{mu_str}]. Labels: [{label_str}].",
            flush=True,
        )

    def _friction_ruler_mu_values(self) -> list[float]:
        """Parse comma-separated friction_ruler_mu_values config into a list of floats."""
        raw = str(self.cfg.friction_ruler_mu_values).strip()
        if not raw:
            return [1.0] * self.num_envs
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
        # Extend to num_envs if fewer values provided
        while len(values) < self.num_envs:
            values.append(values[-1] if values else 1.0)
        return values

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        timing_start = perf_counter()
        for agent_idx, agent_id in enumerate(self._agent_ids):
            self._raw_actions[agent_idx] = actions[agent_id].clone().clamp_(-1.0, 1.0)
            # Rectify throttle and brake so a zero policy output maps to a true neutral command.
            self._semantic_actions[agent_idx, :, 0] = torch.clamp(
                self._raw_actions[agent_idx, :, 0], min=0.0, max=1.0
            )
            self._semantic_actions[agent_idx, :, 1] = self._raw_actions[agent_idx, :, 1]
            self._semantic_actions[agent_idx, :, 2] = torch.clamp(
                self._raw_actions[agent_idx, :, 2], min=0.0, max=1.0
            )
            done_mask = self._agent_done_mask[agent_idx].unsqueeze(-1)
            self._raw_actions[agent_idx].masked_fill_(done_mask, 0.0)
            self._semantic_actions[agent_idx].masked_fill_(done_mask, 0.0)
        self._step_timing_last_ms["pre_physics_ms"] = (perf_counter() - timing_start) * 1000.0

    # ------------------------------------------------------------------
    # Bicycle-model dynamics (kinematic, GPU-batched)
    # ------------------------------------------------------------------
    def _apply_action_bicycle(self):
        """Replace PhysX articulation with a batched kinematic bicycle model.

        Actions: [throttle ∈ [-1,1], steer ∈ [-1,1], brake ∈ [0,1]]
        State written back each step via write_root_pose/velocity_to_sim so
        that observations (which read from root state) remain consistent.
        """
        dt = float(self.cfg.sim.dt) * float(self.cfg.decimation)
        L = float(self.cfg.bicycle_wheelbase_m)
        lr = L * float(self.cfg.bicycle_lr_ratio)
        v_max = float(self.cfg.bicycle_max_speed_mps)
        a_scale = float(self.cfg.bicycle_accel_scale)
        delta_max = float(self.cfg.bicycle_steer_limit_rad)

        for agent_idx, vehicle in enumerate(self._vehicles):
            # root state: [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]  (world frame)
            root_pos = vehicle.data.root_pos_w.clone()          # (num_envs, 3)
            root_quat = vehicle.data.root_quat_w.clone()         # (num_envs, 4) [qw, qx, qy, qz]
            root_lin_vel_w = vehicle.data.root_lin_vel_w.clone() # (num_envs, 3)

            actions = self._semantic_actions[agent_idx]  # (num_envs, 3)
            throttle = actions[:, 0]  # [-1, 1]  (negative = reverse)
            steer    = actions[:, 1]  # [-1, 1]
            brake    = actions[:, 2]  # [ 0, 1]

            # Current longitudinal speed — read from persistent buffer, NOT from PhysX.
            # PhysX zeroes velocity each step (friction + zero joint efforts), so reading
            # root_lin_vel_b would give ~0 every step and kill all acceleration.
            speed = self._bicycle_speed_buf[agent_idx]  # (num_envs,)

            # Acceleration from throttle/brake
            accel = throttle * a_scale - brake * a_scale * 2.0
            speed_new = (speed + accel * dt).clamp(-v_max, v_max)

            # Steer angle
            delta = steer * delta_max  # (num_envs,)

            # Slip angle at CoM
            beta = torch.atan(lr / L * torch.tan(delta))  # (num_envs,)

            # Extract yaw from quaternion — Isaac Lab convention: [qw, qx, qy, qz]
            qw = root_quat[:, 0]; qx = root_quat[:, 1]
            qy = root_quat[:, 2]; qz = root_quat[:, 3]
            yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                              1.0 - 2.0 * (qy * qy + qz * qz))  # (num_envs,)

            # Integrate position
            v_avg = 0.5 * (speed + speed_new)
            dx = v_avg * torch.cos(yaw + beta) * dt
            dy = v_avg * torch.sin(yaw + beta) * dt
            dyaw = v_avg / lr * torch.sin(beta) * dt

            new_x = root_pos[:, 0] + dx
            new_y = root_pos[:, 1] + dy
            new_yaw = yaw + dyaw

            # Build new quaternion from updated yaw (pitch/roll stay zero)
            # Isaac Lab convention: [qw, qx, qy, qz]
            half_yaw = new_yaw * 0.5
            new_quat = torch.zeros_like(root_quat)
            new_quat[:, 0] = torch.cos(half_yaw)   # qw
            new_quat[:, 3] = torch.sin(half_yaw)   # qz

            # World-frame velocity from speed + new heading
            new_vx_w = speed_new * torch.cos(new_yaw)
            new_vy_w = speed_new * torch.sin(new_yaw)

            # Assemble new root state tensors
            new_pos = root_pos.clone()
            new_pos[:, 0] = new_x
            new_pos[:, 1] = new_y
            # z is kept fixed (no vertical dynamics in bicycle model)

            new_lin_vel = root_lin_vel_w.clone()
            new_lin_vel[:, 0] = new_vx_w
            new_lin_vel[:, 1] = new_vy_w
            new_lin_vel[:, 2] = 0.0

            new_ang_vel = torch.zeros_like(vehicle.data.root_ang_vel_w)
            new_ang_vel[:, 2] = dyaw / dt  # yaw rate

            # Mask out done agents (keep them frozen)
            done = self._agent_done_mask[agent_idx]  # (num_envs,)
            new_pos[done] = root_pos[done]
            new_quat[done] = root_quat[done]
            new_lin_vel[done] = root_lin_vel_w[done]
            new_ang_vel[done] = 0.0
            speed_new = speed_new.clone()
            speed_new[done] = 0.0

            # Persist speed for next step (avoids PhysX read-back which gives ~0)
            self._bicycle_speed_buf[agent_idx] = speed_new

            # Write back into Isaac Sim
            new_pose = torch.cat([new_pos, new_quat], dim=-1)   # (num_envs, 7)
            new_vel  = torch.cat([new_lin_vel, new_ang_vel], dim=-1)  # (num_envs, 6)
            vehicle.write_root_pose_to_sim(new_pose)
            vehicle.write_root_velocity_to_sim(new_vel)

    def _apply_action(self):
        if self.cfg.dynamics_mode == "bicycle":
            self._apply_action_bicycle()
            return

        timing_start = perf_counter()
        math_ms = 0.0
        target_submit_ms = 0.0
        wrench_submit_ms = 0.0
        self._sync_timing_device()
        math_start = perf_counter()
        joint_pos_all = torch.stack([vehicle.data.joint_pos for vehicle in self._vehicles], dim=0)
        joint_vel_all = torch.stack([vehicle.data.joint_vel for vehicle in self._vehicles], dim=0)
        root_lin_vel_b_all = torch.stack([vehicle.data.root_lin_vel_b for vehicle in self._vehicles], dim=0)
        root_ang_vel_b_all = torch.stack([vehicle.data.root_ang_vel_b for vehicle in self._vehicles], dim=0)
        done_mask = self._agent_done_mask
        done_mask_joint = done_mask.unsqueeze(-1)
        done_mask_body = done_mask.unsqueeze(-1).unsqueeze(-1)

        joint_effort_targets = torch.stack(self._joint_effort_targets, dim=0)
        joint_effort_targets.zero_()

        steer_idx = self._steer_joint_ids_tensor.unsqueeze(1).expand(-1, self.num_envs, -1)
        drive_idx = self._drive_joint_ids_tensor.unsqueeze(1).expand(-1, self.num_envs, -1)
        brake_idx = self._brake_joint_ids_tensor.unsqueeze(1).expand(-1, self.num_envs, -1)

        steer_target = self._semantic_actions[:, :, 1:2] * self._steer_limit
        steer_pos_error = steer_target - torch.gather(joint_pos_all, 2, steer_idx)
        steer_vel_error = -torch.gather(joint_vel_all, 2, steer_idx)
        steer_effort = (
            float(self._tunable_config.steering_kp_nm_per_rad) * steer_pos_error
            + float(self._tunable_config.steering_kd_nm_s_per_rad) * steer_vel_error
        )
        steer_effort.clamp_(
            -float(self._tunable_config.steering_effort_limit_nm),
            float(self._tunable_config.steering_effort_limit_nm),
        )
        joint_effort_targets.scatter_(2, steer_idx, steer_effort)

        drive_effort = (
            self._semantic_actions[:, :, 0:1]
            * float(self._tunable_config.drive_torque_nm)
            * float(self._dry_longitudinal_scale)
        )
        gathered_drive = torch.gather(joint_effort_targets, 2, drive_idx)
        joint_effort_targets.scatter_(2, drive_idx, gathered_drive + drive_effort.expand_as(gathered_drive))

        brake_joint_vel = torch.gather(joint_vel_all, 2, brake_idx)
        brake_sign_memory = torch.stack(self._brake_sign_memory, dim=0)
        moving_mask = torch.abs(brake_joint_vel) > 1.0e-4
        current_sign = torch.sign(brake_joint_vel)
        current_sign = torch.where(current_sign == 0.0, brake_sign_memory, current_sign)
        brake_sign_memory = torch.where(moving_mask, current_sign, brake_sign_memory)
        brake_sign = torch.where(moving_mask, current_sign, brake_sign_memory)

        brake = self._semantic_actions[:, :, 2:3]
        front_brake_effort = brake * float(self._tunable_config.brake_front_torque_nm) * float(self._dry_longitudinal_scale)
        rear_brake_effort = brake * float(self._tunable_config.brake_rear_torque_nm) * float(self._dry_longitudinal_scale)
        front_idx = brake_idx[:, :, 0:2]
        rear_idx = brake_idx[:, :, 2:4]
        front_gather = torch.gather(joint_effort_targets, 2, front_idx)
        rear_gather = torch.gather(joint_effort_targets, 2, rear_idx)
        joint_effort_targets.scatter_(2, front_idx, front_gather - front_brake_effort.expand_as(front_gather) * brake_sign[:, :, 0:2])
        joint_effort_targets.scatter_(2, rear_idx, rear_gather - rear_brake_effort.expand_as(rear_gather) * brake_sign[:, :, 2:4])
        joint_effort_targets.masked_fill_(done_mask_joint, 0.0)

        external_forces = torch.stack(self._external_forces, dim=0)
        external_torques = torch.stack(self._external_torques, dim=0)
        external_forces.zero_()
        external_torques.zero_()
        if self.cfg.apply_runtime_external_wrench:
            external_forces[:, :, 0, 1] = (
                -float(self._tunable_config.lateral_velocity_damping_n_per_mps)
                * float(self._dry_lateral_scale)
                * root_lin_vel_b_all[:, :, 1]
            )
            external_torques[:, :, 0, 2] = (
                -float(self._tunable_config.yaw_stability_damping_nm_per_rad_s)
                * float(self._dry_lateral_scale)
                * root_ang_vel_b_all[:, :, 2]
            )
        external_forces.masked_fill_(done_mask_body, 0.0)
        external_torques.masked_fill_(done_mask_body, 0.0)
        self._sync_timing_device()
        math_ms += (perf_counter() - math_start) * 1000.0

        for agent_idx, vehicle in enumerate(self._vehicles):
            self._joint_effort_targets[agent_idx].copy_(joint_effort_targets[agent_idx])
            self._brake_sign_memory[agent_idx].copy_(brake_sign_memory[agent_idx])
            self._external_forces[agent_idx].copy_(external_forces[agent_idx])
            self._external_torques[agent_idx].copy_(external_torques[agent_idx])

            self._sync_timing_device()
            target_submit_start = perf_counter()
            vehicle.set_joint_effort_target(self._joint_effort_targets[agent_idx])
            self._sync_timing_device()
            target_submit_ms += (perf_counter() - target_submit_start) * 1000.0

            self._sync_timing_device()
            wrench_submit_start = perf_counter()
            vehicle.permanent_wrench_composer.set_forces_and_torques(
                forces=self._external_forces[agent_idx],
                torques=self._external_torques[agent_idx],
                body_ids=self._base_body_ids[agent_idx],
                is_global=False,
            )
            self._sync_timing_device()
            wrench_submit_ms += (perf_counter() - wrench_submit_start) * 1000.0
            self._step_timing_last_ms["apply_action_ms"] = (perf_counter() - timing_start) * 1000.0
        self._step_timing_last_ms["apply_action_math_ms"] += float(math_ms)
        self._step_timing_last_ms["apply_action_target_submit_ms"] += float(target_submit_ms)
        self._step_timing_last_ms["apply_action_wrench_submit_ms"] += float(wrench_submit_ms)

    def _compute_goal_position_body(self, agent_idx: int) -> torch.Tensor:
        goal_pos_b, _ = subtract_frame_transforms(
            self._vehicles[agent_idx].data.root_pos_w,
            self._vehicles[agent_idx].data.root_quat_w,
            self._goal_pos_w[agent_idx],
        )
        return goal_pos_b

    def _compute_goal_distance(self, agent_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        goal_pos_b = self._compute_goal_position_body(agent_idx)
        distance = torch.linalg.norm(goal_pos_b[:, :2], dim=1)
        return goal_pos_b, distance

    def _compute_goal_distance_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos_w = torch.stack([vehicle.data.root_pos_w for vehicle in self._vehicles], dim=0)
        root_quat_w = torch.stack([vehicle.data.root_quat_w for vehicle in self._vehicles], dim=0)
        goal_pos_w = self._goal_pos_w
        goal_pos_b, _ = subtract_frame_transforms(
            root_pos_w.reshape(-1, 3),
            root_quat_w.reshape(-1, 4),
            goal_pos_w.reshape(-1, 3),
        )
        goal_pos_b = goal_pos_b.reshape(self._num_agents, self.num_envs, 3)
        goal_distance = torch.linalg.norm(goal_pos_b[..., :2], dim=-1)
        return goal_pos_b, goal_distance

    def _pairwise_distances_xy(self) -> torch.Tensor:
        positions = torch.stack([vehicle.data.root_pos_w[:, :2] for vehicle in self._vehicles], dim=1)
        deltas = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = torch.linalg.norm(deltas, dim=-1)
        eye = torch.eye(self._num_agents, device=self.device, dtype=torch.bool).unsqueeze(0)
        return torch.where(eye, torch.full_like(distances, float("inf")), distances)

    def _collision_force_by_agent_n_tensor(self) -> torch.Tensor:
        if self._collision_force_cache_valid and self._collision_force_cache.shape == (self._num_agents, self.num_envs):
            return self._collision_force_cache
        if self._num_agents <= 1:
            forces = torch.zeros(1, self.num_envs, dtype=torch.float32, device=self.device)
            self._collision_force_cache = forces
            self._collision_force_cache_valid = True
            return forces

        forces = torch.zeros((self._num_agents, self.num_envs), dtype=torch.float32, device=self.device)
        for agent_idx, contact_sensor in enumerate(self._collision_sensors_by_agent):
            if contact_sensor is None:
                continue
            force_matrix_w = contact_sensor.data.force_matrix_w
            if force_matrix_w is None or force_matrix_w.numel() == 0:
                continue
            sensor_force = torch.linalg.vector_norm(force_matrix_w, dim=-1).amax(dim=(-1, -2))
            forces[agent_idx] = sensor_force
        warmup_steps = max(0, int(self.cfg.agent_collision_warmup_steps))
        if warmup_steps > 0:
            warmup_mask = self._steps_since_reset_buf < warmup_steps
            if bool(torch.any(warmup_mask).item()):
                forces[:, warmup_mask] = 0.0
        self._collision_force_cache = forces
        self._collision_force_cache_valid = True
        return forces

    def _collision_world_mask(self) -> torch.Tensor:
        if self._num_agents <= 1:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        max_force_per_world = self._collision_force_by_agent_n_tensor().amax(dim=0)
        return max_force_per_world >= float(self.cfg.agent_collision_force_threshold_n)

    def _collision_by_agent_mask_tensor(self) -> torch.Tensor:
        return self._collision_force_by_agent_n_tensor() >= float(self.cfg.agent_collision_force_threshold_n)

    def collision_force_by_agent_n(self) -> dict[str, torch.Tensor]:
        forces = self._collision_force_by_agent_n_tensor()
        return {agent_id: forces[idx].clone() for idx, agent_id in enumerate(self._agent_ids)}

    def collision_world_force_n(self) -> torch.Tensor:
        return self._collision_force_by_agent_n_tensor().amax(dim=0).clone()

    def _compute_yaw_by_agent(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        qw = root_quat_w[..., 0]
        qx = root_quat_w[..., 1]
        qy = root_quat_w[..., 2]
        qz = root_quat_w[..., 3]
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy.square() + qz.square())
        return torch.atan2(siny_cosp, cosy_cosp)

    def _planar_circle_centers_and_forward(
        self,
        root_pos_xy: torch.Tensor,
        yaw_by_agent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        circle_centers_xy_b = self._lane_touch_circle_centers_xy_b
        cos_yaw = torch.cos(yaw_by_agent)
        sin_yaw = torch.sin(yaw_by_agent)
        circle_x = circle_centers_xy_b[:, 0].view(1, 1, -1)
        circle_y = circle_centers_xy_b[:, 1].view(1, 1, -1)
        rotated_x = cos_yaw.unsqueeze(-1) * circle_x - sin_yaw.unsqueeze(-1) * circle_y
        rotated_y = sin_yaw.unsqueeze(-1) * circle_x + cos_yaw.unsqueeze(-1) * circle_y
        centers_w = torch.stack(
            (
                root_pos_xy[..., 0].unsqueeze(-1) + rotated_x,
                root_pos_xy[..., 1].unsqueeze(-1) + rotated_y,
            ),
            dim=-1,
        )
        forward_w = torch.stack((cos_yaw, sin_yaw), dim=-1)
        return centers_w, forward_w

    def _compute_pairwise_vehicle_ttc_s(
        self,
        root_pos_w: torch.Tensor,
        yaw_by_agent: torch.Tensor,
        root_lin_vel_w: torch.Tensor,
    ) -> torch.Tensor:
        if self._num_agents <= 1:
            return (
                torch.full((1, self.num_envs, 1), float("inf"), device=self.device),
                torch.zeros((1, self.num_envs), dtype=torch.float32, device=self.device),
            )

        env_origins_xy = self.scene.env_origins[:, :2]
        circle_radius = float(self._lane_touch_circle_radius_m)
        root_pos_xy = root_pos_w[..., :2] - env_origins_xy.unsqueeze(0)
        centers_w, forward_w = self._planar_circle_centers_and_forward(root_pos_xy, yaw_by_agent)
        velocities_xy = root_lin_vel_w[..., :2]

        combined_radius = 2.0 * circle_radius
        combined_radius_sq = float(combined_radius * combined_radius)
        centers_env = centers_w.permute(1, 0, 2, 3)
        forward_env = forward_w.permute(1, 0, 2)
        velocities_env = velocities_xy.permute(1, 0, 2)

        rel = centers_env[:, None, :, None, :, :] - centers_env[:, :, None, :, None, :]
        rel_vel = velocities_env[:, None, :, :] - velocities_env[:, :, None, :]

        rx = rel[..., 0]
        ry = rel[..., 1]
        rvx = rel_vel[..., 0].unsqueeze(-1).unsqueeze(-1)
        rvy = rel_vel[..., 1].unsqueeze(-1).unsqueeze(-1)
        dist_sq = rx.square() + ry.square()
        rel_speed_sq = rvx.square() + rvy.square()
        rdotv = rx * rvx + ry * rvy

        ego_forward_x = forward_env[:, :, 0].unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        ego_forward_y = forward_env[:, :, 1].unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        forward_dot = rx * ego_forward_x + ry * ego_forward_y
        forward_mask = forward_dot > 0.0

        pair_mask = ~torch.eye(self._num_agents, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        forward_mask = forward_mask & pair_mask
        overlap_mask = forward_mask & (dist_sq <= combined_radius_sq)
        t_pair = torch.full_like(dist_sq, float("inf"))
        t_pair = torch.where(overlap_mask, torch.zeros_like(t_pair), t_pair)

        moving_mask = forward_mask & (~overlap_mask) & (rel_speed_sq > 1.0e-6)
        a = torch.where(moving_mask, rel_speed_sq, torch.ones_like(rel_speed_sq))
        b = torch.where(moving_mask, 2.0 * rdotv, torch.zeros_like(rdotv))
        c = torch.where(moving_mask, dist_sq - combined_radius_sq, torch.zeros_like(dist_sq))
        disc = b.square() - 4.0 * a * c
        valid_quad = moving_mask & (disc >= 0.0)
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))
        denom = 2.0 * torch.clamp(a, min=1.0e-6)
        t_enter = (-b - sqrt_disc) / denom
        t_exit = (-b + sqrt_disc) / denom
        valid_enter = valid_quad & (t_exit >= 0.0)
        t_pair = torch.minimum(t_pair, torch.where(valid_enter, torch.clamp(t_enter, min=0.0), t_pair))

        unresolved = moving_mask & (~torch.isfinite(t_pair))
        dist = torch.sqrt(torch.clamp(dist_sq, min=1.0e-9))
        closing_speed = -rdotv / torch.clamp(dist, min=1.0e-6)
        clearance = torch.clamp(dist - combined_radius, min=0.0)
        valid_fb = unresolved & (rdotv < 0.0) & (closing_speed > 1.0e-6)
        fallback_t = torch.where(valid_fb, clearance / torch.clamp(closing_speed, min=1.0e-6), t_pair)
        t_pair = torch.minimum(t_pair, fallback_t)

        # DRAC: Deceleration Rate to Avoid Collision = closing_speed² / (2 * gap)
        # Only meaningful for approaching, non-self pairs; use 1.0 m floor on gap for stability.
        gap_drac = torch.clamp(dist - combined_radius, min=1.0)  # [envs, A, A, C, C]
        approach_speed = torch.clamp(closing_speed, min=0.0)      # [envs, A, A, C, C]
        drac_pair = torch.where(
            pair_mask & (approach_speed > 0.0),
            approach_speed.square() / (2.0 * gap_drac),
            torch.zeros_like(dist),
        )  # [envs, A, A, C, C]
        max_drac_per_pair = torch.amax(drac_pair, dim=(-1, -2))   # [envs, A, A]
        max_drac_env = torch.amax(max_drac_per_pair, dim=2)        # [envs, A]
        max_drac = max_drac_env.permute(1, 0)                      # [num_agents, num_envs]

        ttc_env = torch.amin(t_pair, dim=(-1, -2))
        return ttc_env.permute(1, 0, 2), max_drac

    def _build_reference_neighbor_context(
        self,
        agent_idx: int,
        root_pos_w: torch.Tensor,
        yaw_by_agent: torch.Tensor,
        speed_by_agent: torch.Tensor,
        pairwise_ttc_s: torch.Tensor | None,
    ) -> torch.Tensor:
        k = max(0, int(self.cfg.obs_neighbor_k))
        feat_dim = _reference_vehicle_feat_dim(
            self.cfg.obs_neighbor_include_ttc,
            self.cfg.obs_neighbor_include_index,
        )
        if (not bool(self.cfg.obs_neighbor_enable)) or k <= 0 or self._num_agents <= 1:
            return torch.zeros((self.num_envs, k * feat_dim), dtype=torch.float32, device=self.device)

        env_origins_xy = self.scene.env_origins[:, :2]
        root_pos_xy = root_pos_w[..., :2] - env_origins_xy.unsqueeze(0)
        dx = root_pos_xy[:, :, 0] - root_pos_xy[agent_idx, :, 0].unsqueeze(0)
        dy = root_pos_xy[:, :, 1] - root_pos_xy[agent_idx, :, 1].unsqueeze(0)
        relx_b, rely_b = _world_to_ego_xy_torch(dx, dy, yaw_by_agent[agent_idx].unsqueeze(0))
        distances_sq = dx.square() + dy.square()
        distances_sq = torch.where(
            self._agent_done_mask,
            torch.full_like(distances_sq, float("inf")),
            distances_sq,
        )
        distances_sq[agent_idx] = float("inf")
        agent_tie_break = (
            torch.arange(self._num_agents, dtype=torch.float32, device=self.device).unsqueeze(1) * 1.0e-4
        )
        sort_keys = distances_sq + agent_tie_break
        sorted_indices = torch.argsort(sort_keys.transpose(0, 1), dim=1)
        bounds_scale = float(max(1.0e-6, self._scene_factory_bounds_size_m))
        speed_scale = 10.0
        yaw_scale = math.pi
        ttc_max_s = float(max(0.1, self.cfg.obs_neighbor_ttc_max_s))
        obs = torch.zeros((self.num_envs, k, feat_dim), dtype=torch.float32, device=self.device)
        available_slots = min(k, max(0, self._num_agents - 1), int(sorted_indices.shape[1]))
        if available_slots <= 0:
            return obs.reshape(self.num_envs, -1)

        neighbor_idx = sorted_indices[:, :available_slots]
        relx_bt = relx_b.transpose(0, 1)
        rely_bt = rely_b.transpose(0, 1)
        yaw_by_env_agent = yaw_by_agent.transpose(0, 1)
        speed_by_env_agent = speed_by_agent.transpose(0, 1)
        obs_view = obs[:, :available_slots]

        obs_view[:, :, 0] = torch.gather(relx_bt, 1, neighbor_idx) / bounds_scale
        obs_view[:, :, 1] = torch.gather(rely_bt, 1, neighbor_idx) / bounds_scale
        obs_view[:, :, 2] = float(self._vehicle_length_m) / bounds_scale
        obs_view[:, :, 3] = float(self._vehicle_width_m) / bounds_scale

        ego_yaw = yaw_by_agent[agent_idx].unsqueeze(1)
        neighbor_yaw = torch.gather(yaw_by_env_agent, 1, neighbor_idx)
        obs_view[:, :, 4] = _wrap_pi_torch(neighbor_yaw - ego_yaw) / yaw_scale
        obs_view[:, :, 5] = torch.gather(speed_by_env_agent, 1, neighbor_idx) / speed_scale

        write_idx = 6
        if bool(self.cfg.obs_neighbor_include_ttc):
            pairwise_ttc_env = pairwise_ttc_s[agent_idx]
            ttc = torch.gather(pairwise_ttc_env, 1, neighbor_idx)
            ttc_n = torch.ones_like(ttc)
            finite_mask = torch.isfinite(ttc)
            ttc_n = torch.where(
                finite_mask,
                torch.clamp(ttc, min=0.0, max=ttc_max_s) / ttc_max_s,
                ttc_n,
            )
            obs_view[:, :, write_idx] = ttc_n
            write_idx += 1
        if bool(self.cfg.obs_neighbor_include_index):
            denom = float(max(1, self._num_agents - 1))
            obs_view[:, :, write_idx] = neighbor_idx.to(torch.float32) / denom
        return obs.reshape(self.num_envs, -1)

    def _build_reference_road_context(
        self,
        agent_idx: int,
        root_pos_w: torch.Tensor,
        yaw_by_agent: torch.Tensor,
    ) -> torch.Tensor:
        k = max(0, int(self.cfg.obs_road_points_k))
        feat_dim = _reference_road_point_feat_dim(self.cfg.obs_road_points_include_dirs)
        if (
            (not bool(self.cfg.obs_road_points_enable))
            or k <= 0
            or self._lane_touch_valid.numel() == 0
            or self._lane_touch_valid.shape[1] == 0
        ):
            return torch.zeros((self.num_envs, k * feat_dim), dtype=torch.float32, device=self.device)

        env_origins_xy = self.scene.env_origins[:, :2]
        root_pos_xy = root_pos_w[agent_idx, :, :2] - env_origins_xy
        dx_all = self._lane_touch_points_xy_m[..., 0] - root_pos_xy[:, 0].unsqueeze(1)
        dy_all = self._lane_touch_points_xy_m[..., 1] - root_pos_xy[:, 1].unsqueeze(1)
        dist_sq = dx_all.square() + dy_all.square()
        valid = self._lane_touch_valid
        radius_m = float(max(0.0, self.cfg.obs_road_points_radius_m))
        if radius_m > 0.0:
            valid = valid & (dist_sq <= float(radius_m * radius_m))
        mode = str(self.cfg.obs_road_points_mode).strip().lower().replace("_", "-")
        if mode == "knn":
            sort_keys = torch.where(valid, dist_sq, torch.full_like(dist_sq, float("inf")))
            sorted_indices = torch.argsort(sort_keys, dim=1)[:, :k]
        elif mode == "road-running":
            insertion_order = (
                torch.arange(self._lane_touch_valid.shape[1], dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .expand(self.num_envs, -1)
            )
            sort_keys = torch.where(valid, insertion_order, torch.full_like(insertion_order, float("inf")))
            sorted_indices = torch.argsort(sort_keys, dim=1)[:, :k]
        else:
            sort_keys = torch.where(valid, dist_sq, torch.full_like(dist_sq, float("inf")))
            sorted_indices = torch.argsort(sort_keys, dim=1)[:, :k]

        norm = float(radius_m if radius_m > 0.0 else max(1.0e-6, self._scene_factory_bounds_size_m))
        type_norm = float(self.cfg.obs_road_points_type_norm)
        obs = torch.zeros((self.num_envs, k, feat_dim), dtype=torch.float32, device=self.device)
        if sorted_indices.shape[1] == 0:
            return obs.reshape(self.num_envs, -1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        selected_dx = dx_all[env_ids.unsqueeze(1), sorted_indices]
        selected_dy = dy_all[env_ids.unsqueeze(1), sorted_indices]
        x_ego, y_ego = _world_to_ego_xy_torch(selected_dx, selected_dy, yaw_by_agent[agent_idx].unsqueeze(1))
        slot_count = int(sorted_indices.shape[1])
        obs[:, :slot_count, 0] = x_ego / norm
        obs[:, :slot_count, 1] = y_ego / norm
        types = self._lane_touch_types[env_ids.unsqueeze(1), sorted_indices].to(torch.float32)
        obs[:, :slot_count, 2] = types / type_norm if type_norm > 0.0 else types
        invalid = ~valid[env_ids.unsqueeze(1), sorted_indices]
        obs_view = obs[:, :slot_count]
        obs_view[invalid] = 0.0
        if feat_dim >= 5:
            dirs = self._lane_touch_dirs_xy[env_ids.unsqueeze(1), sorted_indices]
            dir_x_ego, dir_y_ego = _world_to_ego_xy_torch(dirs[..., 0], dirs[..., 1], yaw_by_agent[agent_idx].unsqueeze(1))
            obs_view[:, :, 3] = dir_x_ego
            obs_view[:, :, 4] = dir_y_ego
            obs_view[invalid] = 0.0
        return obs.reshape(self.num_envs, -1)

    def _nearest_neighbor_features(self, agent_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._num_agents <= 1:
            zeros = torch.zeros(self.num_envs, 3, device=self.device)
            return zeros, zeros, torch.zeros(self.num_envs, device=self.device)

        vehicle = self._vehicles[agent_idx]
        root_pos_w = vehicle.data.root_pos_w
        root_quat_w = vehicle.data.root_quat_w
        root_lin_vel_w = vehicle.data.root_lin_vel_w

        nearest_distance = torch.full((self.num_envs,), float("inf"), device=self.device)
        nearest_rel_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        nearest_rel_vel_b = torch.zeros(self.num_envs, 3, device=self.device)

        for other_idx, other_vehicle in enumerate(self._vehicles):
            if other_idx == agent_idx:
                continue
            rel_pos_b, _ = subtract_frame_transforms(
                root_pos_w,
                root_quat_w,
                other_vehicle.data.root_pos_w,
            )
            rel_vel_w = other_vehicle.data.root_lin_vel_w - root_lin_vel_w
            rel_vel_b = quat_apply_inverse(root_quat_w, rel_vel_w)
            distance = torch.linalg.norm(rel_pos_b[:, :2], dim=1)
            distance = torch.where(
                self._agent_done_mask[other_idx],
                torch.full_like(distance, float("inf")),
                distance,
            )
            update_mask = distance < nearest_distance
            nearest_distance = torch.where(update_mask, distance, nearest_distance)
            nearest_rel_pos_b = torch.where(update_mask.unsqueeze(-1), rel_pos_b, nearest_rel_pos_b)
            nearest_rel_vel_b = torch.where(update_mask.unsqueeze(-1), rel_vel_b, nearest_rel_vel_b)

        nearest_distance = torch.where(torch.isfinite(nearest_distance), nearest_distance, 0.0)
        return nearest_rel_pos_b, nearest_rel_vel_b, nearest_distance

    def _agent_crash_mask(self, agent_idx: int) -> torch.Tensor:
        vehicle = self._vehicles[agent_idx]
        root_pos_rel = vehicle.data.root_pos_w - self.scene.env_origins
        too_low = root_pos_rel[:, 2] < float(self.cfg.fall_height_threshold_m)
        too_far = torch.linalg.norm(root_pos_rel[:, :2], dim=1) > float(self.cfg.max_distance_from_origin_m)
        bad_tilt = vehicle.data.projected_gravity_b[:, 2] > float(self.cfg.bad_tilt_gravity_threshold)
        return too_low | too_far | bad_tilt

    def _agent_crash_components_all(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_pos_w = torch.stack([vehicle.data.root_pos_w for vehicle in self._vehicles], dim=0)
        projected_gravity_b = torch.stack([vehicle.data.projected_gravity_b for vehicle in self._vehicles], dim=0)
        root_pos_rel = root_pos_w - self.scene.env_origins.unsqueeze(0)
        too_low = root_pos_rel[..., 2] < float(self.cfg.fall_height_threshold_m)
        too_far = torch.linalg.norm(root_pos_rel[..., :2], dim=-1) > float(self.cfg.max_distance_from_origin_m)
        bad_tilt = projected_gravity_b[..., 2] > float(self.cfg.bad_tilt_gravity_threshold)
        return too_low, too_far, bad_tilt

    def _agent_crash_mask_all(self) -> torch.Tensor:
        too_low, too_far, bad_tilt = self._agent_crash_components_all()
        return too_low | too_far | bad_tilt

    def _lane_touch_any_type_mask_all(self, lane_types: Sequence[int]) -> torch.Tensor:
        if (
            not bool(self.cfg.lane_touch_enabled)
            or self._lane_touch_mask.numel() == 0
            or self._lane_touch_mask.shape[-1] <= 0
        ):
            return torch.zeros((self._num_agents, self.num_envs), dtype=torch.bool, device=self.device)
        valid_types = []
        type_dim = int(self._lane_touch_type_dim)
        for road_type in lane_types:
            road_type_int = int(road_type)
            if 0 <= road_type_int < type_dim:
                valid_types.append(road_type_int)
        if not valid_types:
            return torch.zeros((self._num_agents, self.num_envs), dtype=torch.bool, device=self.device)
        return torch.any(self._lane_touch_mask[:, :, valid_types], dim=2)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._sync_timing_device()
        obs_start = perf_counter()

        self._sync_timing_device()
        obs_lane_start = perf_counter()
        self._update_lane_touch_mask()
        self._sync_timing_device()
        obs_lane_ms = (perf_counter() - obs_lane_start) * 1000.0

        observations = {}
        observation_mode = str(self.cfg.observation_mode).strip().lower()
        neighbor_obs_scale = float(max(1.0e-6, self.cfg.agent_neighbor_obs_scale_m))
        goal_obs_scale = float(max(1.0e-6, self.cfg.goal_radius_max_m))
        obs_shared_ms = 0.0
        obs_shared_stack_ms = 0.0
        obs_shared_yaw_speed_ms = 0.0
        obs_shared_ttc_ms = 0.0
        if observation_mode == "choco_reference":
            self._sync_timing_device()
            obs_shared_start = perf_counter()

            self._sync_timing_device()
            obs_shared_stack_start = perf_counter()
            root_pos_w = torch.stack([vehicle.data.root_pos_w for vehicle in self._vehicles], dim=0)
            root_quat_w = torch.stack([vehicle.data.root_quat_w for vehicle in self._vehicles], dim=0)
            root_lin_vel_w = torch.stack([vehicle.data.root_lin_vel_w for vehicle in self._vehicles], dim=0)
            self._sync_timing_device()
            obs_shared_stack_ms = (perf_counter() - obs_shared_stack_start) * 1000.0

            self._sync_timing_device()
            obs_shared_yaw_speed_start = perf_counter()
            yaw_by_agent = self._compute_yaw_by_agent(root_quat_w)
            speed_by_agent = torch.linalg.norm(root_lin_vel_w[..., :2], dim=-1)
            self._sync_timing_device()
            obs_shared_yaw_speed_ms = (perf_counter() - obs_shared_yaw_speed_start) * 1000.0

            self._sync_timing_device()
            obs_shared_ttc_start = perf_counter()
            pairwise_ttc_s = (
                self._compute_pairwise_vehicle_ttc_s(root_pos_w, yaw_by_agent, root_lin_vel_w)[0]
                if bool(self.cfg.obs_neighbor_enable and self.cfg.obs_neighbor_include_ttc)
                else None
            )
            self._sync_timing_device()
            obs_shared_ttc_ms = (perf_counter() - obs_shared_ttc_start) * 1000.0
            self._sync_timing_device()
            obs_shared_ms = (perf_counter() - obs_shared_start) * 1000.0
        obs_goal_ms = 0.0
        obs_road_ms = 0.0
        obs_neighbor_ms = 0.0
        obs_finalize_ms = 0.0
        for agent_idx, agent_id in enumerate(self._agent_ids):
            self._sync_timing_device()
            obs_goal_start = perf_counter()
            goal_pos_b, goal_distance = self._compute_goal_distance(agent_idx)
            self._current_goal_distance[agent_idx] = goal_distance
            self._sync_timing_device()
            obs_goal_ms += (perf_counter() - obs_goal_start) * 1000.0
            if observation_mode == "goal_reaching":
                obs = torch.cat(
                    [
                        goal_pos_b[:, :2] / goal_obs_scale,
                        goal_distance.unsqueeze(-1) / goal_obs_scale,
                        self._vehicles[agent_idx].data.root_lin_vel_b[:, :2] / 10.0,
                        self._vehicles[agent_idx].data.root_ang_vel_b[:, 2:3] / 10.0,
                    ],
                    dim=-1,
                )
            elif observation_mode == "full":
                nearest_rel_pos_b, nearest_rel_vel_b, nearest_distance = self._nearest_neighbor_features(agent_idx)
                obs = torch.cat(
                    [
                        goal_pos_b[:, :2] / goal_obs_scale,
                        goal_distance.unsqueeze(-1) / goal_obs_scale,
                        self._vehicles[agent_idx].data.root_lin_vel_b[:, :2] / 10.0,
                        self._vehicles[agent_idx].data.root_ang_vel_b[:, 2:3] / 10.0,
                        self._vehicles[agent_idx].data.projected_gravity_b[:, :2],
                        self._vehicles[agent_idx].data.joint_pos[:, self._steer_joint_ids[agent_idx]]
                        / float(max(1.0e-6, self._steer_limit)),
                        self._vehicles[agent_idx].data.joint_vel[:, self._wheel_joint_ids[agent_idx]] / 50.0,
                        self._raw_actions[agent_idx],
                        nearest_rel_pos_b[:, :2] / neighbor_obs_scale,
                        nearest_distance.unsqueeze(-1) / neighbor_obs_scale,
                        nearest_rel_vel_b[:, :2] / 10.0,
                    ],
                    dim=-1,
                )
            elif observation_mode == "choco_reference":
                heading_error = torch.atan2(goal_pos_b[:, 1], goal_pos_b[:, 0])
                bounds_scale = float(max(1.0e-6, self._scene_factory_bounds_size_m))
                distance_scale = float(max(1.0e-6, self._scene_factory_bounds_size_m * math.sqrt(2.0)))
                obs_parts = [
                    goal_pos_b[:, 0:1] / bounds_scale,
                    goal_pos_b[:, 1:2] / bounds_scale,
                    torch.sin(heading_error).unsqueeze(-1),
                    torch.cos(heading_error).unsqueeze(-1),
                    goal_distance.unsqueeze(-1) / distance_scale,
                    self._vehicles[agent_idx].data.root_lin_vel_b[:, :2] / 10.0,
                ]
                if bool(self.cfg.obs_weather_context_enable):
                    if bool(self.cfg.obs_weather_context_blind):
                        obs_parts.append(self._weather_context_blind_const)
                    else:
                        obs_parts.append(self._weather_context)
                if bool(self.cfg.obs_road_points_enable):
                    self._sync_timing_device()
                    obs_road_start = perf_counter()
                    obs_parts.append(self._build_reference_road_context(agent_idx, root_pos_w, yaw_by_agent))
                    self._sync_timing_device()
                    obs_road_ms += (perf_counter() - obs_road_start) * 1000.0
                if bool(self.cfg.obs_neighbor_enable):
                    self._sync_timing_device()
                    obs_neighbor_start = perf_counter()
                    obs_parts.append(
                        self._build_reference_neighbor_context(
                            agent_idx,
                            root_pos_w,
                            yaw_by_agent,
                            speed_by_agent,
                            pairwise_ttc_s,
                        )
                    )
                    self._sync_timing_device()
                    obs_neighbor_ms += (perf_counter() - obs_neighbor_start) * 1000.0
                obs = torch.cat(obs_parts, dim=-1)
            else:
                raise ValueError(f"Unsupported observation mode: {self.cfg.observation_mode!r}")
            self._sync_timing_device()
            obs_finalize_start = perf_counter()
            done_mask = self._agent_done_mask[agent_idx]
            if bool(torch.any(done_mask).item()):
                obs = obs.clone()
                obs[done_mask] = 0.0
            observations[agent_id] = obs
            self._sync_timing_device()
            obs_finalize_ms += (perf_counter() - obs_finalize_start) * 1000.0
        self._sync_timing_device()
        elapsed_ms = (perf_counter() - obs_start) * 1000.0
        self._obs_timing_call_count += 1
        self._obs_timing_last_ms = float(elapsed_ms)
        if self._obs_timing_call_count == 1:
            self._obs_timing_ema_ms = float(elapsed_ms)
        else:
            self._obs_timing_ema_ms = 0.9 * float(self._obs_timing_ema_ms) + 0.1 * float(elapsed_ms)
        if bool(self.cfg.obs_timing_print_enable):
            every_n = max(1, int(self.cfg.obs_timing_print_every_n))
            if self._obs_timing_call_count % every_n == 0:
                print(
                    "[INFO][SceneFactory][ObsTiming] "
                    f"call={self._obs_timing_call_count} mode={observation_mode} "
                    f"last_ms={self._obs_timing_last_ms:.2f} ema_ms={self._obs_timing_ema_ms:.2f} "
                    f"envs={self.num_envs} agents={self._num_agents}",
                    flush=True,
                )
        self._step_timing_last_ms["obs_ms"] = float(elapsed_ms)
        self._step_timing_last_ms["obs_lane_ms"] = float(obs_lane_ms)
        self._step_timing_last_ms["obs_shared_ms"] = float(obs_shared_ms)
        self._step_timing_last_ms["obs_shared_stack_ms"] = float(obs_shared_stack_ms)
        self._step_timing_last_ms["obs_shared_yaw_speed_ms"] = float(obs_shared_yaw_speed_ms)
        self._step_timing_last_ms["obs_shared_ttc_ms"] = float(obs_shared_ttc_ms)
        self._step_timing_last_ms["obs_goal_ms"] = float(obs_goal_ms)
        self._step_timing_last_ms["obs_road_ms"] = float(obs_road_ms)
        self._step_timing_last_ms["obs_neighbor_ms"] = float(obs_neighbor_ms)
        self._step_timing_last_ms["obs_finalize_ms"] = float(obs_finalize_ms)
        return observations

    def _choco_ttc_abs_penalty_from_min_ttc(self, min_ttc: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(min_ttc, min=float(max(1.0e-3, self.cfg.reward_choco_ttc_penalty_min_ttc)))
        penalty = float(self.cfg.reward_choco_ttc_penalty_alpha) / denom
        penalty = torch.where(
            min_ttc < 0.5,
            torch.full_like(min_ttc, float(self.cfg.reward_choco_ttc_penalty_max)),
            penalty,
        )
        return torch.clamp(penalty, max=float(self.cfg.reward_choco_ttc_penalty_max))

    def _choco_road_edge_ttc_abs_penalty_from_min_ttc(self, min_ttc: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(min_ttc, min=float(max(1.0e-3, self.cfg.reward_choco_road_edge_ttc_penalty_min_ttc)))
        penalty = float(self.cfg.reward_choco_road_edge_ttc_penalty_alpha) / denom
        penalty = torch.where(
            min_ttc < float(self.cfg.reward_choco_road_edge_ttc_hard_min_ttc),
            torch.full_like(min_ttc, float(self.cfg.reward_choco_road_edge_ttc_penalty_max)),
            penalty,
        )
        return torch.clamp(penalty, max=float(self.cfg.reward_choco_road_edge_ttc_penalty_max))

    def _compute_choco_geometric_lane_features(
        self,
        agent_idx: int,
        goal_pos_b: torch.Tensor,
        *,
        root_pos_xy: torch.Tensor | None = None,
        yaw: torch.Tensor | None = None,
        lane_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        false_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if (
            self._lane_touch_valid.numel() == 0
            or self._lane_touch_valid.shape[1] == 0
            or not bool(torch.any(self._lane_touch_valid).item())
        ):
            return zeros, false_mask, zeros, zeros

        if lane_mask is None:
            lane_types = torch.tensor(self.cfg.reward_choco_geom_lane_types, dtype=torch.long, device=self.device)
            lane_mask = self._lane_touch_valid & torch.isin(self._lane_touch_types, lane_types)
        if not bool(torch.any(lane_mask).item()):
            return zeros, false_mask, zeros, zeros

        if root_pos_xy is None:
            vehicle = self._vehicles[agent_idx]
            root_pos_xy = vehicle.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        if yaw is None:
            vehicle = self._vehicles[agent_idx]
            yaw = self._compute_yaw_by_agent(vehicle.data.root_quat_w.unsqueeze(0))[0]

        deltas = self._lane_touch_points_xy_m - root_pos_xy.unsqueeze(1)
        dist_sq = torch.sum(deltas.square(), dim=-1)
        masked_dist_sq = torch.where(lane_mask, dist_sq, torch.full_like(dist_sq, float("inf")))
        nearest_idx = torch.argmin(masked_dist_sq, dim=1)
        env_ids = torch.arange(self.num_envs, device=self.device)
        best_dist_sq = masked_dist_sq[env_ids, nearest_idx]
        has_lane = torch.isfinite(best_dist_sq)

        nearest_point = self._lane_touch_points_xy_m[env_ids, nearest_idx]
        tangent = self._lane_touch_dirs_xy[env_ids, nearest_idx]
        tangent = tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp_min(1.0e-6)

        goal_dx_w, goal_dy_w = _ego_to_world_xy_torch(goal_pos_b[:, 0], goal_pos_b[:, 1], yaw)
        goal_dir_w = torch.stack((goal_dx_w, goal_dy_w), dim=-1)
        tangent = torch.where(torch.sum(goal_dir_w * tangent, dim=-1, keepdim=True) < 0.0, -tangent, tangent)

        normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=-1)
        rel = root_pos_xy - nearest_point
        lateral = torch.abs(torch.sum(rel * normal, dim=-1))
        nearest_dist = torch.sqrt(torch.clamp(best_dist_sq, min=0.0))
        tangent_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
        align = torch.clamp(torch.cos(_wrap_pi_torch(yaw - tangent_yaw)), min=0.0)
        tol = float(max(1.0e-6, self.cfg.reward_choco_geom_lane_tolerance_m))
        heading_weight = float(self.cfg.reward_choco_geom_lane_heading_weight)
        quality = torch.exp(-torch.square(lateral / tol))
        quality = quality * ((1.0 - heading_weight) + heading_weight * align)
        quality = torch.where(has_lane, quality, zeros)

        lane_hit = has_lane & (lateral <= tol) & (align >= float(self.cfg.reward_choco_geom_lane_min_alignment))
        offroad = has_lane & (
            (lateral > float(self.cfg.reward_choco_geom_offroad_lateral_threshold_m))
            | (nearest_dist > float(self.cfg.reward_choco_geom_offroad_distance_threshold_m))
        )
        return quality, offroad, lateral, align

    def _compute_choco_geometric_lane_features_all(
        self,
        goal_pos_b: torch.Tensor,
        *,
        root_pos_xy: torch.Tensor,
        yaw: torch.Tensor,
        lane_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros((self._num_agents, self.num_envs), dtype=torch.float32, device=self.device)
        false_mask = torch.zeros((self._num_agents, self.num_envs), dtype=torch.bool, device=self.device)
        if (
            self._lane_touch_valid.numel() == 0
            or self._lane_touch_valid.shape[1] == 0
            or not bool(torch.any(self._lane_touch_valid).item())
        ):
            return zeros, false_mask, zeros, zeros, zeros

        if lane_mask is None:
            lane_types = torch.tensor(self.cfg.reward_choco_geom_lane_types, dtype=torch.long, device=self.device)
            lane_mask = self._lane_touch_valid & torch.isin(self._lane_touch_types, lane_types)
        if not bool(torch.any(lane_mask).item()):
            return zeros, false_mask, zeros, zeros, zeros

        deltas = self._lane_touch_points_xy_m.unsqueeze(0) - root_pos_xy.unsqueeze(2)
        dist_sq = torch.sum(deltas.square(), dim=-1)
        masked_dist_sq = torch.where(lane_mask.unsqueeze(0), dist_sq, torch.full_like(dist_sq, float("inf")))
        nearest_idx = torch.argmin(masked_dist_sq, dim=2)
        gather_idx_xy = nearest_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)
        nearest_point = torch.gather(
            self._lane_touch_points_xy_m.unsqueeze(0).expand(self._num_agents, -1, -1, -1),
            2,
            gather_idx_xy,
        ).squeeze(2)
        tangent = torch.gather(
            self._lane_touch_dirs_xy.unsqueeze(0).expand(self._num_agents, -1, -1, -1),
            2,
            gather_idx_xy,
        ).squeeze(2)
        gather_idx_scalar = nearest_idx.unsqueeze(-1)
        best_dist_sq = torch.gather(masked_dist_sq, 2, gather_idx_scalar).squeeze(-1)
        has_lane = torch.isfinite(best_dist_sq)

        tangent = tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp_min(1.0e-6)
        goal_dx_w, goal_dy_w = _ego_to_world_xy_torch(goal_pos_b[..., 0], goal_pos_b[..., 1], yaw)
        goal_dir_w = torch.stack((goal_dx_w, goal_dy_w), dim=-1)
        tangent = torch.where(torch.sum(goal_dir_w * tangent, dim=-1, keepdim=True) < 0.0, -tangent, tangent)

        normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
        rel = root_pos_xy - nearest_point
        lateral = torch.abs(torch.sum(rel * normal, dim=-1))
        nearest_dist = torch.sqrt(torch.clamp(best_dist_sq, min=0.0))
        tangent_yaw = torch.atan2(tangent[..., 1], tangent[..., 0])
        align = torch.clamp(torch.cos(_wrap_pi_torch(yaw - tangent_yaw)), min=0.0)
        tol = float(max(1.0e-6, self.cfg.reward_choco_geom_lane_tolerance_m))
        heading_weight = float(self.cfg.reward_choco_geom_lane_heading_weight)
        quality = torch.exp(-torch.square(lateral / tol))
        quality = quality * ((1.0 - heading_weight) + heading_weight * align)
        quality = torch.where(has_lane, quality, zeros)
        route_progress = torch.sum((root_pos_xy - self._previous_root_pos_xy) * tangent, dim=-1)
        route_progress = torch.where(has_lane, route_progress, zeros)
        offroad = has_lane & (
            (lateral > float(self.cfg.reward_choco_geom_offroad_lateral_threshold_m))
            | (nearest_dist > float(self.cfg.reward_choco_geom_offroad_distance_threshold_m))
        )
        return quality, offroad, lateral, align, route_progress

    def _compute_choco_road_edge_ttc_penalty(
        self,
        agent_idx: int,
        *,
        root_pos_xy: torch.Tensor | None = None,
        yaw: torch.Tensor | None = None,
        root_lin_vel_xy: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        if not bool(self.cfg.reward_choco_road_edge_ttc_penalty_enable):
            return zeros

        if edge_mask is None:
            edge_types = torch.tensor(self.cfg.reward_choco_geom_road_edge_types, dtype=torch.long, device=self.device)
            edge_mask = self._lane_touch_valid & torch.isin(self._lane_touch_types, edge_types)
        if not bool(torch.any(edge_mask).item()):
            return zeros

        radius_m = float(self.cfg.reward_choco_road_edge_ttc_radius_m)
        if radius_m <= 0.0:
            return zeros

        if root_pos_xy is None:
            vehicle = self._vehicles[agent_idx]
            root_pos_xy = vehicle.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
        if root_lin_vel_xy is None:
            vehicle = self._vehicles[agent_idx]
            root_lin_vel_xy = vehicle.data.root_lin_vel_w[:, :2]
        if yaw is None:
            vehicle = self._vehicles[agent_idx]
            yaw = self._compute_yaw_by_agent(vehicle.data.root_quat_w.unsqueeze(0))[0]
        circle_centers_w, forward = self._planar_circle_centers_and_forward(root_pos_xy.unsqueeze(0), yaw.unsqueeze(0))
        circle_centers_w = circle_centers_w[0]
        forward = forward[0]

        rel = self._lane_touch_points_xy_m.unsqueeze(1) - circle_centers_w.unsqueeze(2)
        dist_sq = torch.sum(rel.square(), dim=-1)
        near_mask = edge_mask.unsqueeze(1) & (dist_sq <= float(radius_m * radius_m))
        if not bool(torch.any(near_mask).item()):
            return zeros

        forward_dot = rel[..., 0] * forward[:, 0].unsqueeze(1).unsqueeze(1) + rel[..., 1] * forward[:, 1].unsqueeze(1).unsqueeze(1)
        candidate_mask = near_mask & (forward_dot > 0.0)
        dist = torch.sqrt(torch.clamp(dist_sq, min=1.0e-12))
        clearance = torch.clamp(dist - float(self._lane_touch_circle_radius_m), min=0.0)
        dirs = rel / torch.clamp(dist.unsqueeze(-1), min=1.0e-6)
        closing_speed = dirs[..., 0] * root_lin_vel_xy[:, 0].unsqueeze(1).unsqueeze(1) + dirs[..., 1] * root_lin_vel_xy[:, 1].unsqueeze(1).unsqueeze(1)
        ttc = torch.full_like(dist, float("inf"))
        ttc = torch.where(candidate_mask & (dist <= float(self._lane_touch_circle_radius_m)), torch.zeros_like(ttc), ttc)
        valid = candidate_mask & (closing_speed > 1.0e-6)
        ttc = torch.where(valid, torch.minimum(ttc, clearance / torch.clamp(closing_speed, min=1.0e-6)), ttc)
        finite_mask = torch.isfinite(ttc)
        if not bool(torch.any(finite_mask).item()):
            return zeros
        flat_ttc = torch.where(finite_mask, ttc, torch.full_like(ttc, float("inf"))).reshape(self.num_envs, -1)
        min_ttc = torch.amin(flat_ttc, dim=1)
        valid_env = torch.isfinite(min_ttc)
        return torch.where(valid_env, -self._choco_road_edge_ttc_abs_penalty_from_min_ttc(min_ttc), zeros)

    def _compute_choco_road_edge_ttc_penalty_all(
        self,
        *,
        root_pos_xy: torch.Tensor,
        yaw: torch.Tensor,
        root_lin_vel_xy: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        zeros = torch.zeros((self._num_agents, self.num_envs), dtype=torch.float32, device=self.device)
        if not bool(self.cfg.reward_choco_road_edge_ttc_penalty_enable):
            return zeros

        if edge_mask is None:
            edge_types = torch.tensor(self.cfg.reward_choco_geom_road_edge_types, dtype=torch.long, device=self.device)
            edge_mask = self._lane_touch_valid & torch.isin(self._lane_touch_types, edge_types)
        if not bool(torch.any(edge_mask).item()):
            return zeros

        radius_m = float(self.cfg.reward_choco_road_edge_ttc_radius_m)
        if radius_m <= 0.0:
            return zeros

        circle_centers_w, forward = self._planar_circle_centers_and_forward(root_pos_xy, yaw)
        rel = self._lane_touch_points_xy_m.unsqueeze(0).unsqueeze(2) - circle_centers_w.unsqueeze(3)
        dist_sq = torch.sum(rel.square(), dim=-1)
        near_mask = edge_mask.unsqueeze(0).unsqueeze(2) & (dist_sq <= float(radius_m * radius_m))
        if not bool(torch.any(near_mask).item()):
            return zeros

        forward_dot = (
            rel[..., 0] * forward[..., 0].unsqueeze(-1).unsqueeze(-1)
            + rel[..., 1] * forward[..., 1].unsqueeze(-1).unsqueeze(-1)
        )
        candidate_mask = near_mask & (forward_dot > 0.0)
        dist = torch.sqrt(torch.clamp(dist_sq, min=1.0e-12))
        clearance = torch.clamp(dist - float(self._lane_touch_circle_radius_m), min=0.0)
        dirs = rel / torch.clamp(dist.unsqueeze(-1), min=1.0e-6)
        closing_speed = (
            dirs[..., 0] * root_lin_vel_xy[..., 0].unsqueeze(-1).unsqueeze(-1)
            + dirs[..., 1] * root_lin_vel_xy[..., 1].unsqueeze(-1).unsqueeze(-1)
        )
        ttc = torch.full_like(dist, float("inf"))
        ttc = torch.where(candidate_mask & (dist <= float(self._lane_touch_circle_radius_m)), torch.zeros_like(ttc), ttc)
        valid = candidate_mask & (closing_speed > 1.0e-6)
        ttc = torch.where(valid, torch.minimum(ttc, clearance / torch.clamp(closing_speed, min=1.0e-6)), ttc)
        finite_mask = torch.isfinite(ttc)
        if not bool(torch.any(finite_mask).item()):
            return zeros
        flat_ttc = torch.where(finite_mask, ttc, torch.full_like(ttc, float("inf"))).reshape(self._num_agents, self.num_envs, -1)
        min_ttc = torch.amin(flat_ttc, dim=-1)
        valid_env = torch.isfinite(min_ttc)
        penalties = -self._choco_road_edge_ttc_abs_penalty_from_min_ttc(min_ttc)
        return torch.where(valid_env, penalties, zeros)

    def _get_rewards_choco_aligned(self) -> dict[str, torch.Tensor]:
        rewards = {}

        self._sync_timing_device()
        reward_shared_start = perf_counter()
        env_origins_xy = self.scene.env_origins[:, :2]
        root_pos_w = torch.stack([vehicle.data.root_pos_w for vehicle in self._vehicles], dim=0)
        root_quat_w = torch.stack([vehicle.data.root_quat_w for vehicle in self._vehicles], dim=0)
        root_lin_vel_w = torch.stack([vehicle.data.root_lin_vel_w for vehicle in self._vehicles], dim=0)
        root_pos_xy = root_pos_w[..., :2] - env_origins_xy.unsqueeze(0)
        root_lin_vel_xy = root_lin_vel_w[..., :2]
        goal_pos_b_all, goal_distance_all = self._compute_goal_distance_all()
        yaw_by_agent = self._compute_yaw_by_agent(root_quat_w)
        pairwise_ttc_s, max_drac_by_agent = self._compute_pairwise_vehicle_ttc_s(root_pos_w, yaw_by_agent, root_lin_vel_w)
        geom_lane_types = torch.tensor(self.cfg.reward_choco_geom_lane_types, dtype=torch.long, device=self.device)
        geom_lane_mask = self._lane_touch_valid & torch.isin(self._lane_touch_types, geom_lane_types)
        road_edge_types = torch.tensor(self.cfg.reward_choco_geom_road_edge_types, dtype=torch.long, device=self.device)
        road_edge_mask = self._lane_touch_valid & torch.isin(self._lane_touch_types, road_edge_types)
        self._sync_timing_device()
        reward_shared_ms = (perf_counter() - reward_shared_start) * 1000.0

        reward_goal_ms = 0.0
        reward_geom_ms = 0.0
        reward_route_progress_ms = 0.0
        reward_ttc_ms = 0.0
        reward_road_edge_ttc_ms = 0.0
        reward_finalize_ms = 0.0

        self._sync_timing_device()
        reward_geom_start = perf_counter()
        lane_quality_all, offroad_mask_all, _lane_error_all, _heading_alignment_all, route_progress_all = self._compute_choco_geometric_lane_features_all(
            goal_pos_b_all,
            root_pos_xy=root_pos_xy,
            yaw=yaw_by_agent,
            lane_mask=geom_lane_mask,
        )
        self._sync_timing_device()
        reward_geom_ms = (perf_counter() - reward_geom_start) * 1000.0

        self._sync_timing_device()
        reward_road_edge_ttc_start = perf_counter()
        road_edge_ttc_penalty_all = self._compute_choco_road_edge_ttc_penalty_all(
            root_pos_xy=root_pos_xy,
            yaw=yaw_by_agent,
            root_lin_vel_xy=root_lin_vel_xy,
            edge_mask=road_edge_mask,
        )
        self._sync_timing_device()
        reward_road_edge_ttc_ms = (perf_counter() - reward_road_edge_ttc_start) * 1000.0

        for agent_idx, agent_id in enumerate(self._agent_ids):
            self._sync_timing_device()
            reward_goal_start = perf_counter()
            goal_pos_b = goal_pos_b_all[agent_idx]
            goal_distance = goal_distance_all[agent_idx]
            self._current_goal_distance[agent_idx] = goal_distance
            active_mask = ~self._agent_done_mask[agent_idx]
            active_float = active_mask.float()
            planar_speed = torch.linalg.norm(root_lin_vel_xy[agent_idx], dim=1)
            self._sync_timing_device()
            reward_goal_ms += (perf_counter() - reward_goal_start) * 1000.0

            success_bonus = self._pending_goal_done_mask[agent_idx].float() * float(self.cfg.reward_goal_bonus)
            new_collision_event = self._pending_collision_done_mask[agent_idx] & (~self._collision_done_mask[agent_idx])
            new_crash_event = self._pending_crash_done_mask[agent_idx] & (~self._crash_done_mask[agent_idx])
            new_crash_too_low_event = self._pending_crash_too_low_mask[agent_idx] & (~self._crash_too_low_done_mask[agent_idx])
            new_crash_too_far_event = self._pending_crash_too_far_mask[agent_idx] & (~self._crash_too_far_done_mask[agent_idx])
            new_crash_bad_tilt_event = self._pending_crash_bad_tilt_mask[agent_idx] & (~self._crash_bad_tilt_done_mask[agent_idx])
            new_lane_forbidden_event = self._pending_lane_forbidden_done_mask[agent_idx] & (~self._lane_forbidden_done_mask[agent_idx])
            collision_penalty = new_collision_event.float() * float(self.cfg.reward_collision_penalty)
            crash_penalty = new_crash_event.float() * float(self.cfg.reward_crash_penalty)
            lane_forbidden_penalty = new_lane_forbidden_event.float() * float(self.cfg.reward_lane_forbidden_penalty)

            idle_penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            if bool(self.cfg.reward_choco_idle_penalty_enable):
                idle_penalty = -(
                    active_mask & (planar_speed < float(self.cfg.reward_choco_idle_speed_threshold_mps))
                ).float() * float(self.cfg.reward_choco_idle_penalty_per_step)

            speed_bonus = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            if bool(self.cfg.reward_choco_speed_bonus_enable):
                max_speed = float(self.cfg.reward_choco_speed_bonus_max_mps)
                speed_bonus = (
                    torch.clamp(planar_speed, min=0.0, max=max_speed) / max_speed
                    * float(self.cfg.reward_choco_speed_bonus_per_step)
                    * active_float
                )

            lane_quality = lane_quality_all[agent_idx]
            offroad_mask = offroad_mask_all[agent_idx]
            geom_lane_reward = (
                lane_quality * float(self.cfg.reward_choco_geom_lane_per_step) * active_float
                if bool(self.cfg.reward_choco_geom_lane_enable)
                else torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            )
            self._sync_timing_device()
            reward_route_progress_start = perf_counter()
            geom_route_progress = (
                torch.clamp(route_progress_all[agent_idx], min=-2.0, max=2.0)
                * float(self.cfg.reward_choco_geom_route_progress_weight)
                * active_float
            )
            self._sync_timing_device()
            reward_route_progress_ms += (perf_counter() - reward_route_progress_start) * 1000.0
            offroad_penalty = (
                offroad_mask.float() * float(self.cfg.reward_choco_offroad_penalty) * active_float
                if bool(self.cfg.reward_choco_geom_offroad_enable)
                else torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            )

            # Compute min-TTC unconditionally (used for both penalty and safety metrics)
            min_ttc = torch.full((self.num_envs,), float("inf"), device=self.device)
            if self._num_agents > 1:
                ego_ttc = pairwise_ttc_s[agent_idx]
                other_active = (~self._agent_done_mask).transpose(0, 1)
                finite_mask = torch.isfinite(ego_ttc) & other_active
                finite_mask[:, agent_idx] = False
                min_ttc = torch.amin(
                    torch.where(finite_mask, ego_ttc, torch.full_like(ego_ttc, float("inf"))),
                    dim=1,
                )

            ttc_penalty = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self._sync_timing_device()
            reward_ttc_start = perf_counter()
            if bool(self.cfg.reward_choco_ttc_penalty_enable) and self._num_agents > 1:
                valid_env = torch.isfinite(min_ttc)
                ttc_penalty = torch.where(valid_env, -self._choco_ttc_abs_penalty_from_min_ttc(min_ttc), ttc_penalty) * active_float
            self._sync_timing_device()
            reward_ttc_ms += (perf_counter() - reward_ttc_start) * 1000.0

            # Update per-episode TTC / DRAC buffers (always, regardless of penalty enable)
            valid_ttc = torch.isfinite(min_ttc)
            _ttc_clamped = torch.clamp(min_ttc, max=float(self.cfg.obs_neighbor_ttc_max_s))  # cap at tau_max (e.g. 10 s)
            self._episode_min_ttc_sum[agent_idx] += torch.where(valid_ttc, _ttc_clamped, torch.zeros_like(min_ttc))
            self._episode_ttc_finite_steps[agent_idx] += valid_ttc.float()
            self._episode_near_miss_steps[agent_idx] += (valid_ttc & (min_ttc < _TTC_NEAR_MISS_THRESHOLD_S)).float()
            self._episode_max_drac[agent_idx] = torch.maximum(
                self._episode_max_drac[agent_idx], max_drac_by_agent[agent_idx]
            )

            road_edge_ttc_penalty = road_edge_ttc_penalty_all[agent_idx] * active_float

            self._sync_timing_device()
            reward_finalize_start = perf_counter()
            rewards[agent_id] = (
                success_bonus
                + collision_penalty
                + crash_penalty
                + lane_forbidden_penalty
                + offroad_penalty
                + idle_penalty
                + speed_bonus
                + ttc_penalty
                + road_edge_ttc_penalty
                + geom_lane_reward
                + geom_route_progress
            )

            self._episode_sums["goal_bonus"][agent_idx] += success_bonus
            self._episode_sums["collision_penalty"][agent_idx] += collision_penalty
            self._episode_sums["crash_penalty"][agent_idx] += crash_penalty
            self._episode_sums["lane_forbidden_penalty"][agent_idx] += lane_forbidden_penalty
            self._episode_sums["offroad_penalty"][agent_idx] += offroad_penalty
            self._episode_sums["idle_penalty"][agent_idx] += idle_penalty
            self._episode_sums["speed_bonus"][agent_idx] += speed_bonus
            self._episode_sums["ttc_penalty"][agent_idx] += ttc_penalty
            self._episode_sums["road_edge_ttc_penalty"][agent_idx] += road_edge_ttc_penalty
            self._episode_sums["geom_lane_reward"][agent_idx] += geom_lane_reward
            self._episode_sums["geom_route_progress"][agent_idx] += geom_route_progress

            self._previous_goal_distance[agent_idx] = goal_distance
            self._previous_raw_actions[agent_idx] = self._raw_actions[agent_idx]
            if self._invincible_mode_enabled():
                new_done = self._pending_goal_done_mask[agent_idx]
            else:
                new_done = (
                    self._pending_goal_done_mask[agent_idx]
                    | self._pending_collision_done_mask[agent_idx]
                    | self._pending_crash_done_mask[agent_idx]
                    | self._pending_lane_forbidden_done_mask[agent_idx]
                )
            new_done_env_ids = torch.nonzero(new_done, as_tuple=False).squeeze(-1)
            if new_done_env_ids.numel() > 0:
                self._park_done_vehicle(agent_idx, new_done_env_ids)
            self._terminal_goal_distance[agent_idx] = torch.where(new_done, goal_distance, self._terminal_goal_distance[agent_idx])
            self._agent_done_mask[agent_idx] |= new_done
            self._goal_done_mask[agent_idx] |= self._pending_goal_done_mask[agent_idx]
            self._collision_done_mask[agent_idx] |= new_collision_event
            self._crash_done_mask[agent_idx] |= new_crash_event
            self._crash_too_low_done_mask[agent_idx] |= new_crash_too_low_event
            self._crash_too_far_done_mask[agent_idx] |= new_crash_too_far_event
            self._crash_bad_tilt_done_mask[agent_idx] |= new_crash_bad_tilt_event
            self._lane_forbidden_done_mask[agent_idx] |= new_lane_forbidden_event
            self._sync_timing_device()
            reward_finalize_ms += (perf_counter() - reward_finalize_start) * 1000.0
        self._previous_root_pos_xy.copy_(root_pos_xy)

        self._step_timing_last_ms["reward_shared_ms"] = float(reward_shared_ms)
        self._step_timing_last_ms["reward_goal_ms"] = float(reward_goal_ms)
        self._step_timing_last_ms["reward_geom_ms"] = float(reward_geom_ms)
        self._step_timing_last_ms["reward_route_progress_ms"] = float(reward_route_progress_ms)
        self._step_timing_last_ms["reward_ttc_ms"] = float(reward_ttc_ms)
        self._step_timing_last_ms["reward_road_edge_ttc_ms"] = float(reward_road_edge_ttc_ms)
        self._step_timing_last_ms["reward_finalize_ms"] = float(reward_finalize_ms)
        return rewards

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        reward_start = perf_counter()
        self._sync_timing_device()
        reward_lane_start = perf_counter()
        self._update_lane_touch_mask()
        self._sync_timing_device()
        self._step_timing_last_ms["reward_lane_ms"] = (perf_counter() - reward_lane_start) * 1000.0
        reward_mode = str(getattr(self.cfg, "reward_mode", "scene_factory_default")).strip().lower().replace("-", "_")
        if reward_mode == "choco_baseline":
            rewards = self._get_rewards_choco_aligned()
            self._pending_goal_done_mask.zero_()
            self._pending_collision_done_mask.zero_()
            self._pending_crash_done_mask.zero_()
            self._pending_crash_too_low_mask.zero_()
            self._pending_crash_too_far_mask.zero_()
            self._pending_crash_bad_tilt_mask.zero_()
            self._pending_lane_forbidden_done_mask.zero_()
            self._step_timing_last_ms["reward_ms"] = (perf_counter() - reward_start) * 1000.0
            return rewards
        rewards = {}
        pairwise_distances = self._pairwise_distances_xy()
        nearest_distances = pairwise_distances.min(dim=2).values if self._num_agents > 1 else None

        for agent_idx, agent_id in enumerate(self._agent_ids):
            goal_pos_b, goal_distance = self._compute_goal_distance(agent_idx)
            self._current_goal_distance[agent_idx] = goal_distance

            active_mask = ~self._agent_done_mask[agent_idx]
            active_float = active_mask.float()
            goal_dir_b = goal_pos_b[:, :2] / goal_distance.unsqueeze(-1).clamp_min(1.0e-6)
            progress = self._previous_goal_distance[agent_idx] - goal_distance
            heading_alignment = goal_dir_b[:, 0]
            speed_to_goal = torch.sum(self._vehicles[agent_idx].data.root_lin_vel_b[:, :2] * goal_dir_b, dim=1).clamp_min(
                0.0
            )
            lateral_velocity = torch.abs(self._vehicles[agent_idx].data.root_lin_vel_b[:, 1])
            yaw_rate = torch.abs(self._vehicles[agent_idx].data.root_ang_vel_b[:, 2])
            action_rate = torch.sum(torch.square(self._raw_actions[agent_idx] - self._previous_raw_actions[agent_idx]), dim=1)
            action_magnitude = torch.sum(torch.square(self._semantic_actions[agent_idx]), dim=1)
            throttle_brake_conflict = self._semantic_actions[agent_idx, :, 0] * self._semantic_actions[agent_idx, :, 2]
            goal_shaping = 1.0 - torch.tanh(goal_distance / float(max(1.0, self.cfg.goal_radius_max_m * 0.5)))
            goal_bonus = self._pending_goal_done_mask[agent_idx].float() * float(self.cfg.reward_goal_bonus)
            new_collision_event = self._pending_collision_done_mask[agent_idx] & (~self._collision_done_mask[agent_idx])
            new_crash_event = self._pending_crash_done_mask[agent_idx] & (~self._crash_done_mask[agent_idx])
            new_crash_too_low_event = self._pending_crash_too_low_mask[agent_idx] & (~self._crash_too_low_done_mask[agent_idx])
            new_crash_too_far_event = self._pending_crash_too_far_mask[agent_idx] & (~self._crash_too_far_done_mask[agent_idx])
            new_crash_bad_tilt_event = self._pending_crash_bad_tilt_mask[agent_idx] & (~self._crash_bad_tilt_done_mask[agent_idx])
            crash_penalty = new_crash_event.float() * float(self.cfg.reward_crash_penalty)
            collision_penalty = new_collision_event.float() * float(self.cfg.reward_collision_penalty)
            lane_center_touch = (
                self._lane_touch_any_type_mask(agent_idx, self.cfg.reward_lane_center_types)
                if bool(self.cfg.reward_lane_center_enable)
                else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            )
            lane_forbidden_new = self._pending_lane_forbidden_done_mask[agent_idx] & (~self._lane_forbidden_done_mask[agent_idx])

            if nearest_distances is not None:
                proximity_violation = torch.relu(
                    float(self.cfg.agent_safe_distance_m) - nearest_distances[:, agent_idx]
                ) / float(max(1.0e-6, self.cfg.agent_safe_distance_m))
            else:
                proximity_violation = torch.zeros(self.num_envs, device=self.device)
            neighbor_proximity = proximity_violation * float(self.cfg.reward_scale_neighbor_proximity)

            reward_terms = {
                "alive": torch.full_like(goal_distance, float(self.cfg.reward_scale_alive)) * active_float,
                "progress": progress * float(self.cfg.reward_scale_progress) * active_float,
                "goal_shaping": goal_shaping * float(self.cfg.reward_scale_goal_shaping) * active_float,
                "heading": heading_alignment * float(self.cfg.reward_scale_heading) * active_float,
                "speed_to_goal": speed_to_goal * float(self.cfg.reward_scale_speed_to_goal) * active_float,
                "lateral_velocity": lateral_velocity * float(self.cfg.reward_scale_lateral_velocity) * active_float,
                "yaw_rate": yaw_rate * float(self.cfg.reward_scale_yaw_rate) * active_float,
                "action_rate": action_rate * float(self.cfg.reward_scale_action_rate) * active_float,
                "action_magnitude": action_magnitude * float(self.cfg.reward_scale_action_magnitude) * active_float,
                "throttle_brake_conflict": throttle_brake_conflict
                * float(self.cfg.reward_scale_throttle_brake_conflict)
                * active_float,
                "neighbor_proximity": neighbor_proximity * active_float,
                "goal_bonus": goal_bonus,
                "collision_penalty": collision_penalty,
                "crash_penalty": crash_penalty,
                "lane_center_bonus": lane_center_touch.float() * float(self.cfg.reward_lane_center_per_step) * active_float,
                "lane_forbidden_penalty": lane_forbidden_new.float() * float(self.cfg.reward_lane_forbidden_penalty),
            }
            rewards[agent_id] = torch.sum(torch.stack(list(reward_terms.values())), dim=0)

            for key, value in reward_terms.items():
                self._episode_sums[key][agent_idx] += value

            self._previous_goal_distance[agent_idx] = goal_distance
            self._previous_raw_actions[agent_idx] = self._raw_actions[agent_idx]
            if self._invincible_mode_enabled():
                new_done = self._pending_goal_done_mask[agent_idx]
            else:
                new_done = (
                    self._pending_goal_done_mask[agent_idx]
                    | self._pending_collision_done_mask[agent_idx]
                    | self._pending_crash_done_mask[agent_idx]
                    | self._pending_lane_forbidden_done_mask[agent_idx]
                )
            new_done_env_ids = torch.nonzero(new_done, as_tuple=False).squeeze(-1)
            if new_done_env_ids.numel() > 0:
                self._park_done_vehicle(agent_idx, new_done_env_ids)
            self._terminal_goal_distance[agent_idx] = torch.where(
                new_done,
                goal_distance,
                self._terminal_goal_distance[agent_idx],
            )
            self._agent_done_mask[agent_idx] |= new_done
            self._goal_done_mask[agent_idx] |= self._pending_goal_done_mask[agent_idx]
            self._collision_done_mask[agent_idx] |= new_collision_event
            self._crash_done_mask[agent_idx] |= new_crash_event
            self._crash_too_low_done_mask[agent_idx] |= new_crash_too_low_event
            self._crash_too_far_done_mask[agent_idx] |= new_crash_too_far_event
            self._crash_bad_tilt_done_mask[agent_idx] |= new_crash_bad_tilt_event
            self._lane_forbidden_done_mask[agent_idx] |= lane_forbidden_new

        self._pending_goal_done_mask.zero_()
        self._pending_collision_done_mask.zero_()
        self._pending_crash_done_mask.zero_()
        self._pending_crash_too_low_mask.zero_()
        self._pending_crash_too_far_mask.zero_()
        self._pending_crash_bad_tilt_mask.zero_()
        self._pending_lane_forbidden_done_mask.zero_()
        self._step_timing_last_ms["reward_ms"] = (perf_counter() - reward_start) * 1000.0
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._sync_timing_device()
        done_start = perf_counter()
        if str(self.cfg.test_mode).strip().lower() in {
            "collision_test",
            "scene_factory_collision_test",
            "scene_factory_multiworld_random_steer_test",
        }:
            false_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            terminated = {agent_id: false_buf.clone() for agent_id in self._agent_ids}
            time_outs = {agent_id: false_buf.clone() for agent_id in self._agent_ids}
            self._sync_timing_device()
            self._step_timing_last_ms["done_ms"] = (perf_counter() - done_start) * 1000.0
            return terminated, time_outs

        self._steps_since_reset_buf += 1

        self._sync_timing_device()
        done_lane_start = perf_counter()
        self._update_lane_touch_mask()
        self._sync_timing_device()
        done_lane_ms = (perf_counter() - done_lane_start) * 1000.0

        self._sync_timing_device()
        done_collision_start = perf_counter()
        collision_masks = self._collision_by_agent_mask_tensor()
        self._sync_timing_device()
        done_collision_ms = (perf_counter() - done_collision_start) * 1000.0

        self._sync_timing_device()
        done_state_start = perf_counter()
        _goal_pos_b, goal_distance_all = self._compute_goal_distance_all()
        crash_too_low_all, crash_too_far_all, crash_bad_tilt_all = self._agent_crash_components_all()
        crash_masks = crash_too_low_all | crash_too_far_all | crash_bad_tilt_all
        lane_forbidden_touch_all = (
            self._lane_touch_any_type_mask_all(self.cfg.reward_lane_forbidden_types)
            if bool(self.cfg.reward_lane_forbidden_enable)
            else torch.zeros((self._num_agents, self.num_envs), dtype=torch.bool, device=self.device)
        )
        self._pending_goal_done_mask.zero_()
        self._pending_collision_done_mask.zero_()
        self._pending_crash_done_mask.zero_()
        self._pending_crash_too_low_mask.zero_()
        self._pending_crash_too_far_mask.zero_()
        self._pending_crash_bad_tilt_mask.zero_()
        self._pending_lane_forbidden_done_mask.zero_()
        terminated = {}
        for agent_idx in range(self._num_agents):
            goal_distance = goal_distance_all[agent_idx]
            self._current_goal_distance[agent_idx] = goal_distance
            active_mask = ~self._agent_done_mask[agent_idx]
            goal_reached = goal_distance <= float(self.cfg.goal_reached_threshold_m)
            crash_mask = crash_masks[agent_idx]
            collision_mask = collision_masks[agent_idx]
            lane_forbidden_touch = lane_forbidden_touch_all[agent_idx]
            self._pending_goal_done_mask[agent_idx] = active_mask & goal_reached
            self._pending_crash_done_mask[agent_idx] = active_mask & crash_mask
            self._pending_crash_too_low_mask[agent_idx] = active_mask & crash_too_low_all[agent_idx]
            self._pending_crash_too_far_mask[agent_idx] = active_mask & crash_too_far_all[agent_idx]
            self._pending_crash_bad_tilt_mask[agent_idx] = active_mask & crash_bad_tilt_all[agent_idx]
            self._pending_collision_done_mask[agent_idx] = active_mask & collision_mask
            self._pending_lane_forbidden_done_mask[agent_idx] = active_mask & lane_forbidden_touch
            if self._invincible_mode_enabled():
                terminated[self._agent_ids[agent_idx]] = (
                    self._agent_done_mask[agent_idx]
                    | self._pending_goal_done_mask[agent_idx]
                )
            else:
                terminated[self._agent_ids[agent_idx]] = (
                    self._agent_done_mask[agent_idx]
                    | self._pending_goal_done_mask[agent_idx]
                    | self._pending_crash_done_mask[agent_idx]
                    | self._pending_collision_done_mask[agent_idx]
                    | self._pending_lane_forbidden_done_mask[agent_idx]
                )
        self._sync_timing_device()
        done_state_ms = (perf_counter() - done_state_start) * 1000.0

        self._sync_timing_device()
        done_finalize_start = perf_counter()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_outs = {agent_id: time_out.clone() for agent_id in self._agent_ids}
        self._sync_timing_device()
        done_finalize_ms = (perf_counter() - done_finalize_start) * 1000.0

        self._sync_timing_device()
        self._step_timing_last_ms["done_lane_ms"] = float(done_lane_ms)
        self._step_timing_last_ms["done_collision_ms"] = float(done_collision_ms)
        self._step_timing_last_ms["done_state_ms"] = float(done_state_ms)
        self._step_timing_last_ms["done_finalize_ms"] = float(done_finalize_ms)
        self._step_timing_last_ms["done_ms"] = (perf_counter() - done_start) * 1000.0
        return terminated, time_outs

    def consume_last_reset_world_episode_summaries(self) -> list[dict[str, Any]]:
        payload = list(self._last_reset_world_episode_summaries)
        self._last_reset_world_episode_summaries = []
        return payload

    def _reset_idx(self, env_ids: Sequence[int] | None):
        self._lane_touch_mask_cache_valid = False
        self._last_reset_world_episode_summaries = []
        self._sync_timing_device()
        reset_start = perf_counter()
        reset_write_ms = 0.0
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._vehicles[0]._ALL_INDICES

        self._sync_timing_device()
        reset_metrics_prep_start = perf_counter()
        collision_force_by_agent = self._collision_force_by_agent_n_tensor()
        lane_center_touch_by_agent = torch.zeros((self._num_agents, self.num_envs), dtype=torch.bool, device=self.device)
        if bool(self.cfg.reward_lane_center_enable):
            self._update_lane_touch_mask()
            lane_center_touch_by_agent = self._lane_touch_any_type_mask_all(self.cfg.reward_lane_center_types)
        self._sync_timing_device()
        self._step_timing_last_ms["reset_metrics_prep_ms"] = (perf_counter() - reset_metrics_prep_start) * 1000.0
        self._step_timing_last_ms["reset_write_ms"] = 0.0

        self._sync_timing_device()
        reset_log_start = perf_counter()
        for agent_id in self._agent_ids:
            self.extras[agent_id] = {"log": {}}

        if self._agent_ids:
            world_active_mask = self._spawned_agent_mask[:, env_ids]
            active_spawn_float = world_active_mask.float()
            total_spawned_count = active_spawn_float.sum()
            spawned_denom = torch.clamp(total_spawned_count, min=1.0)

            aggregate_log: dict[str, float | torch.Tensor] = {}
            reward_time_norm = max(1.0, float(self.max_episode_length_s))
            for key, value in self._episode_sums.items():
                aggregate_log[f"Episode_Reward/{key}"] = (value[:, env_ids].mean(dim=1).mean() / reward_time_norm).item()
                value[:, env_ids] = 0.0

            final_distance = torch.where(
                self._agent_done_mask[:, env_ids],
                self._terminal_goal_distance[:, env_ids],
                self._current_goal_distance[:, env_ids],
            )
            aggregate_log["Metrics/final_distance_to_goal"] = (
                (final_distance * active_spawn_float).sum() / spawned_denom
            ).item()
            success_count_total = (self._goal_done_mask[:, env_ids].float() * active_spawn_float).sum()
            crash_count_total = (self._crash_done_mask[:, env_ids].float() * active_spawn_float).sum()
            crash_too_low_count_total = (self._crash_too_low_done_mask[:, env_ids].float() * active_spawn_float).sum()
            crash_too_far_count_total = (self._crash_too_far_done_mask[:, env_ids].float() * active_spawn_float).sum()
            crash_bad_tilt_count_total = (self._crash_bad_tilt_done_mask[:, env_ids].float() * active_spawn_float).sum()
            collision_count_total = (self._collision_done_mask[:, env_ids].float() * active_spawn_float).sum()
            lane_center_touch_count_total = (lane_center_touch_by_agent[:, env_ids].float() * active_spawn_float).sum()
            lane_forbidden_done_count_total = (self._lane_forbidden_done_mask[:, env_ids].float() * active_spawn_float).sum()
            aggregate_log["Metrics/success_rate"] = (
                success_count_total / spawned_denom
            ).item()
            all_goals_reached = torch.all(self._goal_done_mask[:, env_ids] | (~world_active_mask), dim=0)
            aggregate_log["Metrics/all_goals_reached_count"] = float(torch.count_nonzero(all_goals_reached).item())
            aggregate_log["Metrics/all_goals_reached_rate"] = torch.mean(all_goals_reached.float()).item()
            aggregate_log["Metrics/crash_rate"] = (
                crash_count_total / spawned_denom
            ).item()
            aggregate_log["Metrics/crash_too_low_count"] = float(crash_too_low_count_total.item())
            aggregate_log["Metrics/crash_too_far_count"] = float(crash_too_far_count_total.item())
            aggregate_log["Metrics/crash_bad_tilt_count"] = float(crash_bad_tilt_count_total.item())
            aggregate_log["Metrics/crash_too_low_rate"] = (
                crash_too_low_count_total / spawned_denom
            ).item()
            aggregate_log["Metrics/crash_too_far_rate"] = (
                crash_too_far_count_total / spawned_denom
            ).item()
            aggregate_log["Metrics/crash_bad_tilt_rate"] = (
                crash_bad_tilt_count_total / spawned_denom
            ).item()
            aggregate_log["Metrics/collision_rate"] = (
                collision_count_total / spawned_denom
            ).item()
            aggregate_log["Metrics/collision_count"] = float(collision_count_total.item())
            aggregate_log["Metrics/max_collision_force_n"] = (
                (collision_force_by_agent[:, env_ids] * active_spawn_float).sum() / spawned_denom
            ).item()
            aggregate_log["Metrics/lane_center_touch_rate"] = (
                lane_center_touch_count_total / spawned_denom
            ).item()
            aggregate_log["Metrics/lane_center_touch_count"] = float(lane_center_touch_count_total.item())
            aggregate_log["Metrics/lane_forbidden_done_count"] = float(lane_forbidden_done_count_total.item())
            aggregate_log["Metrics/lane_forbidden_done_rate"] = (
                lane_forbidden_done_count_total / spawned_denom
            ).item()
            aggregate_log["Metrics/controlled_spawn_count"] = float(total_spawned_count.item())

            # TTC / DRAC safety metrics (only meaningful with >1 agent per env)
            if self._num_agents > 1:
                ttc_steps = self._episode_ttc_finite_steps[:, env_ids]          # [A, E]
                ttc_sum = self._episode_min_ttc_sum[:, env_ids]                  # [A, E]
                ttc_valid_mask = (ttc_steps > 0) & world_active_mask             # [A, E]
                ttc_denom = ttc_valid_mask.float().sum().clamp(min=1.0)
                mean_ttc_per_slot = ttc_sum / torch.clamp(ttc_steps, min=1.0)   # [A, E]
                aggregate_log["Metrics/mean_min_ttc_s"] = (
                    (mean_ttc_per_slot * ttc_valid_mask.float()).sum() / ttc_denom
                ).item()
                near_miss_any = (self._episode_near_miss_steps[:, env_ids] > 0) & world_active_mask
                aggregate_log["Metrics/near_miss_rate"] = (
                    near_miss_any.float().sum() / spawned_denom
                ).item()
                max_drac = self._episode_max_drac[:, env_ids]                    # [A, E]
                drac_denom = world_active_mask.float().sum().clamp(min=1.0)
                aggregate_log["Metrics/mean_max_drac"] = (
                    (max_drac * world_active_mask.float()).sum() / drac_denom
                ).item()
                high_drac_any = (max_drac > _DRAC_HIGH_THRESHOLD) & world_active_mask
                aggregate_log["Metrics/high_drac_rate"] = (
                    high_drac_any.float().sum() / spawned_denom
                ).item()
                # Accumulate lifetime counters
                self._lifetime_near_miss_count += float(near_miss_any.float().sum().item())
                self._lifetime_high_drac_count += float(high_drac_any.float().sum().item())
                self._lifetime_ttc_episode_count += float(ttc_valid_mask.float().sum().item())

                # --- Per-world TTC/DRAC (stored for per-world summary dicts) ---
                _w_spawned_f = world_active_mask.float().sum(dim=0).clamp(min=1.0)  # [E_local]
                _w_ttc_valid_count = ttc_valid_mask.float().sum(dim=0).clamp(min=1.0)  # [E_local]
                _w_mean_ttc_s = (mean_ttc_per_slot * ttc_valid_mask.float()).sum(dim=0) / _w_ttc_valid_count
                _w_near_miss_rate = near_miss_any.float().sum(dim=0) / _w_spawned_f
                _w_mean_max_drac = (max_drac * world_active_mask.float()).sum(dim=0) / _w_spawned_f
                _w_high_drac_rate = high_drac_any.float().sum(dim=0) / _w_spawned_f
                self._tmp_world_mean_ttc_s: list[float] = _w_mean_ttc_s.detach().cpu().tolist()
                self._tmp_world_near_miss_rate: list[float] = _w_near_miss_rate.detach().cpu().tolist()
                self._tmp_world_mean_max_drac: list[float] = _w_mean_max_drac.detach().cpu().tolist()
                self._tmp_world_high_drac_rate: list[float] = _w_high_drac_rate.detach().cpu().tolist()

            # Reset TTC / DRAC episode buffers for the environments being reset
            self._episode_min_ttc_sum[:, env_ids] = 0.0
            self._episode_ttc_finite_steps[:, env_ids] = 0.0
            self._episode_near_miss_steps[:, env_ids] = 0.0
            self._episode_max_drac[:, env_ids] = 0.0

            self._lifetime_controlled_spawn_count += float(total_spawned_count.item())
            self._lifetime_success_count += float(success_count_total.item())
            self._lifetime_all_goals_reached_count += float(torch.count_nonzero(all_goals_reached).item())
            self._lifetime_crash_count += float(crash_count_total.item())
            self._lifetime_crash_too_low_count += float(crash_too_low_count_total.item())
            self._lifetime_crash_too_far_count += float(crash_too_far_count_total.item())
            self._lifetime_crash_bad_tilt_count += float(crash_bad_tilt_count_total.item())
            self._lifetime_collision_count += float(collision_count_total.item())
            self._lifetime_lane_center_touch_count += float(lane_center_touch_count_total.item())
            self._lifetime_lane_forbidden_count += float(lane_forbidden_done_count_total.item())

            lifetime_spawned_denom = max(1.0, float(self._lifetime_controlled_spawn_count))
            aggregate_log["LifetimeMetrics/controlled_spawn_count"] = float(self._lifetime_controlled_spawn_count)
            aggregate_log["LifetimeMetrics/success_count"] = float(self._lifetime_success_count)
            aggregate_log["LifetimeMetrics/all_goals_reached_count"] = float(self._lifetime_all_goals_reached_count)
            aggregate_log["LifetimeMetrics/crash_count"] = float(self._lifetime_crash_count)
            aggregate_log["LifetimeMetrics/crash_too_low_count"] = float(self._lifetime_crash_too_low_count)
            aggregate_log["LifetimeMetrics/crash_too_far_count"] = float(self._lifetime_crash_too_far_count)
            aggregate_log["LifetimeMetrics/crash_bad_tilt_count"] = float(self._lifetime_crash_bad_tilt_count)
            aggregate_log["LifetimeMetrics/collision_count"] = float(self._lifetime_collision_count)
            aggregate_log["LifetimeMetrics/lane_center_touch_count"] = float(self._lifetime_lane_center_touch_count)
            aggregate_log["LifetimeMetrics/lane_forbidden_done_count"] = float(self._lifetime_lane_forbidden_count)
            aggregate_log["LifetimeMetrics/success_rate"] = float(self._lifetime_success_count / lifetime_spawned_denom)
            aggregate_log["LifetimeMetrics/crash_rate"] = float(self._lifetime_crash_count / lifetime_spawned_denom)
            aggregate_log["LifetimeMetrics/crash_too_low_rate"] = float(
                self._lifetime_crash_too_low_count / lifetime_spawned_denom
            )
            aggregate_log["LifetimeMetrics/crash_too_far_rate"] = float(
                self._lifetime_crash_too_far_count / lifetime_spawned_denom
            )
            aggregate_log["LifetimeMetrics/crash_bad_tilt_rate"] = float(
                self._lifetime_crash_bad_tilt_count / lifetime_spawned_denom
            )
            aggregate_log["LifetimeMetrics/collision_rate"] = float(self._lifetime_collision_count / lifetime_spawned_denom)
            aggregate_log["LifetimeMetrics/lane_center_touch_rate"] = float(
                self._lifetime_lane_center_touch_count / lifetime_spawned_denom
            )
            aggregate_log["LifetimeMetrics/lane_forbidden_done_rate"] = float(
                self._lifetime_lane_forbidden_count / lifetime_spawned_denom
            )
            self.extras[self._agent_ids[0]]["log"].update(aggregate_log)

        if self._agent_ids:
            world_spawned_mask = self._spawned_agent_mask[:, env_ids]
            world_spawned_count = world_spawned_mask.sum(dim=0).to(dtype=torch.float32)
            world_success_mask = self._goal_done_mask[:, env_ids] & world_spawned_mask
            if self._invincible_mode_enabled():
                world_collision_mask = self._collision_done_mask[:, env_ids] & world_spawned_mask
                world_lane_forbidden_mask = self._lane_forbidden_done_mask[:, env_ids] & world_spawned_mask
                world_crash_mask = self._crash_done_mask[:, env_ids] & world_spawned_mask
            else:
                world_collision_mask = self._collision_done_mask[:, env_ids] & world_spawned_mask & (~world_success_mask)
                world_lane_forbidden_mask = (
                    self._lane_forbidden_done_mask[:, env_ids]
                    & world_spawned_mask
                    & (~world_success_mask)
                    & (~world_collision_mask)
                )
                world_crash_mask = (
                    self._crash_done_mask[:, env_ids]
                    & world_spawned_mask
                    & (~world_success_mask)
                    & (~world_collision_mask)
                    & (~world_lane_forbidden_mask)
                )
            world_success_count = world_success_mask.sum(dim=0).to(dtype=torch.float32)
            world_collision_count = world_collision_mask.sum(dim=0).to(dtype=torch.float32)
            world_lane_forbidden_count = world_lane_forbidden_mask.sum(dim=0).to(dtype=torch.float32)
            world_crash_count = world_crash_mask.sum(dim=0).to(dtype=torch.float32)
            world_crash_too_low_count = (self._crash_too_low_done_mask[:, env_ids] & world_spawned_mask).sum(dim=0).to(dtype=torch.float32)
            world_crash_too_far_count = (self._crash_too_far_done_mask[:, env_ids] & world_spawned_mask).sum(dim=0).to(dtype=torch.float32)
            world_crash_bad_tilt_count = (self._crash_bad_tilt_done_mask[:, env_ids] & world_spawned_mask).sum(dim=0).to(dtype=torch.float32)
            if self._invincible_mode_enabled():
                world_active_not_done_count = (world_spawned_count - world_success_count).clamp_min(0.0)
            else:
                world_active_not_done_count = (
                    world_spawned_count
                    - world_success_count
                    - world_collision_count
                    - world_lane_forbidden_count
                    - world_crash_count
                ).clamp_min(0.0)
            world_final_distance = torch.where(
                world_spawned_count > 0.0,
                (final_distance * active_spawn_float).sum(dim=0) / torch.clamp(world_spawned_count, min=1.0),
                torch.zeros_like(world_spawned_count),
            )
            total_world_spawned_count = world_spawned_count.sum()
            world_spawned_denom = torch.clamp(total_world_spawned_count, min=1.0)
            world_log = self.extras[self._agent_ids[0]]["log"]
            world_log["WorldEpisode/spawned_count"] = float(total_world_spawned_count.item())
            world_log["WorldEpisode/success_count"] = float(world_success_count.sum().item())
            world_log["WorldEpisode/collision_count"] = float(world_collision_count.sum().item())
            world_log["WorldEpisode/lane_forbidden_count"] = float(world_lane_forbidden_count.sum().item())
            world_log["WorldEpisode/crash_count"] = float(world_crash_count.sum().item())
            world_log["WorldEpisode/crash_too_low_count"] = float(world_crash_too_low_count.sum().item())
            world_log["WorldEpisode/crash_too_far_count"] = float(world_crash_too_far_count.sum().item())
            world_log["WorldEpisode/crash_bad_tilt_count"] = float(world_crash_bad_tilt_count.sum().item())
            world_log["WorldEpisode/active_not_done_count"] = float(world_active_not_done_count.sum().item())
            world_log["WorldEpisode/success_rate"] = float((world_success_count.sum() / world_spawned_denom).item())
            world_log["WorldEpisode/collision_rate"] = float((world_collision_count.sum() / world_spawned_denom).item())
            world_log["WorldEpisode/lane_forbidden_rate"] = float((world_lane_forbidden_count.sum() / world_spawned_denom).item())
            world_log["WorldEpisode/crash_rate"] = float((world_crash_count.sum() / world_spawned_denom).item())
            world_log["WorldEpisode/crash_too_low_rate"] = float((world_crash_too_low_count.sum() / world_spawned_denom).item())
            world_log["WorldEpisode/crash_too_far_rate"] = float((world_crash_too_far_count.sum() / world_spawned_denom).item())
            world_log["WorldEpisode/crash_bad_tilt_rate"] = float((world_crash_bad_tilt_count.sum() / world_spawned_denom).item())
            world_log["LifetimeEpisode/spawned_count"] = float(self._lifetime_controlled_spawn_count)
            world_log["LifetimeEpisode/success_count"] = float(self._lifetime_success_count)
            world_log["LifetimeEpisode/collision_count"] = float(self._lifetime_collision_count)
            world_log["LifetimeEpisode/lane_forbidden_count"] = float(self._lifetime_lane_forbidden_count)
            world_log["LifetimeEpisode/crash_count"] = float(self._lifetime_crash_count)
            world_log["LifetimeEpisode/crash_too_low_count"] = float(self._lifetime_crash_too_low_count)
            world_log["LifetimeEpisode/crash_too_far_count"] = float(self._lifetime_crash_too_far_count)
            world_log["LifetimeEpisode/crash_bad_tilt_count"] = float(self._lifetime_crash_bad_tilt_count)
            world_log["LifetimeEpisode/success_rate"] = float(self._lifetime_success_count / lifetime_spawned_denom)
            world_log["LifetimeEpisode/collision_rate"] = float(self._lifetime_collision_count / lifetime_spawned_denom)
            world_log["LifetimeEpisode/lane_forbidden_rate"] = float(self._lifetime_lane_forbidden_count / lifetime_spawned_denom)
            world_log["LifetimeEpisode/crash_rate"] = float(self._lifetime_crash_count / lifetime_spawned_denom)
            world_log["LifetimeEpisode/crash_too_low_rate"] = float(self._lifetime_crash_too_low_count / lifetime_spawned_denom)
            world_log["LifetimeEpisode/crash_too_far_rate"] = float(self._lifetime_crash_too_far_count / lifetime_spawned_denom)
            world_log["LifetimeEpisode/crash_bad_tilt_rate"] = float(self._lifetime_crash_bad_tilt_count / lifetime_spawned_denom)

            env_id_list = [int(v) for v in env_ids.detach().cpu().tolist()]
            world_episode_length_steps = self.episode_length_buf[env_ids].detach().to(dtype=torch.long).cpu().tolist()
            for local_idx, env_id in enumerate(env_id_list):
                scene_json_path = ""
                scene_json_name = ""
                world_index = int(env_id)
                if self._scene_factory_scene_json_paths_by_env is not None and env_id < len(self._scene_factory_scene_json_paths_by_env):
                    scene_json_path = str(self._scene_factory_scene_json_paths_by_env[env_id])
                    scene_json_name = Path(scene_json_path).name
                if self._scene_factory_specs_by_env is not None and env_id < len(self._scene_factory_specs_by_env):
                    world_index = int(self._scene_factory_specs_by_env[env_id].world_index)
                spawned_count = float(world_spawned_count[local_idx].item())
                denom = max(1.0, spawned_count)
                self._last_reset_world_episode_summaries.append(
                    {
                        "env_index": int(env_id),
                        "world_index": int(world_index),
                        "scene_json_path": scene_json_path,
                        "scene_json_name": scene_json_name,
                        "episode_length_steps": int(world_episode_length_steps[local_idx]),
                        "spawned_count": spawned_count,
                        "success_count": float(world_success_count[local_idx].item()),
                        "collision_count": float(world_collision_count[local_idx].item()),
                        "lane_forbidden_count": float(world_lane_forbidden_count[local_idx].item()),
                        "crash_count": float(world_crash_count[local_idx].item()),
                        "crash_too_low_count": float(world_crash_too_low_count[local_idx].item()),
                        "crash_too_far_count": float(world_crash_too_far_count[local_idx].item()),
                        "crash_bad_tilt_count": float(world_crash_bad_tilt_count[local_idx].item()),
                        "active_not_done_count": float(world_active_not_done_count[local_idx].item()),
                        "success_rate": float(world_success_count[local_idx].item() / denom),
                        "collision_rate": float(world_collision_count[local_idx].item() / denom),
                        "lane_forbidden_rate": float(world_lane_forbidden_count[local_idx].item() / denom),
                        "crash_rate": float(world_crash_count[local_idx].item() / denom),
                        "mean_final_distance_to_goal": float(world_final_distance[local_idx].item()),
                        "mean_min_ttc_s": float(self._tmp_world_mean_ttc_s[local_idx]) if getattr(self, "_tmp_world_mean_ttc_s", None) is not None else -1.0,
                        "near_miss_rate": float(self._tmp_world_near_miss_rate[local_idx]) if getattr(self, "_tmp_world_near_miss_rate", None) is not None else -1.0,
                        "mean_max_drac": float(self._tmp_world_mean_max_drac[local_idx]) if getattr(self, "_tmp_world_mean_max_drac", None) is not None else -1.0,
                        "high_drac_rate": float(self._tmp_world_high_drac_rate[local_idx]) if getattr(self, "_tmp_world_high_drac_rate", None) is not None else -1.0,
                    }
                )
        self._sync_timing_device()
        self._step_timing_last_ms["reset_log_ms"] = (perf_counter() - reset_log_start) * 1000.0

        self._sync_timing_device()
        reset_backend_start = perf_counter()
        if self._should_use_teleport_only_reset():
            self._apply_lightweight_reset_state(env_ids)
        else:
            for vehicle in self._vehicles:
                vehicle.reset(env_ids)
            super()._reset_idx(env_ids)
            self._teleport_only_reset_initialized = True
            if self._reset_mode_name() == "teleport_only" and not bool(self._teleport_only_reset_announced):
                print(
                    "[INFO][SceneFactory] reset_mode=teleport_only: using one full initialization reset, "
                    "then teleport-only resets for subsequent episodes.",
                    flush=True,
                )
                self._teleport_only_reset_announced = True
                self._sync_timing_device()
        self._step_timing_last_ms["reset_backend_ms"] = (perf_counter() - reset_backend_start) * 1000.0

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._steps_since_reset_buf[env_ids] = 0
        # Reset bicycle speed buffer for re-spawned envs
        self._bicycle_speed_buf[:, env_ids] = 0.0

        self._sync_timing_device()
        reset_spawn_prep_start = perf_counter()
        reset_state_clear_ms = 0.0
        reset_pose_build_ms = 0.0
        reset_pose_park_ms = 0.0
        reset_pose_spawn_ms = 0.0
        reset_pose_finalize_ms = 0.0
        reset_pose_quat_ms = 0.0
        reset_pose_goal_ms = 0.0
        reset_pose_inactive_ms = 0.0
        reset_pose_joint_defaults_ms = 0.0
        num_resets = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]
        test_mode = str(self.cfg.test_mode).strip().lower()
        use_collision_test = test_mode in {"collision_test", "scene_factory_collision_test"}
        use_scene_factory_roads = bool(
            self.cfg.use_scene_factory_roads and self._scenario_spawns_by_env is not None and not use_collision_test
        )
        if self.cfg.randomize_spawn_phase and not use_scene_factory_roads:
            phase = sample_uniform(-math.pi, math.pi, (num_resets,), self.device)
        else:
            phase = torch.zeros(num_resets, device=self.device)
        if use_scene_factory_roads:
            shared_world_offset = torch.zeros(num_resets, 2, device=self.device)
            goal_radius = torch.zeros(num_resets, device=self.device)
        else:
            shared_world_offset = sample_uniform(
                -float(self.cfg.start_radius_m),
                float(self.cfg.start_radius_m),
                (num_resets, 2),
                self.device,
            )
            goal_radius = sample_uniform(
                float(self.cfg.goal_radius_min_m),
                float(self.cfg.goal_radius_max_m),
                (num_resets,),
                self.device,
            )

        for agent_idx, vehicle in enumerate(self._vehicles):
            self._sync_timing_device()
            reset_state_clear_start = perf_counter()
            if use_scene_factory_roads:
                spawned_mask = self._scene_factory_spawn_valid[env_ids, agent_idx]
            else:
                spawned_mask = torch.ones(num_resets, dtype=torch.bool, device=self.device)
            self._spawned_agent_mask[agent_idx, env_ids] = spawned_mask
            self._raw_actions[agent_idx, env_ids] = 0.0
            self._semantic_actions[agent_idx, env_ids] = 0.0
            self._previous_raw_actions[agent_idx, env_ids] = 0.0
            self._terminal_goal_distance[agent_idx, env_ids] = 0.0
            self._agent_done_mask[agent_idx, env_ids] = False
            self._goal_done_mask[agent_idx, env_ids] = False
            self._collision_done_mask[agent_idx, env_ids] = False
            self._crash_done_mask[agent_idx, env_ids] = False
            self._crash_too_low_done_mask[agent_idx, env_ids] = False
            self._crash_too_far_done_mask[agent_idx, env_ids] = False
            self._crash_bad_tilt_done_mask[agent_idx, env_ids] = False
            self._lane_forbidden_done_mask[agent_idx, env_ids] = False
            self._pending_goal_done_mask[agent_idx, env_ids] = False
            self._pending_collision_done_mask[agent_idx, env_ids] = False
            self._pending_crash_done_mask[agent_idx, env_ids] = False
            self._pending_crash_too_low_mask[agent_idx, env_ids] = False
            self._pending_crash_too_far_mask[agent_idx, env_ids] = False
            self._pending_crash_bad_tilt_mask[agent_idx, env_ids] = False
            self._pending_lane_forbidden_done_mask[agent_idx, env_ids] = False
            self._joint_effort_targets[agent_idx][env_ids] = 0.0
            self._external_forces[agent_idx][env_ids] = 0.0
            self._external_torques[agent_idx][env_ids] = 0.0
            self._brake_sign_memory[agent_idx][env_ids] = 1.0
            self._sync_timing_device()
            reset_state_clear_ms += (perf_counter() - reset_state_clear_start) * 1000.0

            self._sync_timing_device()
            reset_pose_build_start = perf_counter()
            self._sync_timing_device()
            reset_pose_park_start = perf_counter()
            root_pose = self._done_vehicle_root_pose(agent_idx, env_ids)
            root_velocity = self._default_joint_vel[agent_idx].new_zeros((num_resets, 6))
            goal_pos_w = root_pose[:, :3].clone()
            yaw = torch.zeros(num_resets, device=self.device)
            self._sync_timing_device()
            reset_pose_park_ms += (perf_counter() - reset_pose_park_start) * 1000.0

            self._sync_timing_device()
            reset_pose_spawn_start = perf_counter()
            if use_collision_test:
                half_distance = float(self.cfg.collision_test_half_distance_m)
                goal_distance = float(self.cfg.collision_test_goal_distance_m)
                start_x = -half_distance if agent_idx == 0 else half_distance
                goal_x = goal_distance if agent_idx == 0 else -goal_distance
                yaw_value = 0.0 if agent_idx == 0 else math.pi
                root_pose[:, 0] = env_origins[:, 0] + start_x
                root_pose[:, 1] = env_origins[:, 1]
                root_pose[:, 2] = env_origins[:, 2] + float(self.cfg.spawn_height_m)
                yaw = torch.full((num_resets,), yaw_value, device=self.device)
                goal_local = torch.tensor(
                    [goal_x, 0.0, float(self.cfg.goal_height_m)],
                    device=self.device,
                ).unsqueeze(0).repeat(num_resets, 1)
                goal_pos_w = env_origins + goal_local
            elif use_scene_factory_roads:
                if bool(self.cfg.random_od) and self._scene_factory_scene_cfgs_by_env is not None and agent_idx == 0:
                    self._resample_random_od_for_envs(env_ids)
                    # Re-read spawned_mask after resample may have changed valid flags
                    spawned_mask = self._scene_factory_spawn_valid[env_ids, agent_idx]
                    self._spawned_agent_mask[agent_idx, env_ids] = spawned_mask
                if bool(torch.any(spawned_mask).item()):
                    valid_rows = torch.nonzero(spawned_mask, as_tuple=False).squeeze(-1)
                    spawn_jitter = sample_uniform(
                        -float(self.cfg.agent_spawn_jitter_m),
                        float(self.cfg.agent_spawn_jitter_m),
                        (int(valid_rows.numel()), 2),
                        self.device,
                    )
                    start_local_xyz = self._scene_factory_spawn_start_local[env_ids[valid_rows], agent_idx]
                    start_yaw = self._scene_factory_spawn_start_yaw[env_ids[valid_rows], agent_idx]
                    goal_local_xyz = self._scene_factory_spawn_goal_local[env_ids[valid_rows], agent_idx]
                    root_pose[valid_rows, 0] = env_origins[valid_rows, 0] + start_local_xyz[:, 0] + spawn_jitter[:, 0]
                    root_pose[valid_rows, 1] = env_origins[valid_rows, 1] + start_local_xyz[:, 1] + spawn_jitter[:, 1]
                    root_pose[valid_rows, 2] = env_origins[valid_rows, 2] + float(self.cfg.spawn_height_m)
                    if not bool(self._scene_factory_ignore_dataset_spawn_z):
                        root_pose[valid_rows, 2] += start_local_xyz[:, 2]
                    yaw[valid_rows] = start_yaw + sample_uniform(
                        -float(self.cfg.spawn_yaw_noise_rad),
                        float(self.cfg.spawn_yaw_noise_rad),
                        (int(valid_rows.numel()),),
                        self.device,
                    )
                    goal_local = goal_local_xyz.clone()
                    if bool(self._scene_factory_ignore_dataset_spawn_z):
                        goal_local[:, 2] = float(self.cfg.goal_height_m)
                    else:
                        goal_local[:, 2] = torch.maximum(
                            goal_local[:, 2],
                            torch.full(
                                (int(valid_rows.numel()),),
                                float(self.cfg.goal_height_m),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        )
                    goal_pos_w[valid_rows] = env_origins[valid_rows] + goal_local
            else:
                if bool(self.cfg.friction_ruler_mode):
                    # Friction ruler: spawn at env origin, facing +Y (north)
                    root_pose[:, 0] = env_origins[:, 0]
                    root_pose[:, 1] = env_origins[:, 1]
                    root_pose[:, 2] = env_origins[:, 2] + float(self.cfg.spawn_height_m)
                    yaw = torch.full((num_resets,), math.pi / 2.0, device=self.device)
                    goal_pos_w[:, 0] = env_origins[:, 0]
                    goal_pos_w[:, 1] = env_origins[:, 1] + 50.0
                    goal_pos_w[:, 2] = env_origins[:, 2] + float(self.cfg.goal_height_m)
                else:
                    formation_angle = phase + (2.0 * math.pi * agent_idx / max(1, self._num_agents))
                    spawn_jitter = sample_uniform(
                        -float(self.cfg.agent_spawn_jitter_m),
                        float(self.cfg.agent_spawn_jitter_m),
                        (num_resets, 2),
                        self.device,
                    )
                    spawn_offset = torch.stack(
                        [
                            float(self.cfg.agent_spawn_circle_radius_m) * torch.cos(formation_angle),
                            float(self.cfg.agent_spawn_circle_radius_m) * torch.sin(formation_angle),
                        ],
                        dim=-1,
                    )
                    root_pose[:, 0:2] = env_origins[:, 0:2] + shared_world_offset + spawn_offset + spawn_jitter
                    root_pose[:, 2] = env_origins[:, 2] + float(self.cfg.spawn_height_m)
                    yaw = formation_angle + math.pi + sample_uniform(
                        -float(self.cfg.spawn_yaw_noise_rad),
                        float(self.cfg.spawn_yaw_noise_rad),
                        (num_resets,),
                        self.device,
                    )
                    goal_heading = formation_angle + math.pi + sample_uniform(
                        -float(self.cfg.goal_heading_noise_rad),
                        float(self.cfg.goal_heading_noise_rad),
                        (num_resets,),
                        self.device,
                    )
                    env_goal_offset = torch.stack(
                        [
                            goal_radius * torch.cos(goal_heading),
                            goal_radius * torch.sin(goal_heading),
                            torch.full_like(goal_radius, float(self.cfg.goal_height_m)),
                        ],
                        dim=-1,
                    )
                    goal_pos_w = env_origins + env_goal_offset
            self._sync_timing_device()
            reset_pose_spawn_ms += (perf_counter() - reset_pose_spawn_start) * 1000.0

            self._sync_timing_device()
            reset_pose_finalize_start = perf_counter()

            self._sync_timing_device()
            reset_pose_quat_start = perf_counter()
            self._set_root_pose_quat_from_yaw(root_pose, yaw)
            root_velocity.zero_()
            self._sync_timing_device()
            reset_pose_quat_ms += (perf_counter() - reset_pose_quat_start) * 1000.0

            self._sync_timing_device()
            reset_pose_goal_start = perf_counter()
            self._goal_pos_w[agent_idx, env_ids] = goal_pos_w
            self._previous_goal_distance[agent_idx, env_ids] = torch.linalg.norm(
                self._goal_pos_w[agent_idx, env_ids, :2] - root_pose[:, :2],
                dim=1,
            )
            self._current_goal_distance[agent_idx, env_ids] = self._previous_goal_distance[agent_idx, env_ids]
            self._previous_root_pos_xy[agent_idx, env_ids] = root_pose[:, :2] - env_origins[:, :2]
            self._sync_timing_device()
            reset_pose_goal_ms += (perf_counter() - reset_pose_goal_start) * 1000.0

            self._sync_timing_device()
            reset_pose_inactive_start = perf_counter()
            if bool(torch.any(~spawned_mask).item()):
                inactive_env_ids = env_ids[~spawned_mask]
                self._previous_goal_distance[agent_idx, inactive_env_ids] = 0.0
                self._current_goal_distance[agent_idx, inactive_env_ids] = 0.0
                self._previous_root_pos_xy[agent_idx, inactive_env_ids] = 0.0
                self._agent_done_mask[agent_idx, inactive_env_ids] = True
            self._sync_timing_device()
            reset_pose_inactive_ms += (perf_counter() - reset_pose_inactive_start) * 1000.0

            self._sync_timing_device()
            reset_pose_joint_defaults_start = perf_counter()
            joint_pos = self._default_joint_pos[agent_idx][env_ids].clone()
            joint_vel = self._default_joint_vel[agent_idx][env_ids].clone()
            self._sync_timing_device()
            reset_pose_joint_defaults_ms += (perf_counter() - reset_pose_joint_defaults_start) * 1000.0
            reset_pose_finalize_ms += (perf_counter() - reset_pose_finalize_start) * 1000.0
            self._sync_timing_device()
            reset_pose_build_ms += (perf_counter() - reset_pose_build_start) * 1000.0

            self._sync_timing_device()
            reset_write_start = perf_counter()
            vehicle.write_root_pose_to_sim(root_pose, env_ids)
            vehicle.write_root_velocity_to_sim(root_velocity, env_ids)
            vehicle.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            vehicle.set_joint_effort_target(self._joint_effort_targets[agent_idx][env_ids], env_ids=env_ids)
            vehicle.permanent_wrench_composer.set_forces_and_torques(
                forces=self._external_forces[agent_idx][env_ids],
                torques=self._external_torques[agent_idx][env_ids],
                body_ids=self._base_body_ids[agent_idx],
                env_ids=env_ids,
                is_global=False,
            )
            self._sync_timing_device()
            reset_write_ms += (perf_counter() - reset_write_start) * 1000.0
        self._step_timing_last_ms["reset_state_clear_ms"] = float(reset_state_clear_ms)
        self._step_timing_last_ms["reset_pose_build_ms"] = float(reset_pose_build_ms)
        self._step_timing_last_ms["reset_pose_park_ms"] = float(reset_pose_park_ms)
        self._step_timing_last_ms["reset_pose_spawn_ms"] = float(reset_pose_spawn_ms)
        self._step_timing_last_ms["reset_pose_finalize_ms"] = float(reset_pose_finalize_ms)
        self._step_timing_last_ms["reset_pose_quat_ms"] = float(reset_pose_quat_ms)
        self._step_timing_last_ms["reset_pose_goal_ms"] = float(reset_pose_goal_ms)
        self._step_timing_last_ms["reset_pose_inactive_ms"] = float(reset_pose_inactive_ms)
        self._step_timing_last_ms["reset_pose_joint_defaults_ms"] = float(reset_pose_joint_defaults_ms)
        self._step_timing_last_ms["reset_write_ms"] = float(reset_write_ms)
        self._sync_timing_device()
        self._step_timing_last_ms["reset_spawn_prep_ms"] = max(
            0.0,
            (perf_counter() - reset_spawn_prep_start) * 1000.0 - float(reset_write_ms),
        )
        self._lane_touch_mask_cache_valid = False
        self._sync_timing_device()
        self._step_timing_last_ms["reset_ms"] = (perf_counter() - reset_start) * 1000.0

    def _inject_step_timing_logs(self, extras: dict) -> dict:
        if not bool(self.cfg.step_timing_log_enable):
            return extras
        if not isinstance(extras, dict):
            extras = {}

        self._step_timing_call_count += 1
        known_ms = sum(
            float(self._step_timing_last_ms.get(key, 0.0))
            for key in (
                "pre_physics_ms",
                "apply_action_ms",
                "physics_write_ms",
                "physics_sim_ms",
                "physics_update_ms",
                "obs_ms",
                "reward_ms",
                "done_ms",
                "reset_ms",
                "step_bookkeeping_ms",
                "step_event_ms",
                "step_obs_noise_ms",
            )
        )
        step_total_ms = float(self._step_timing_last_ms.get("step_total_ms", 0.0))
        self._step_timing_last_ms["step_other_ms"] = max(0.0, step_total_ms - known_ms)

        for key, value in self._step_timing_last_ms.items():
            prev = float(self._step_timing_ema_ms.get(key, 0.0))
            self._step_timing_ema_ms[key] = float(value) if self._step_timing_call_count == 1 else (0.9 * prev + 0.1 * float(value))

        timing_agent_ids = self._agent_ids[:1] if self._agent_ids else []
        for agent_id in timing_agent_ids:
            agent_extra = extras.get(agent_id)
            if not isinstance(agent_extra, dict):
                agent_extra = {}
                extras[agent_id] = agent_extra
            log = agent_extra.get("log")
            if not isinstance(log, dict):
                log = {}
                agent_extra["log"] = log
            for key, value in self._step_timing_last_ms.items():
                log[f"Perf/{key}"] = float(value)
            for key, value in self._step_timing_ema_ms.items():
                log[f"PerfEma/{key}"] = float(value)

        if bool(self.cfg.step_timing_print_enable):
            every_n = max(1, int(self.cfg.step_timing_print_every_n))
            if self._step_timing_call_count % every_n == 0:
                print(
                    "[INFO][SceneFactory][StepTiming] "
                    f"step={self._step_timing_call_count} total_ms={self._step_timing_last_ms['step_total_ms']:.2f} "
                    f"pre_ms={self._step_timing_last_ms['pre_physics_ms']:.2f} "
                    f"apply_ms={self._step_timing_last_ms['apply_action_ms']:.2f} "
                    f"write_ms={self._step_timing_last_ms['physics_write_ms']:.2f} "
                    f"sim_ms={self._step_timing_last_ms['physics_sim_ms']:.2f} "
                    f"update_ms={self._step_timing_last_ms['physics_update_ms']:.2f} "
                    f"obs_ms={self._step_timing_last_ms['obs_ms']:.2f} "
                    f"obs_lane_ms={self._step_timing_last_ms['obs_lane_ms']:.2f} "
                    f"obs_shared_ms={self._step_timing_last_ms['obs_shared_ms']:.2f} "
                    f"obs_goal_ms={self._step_timing_last_ms['obs_goal_ms']:.2f} "
                    f"obs_road_ms={self._step_timing_last_ms['obs_road_ms']:.2f} "
                    f"obs_neighbor_ms={self._step_timing_last_ms['obs_neighbor_ms']:.2f} "
                    f"obs_finalize_ms={self._step_timing_last_ms['obs_finalize_ms']:.2f} "
                    f"reward_ms={self._step_timing_last_ms['reward_ms']:.2f} "
                    f"reward_lane_ms={self._step_timing_last_ms['reward_lane_ms']:.2f} "
                    f"reward_shared_ms={self._step_timing_last_ms['reward_shared_ms']:.2f} "
                    f"reward_goal_ms={self._step_timing_last_ms['reward_goal_ms']:.2f} "
                    f"reward_geom_ms={self._step_timing_last_ms['reward_geom_ms']:.2f} "
                    f"reward_route_progress_ms={self._step_timing_last_ms['reward_route_progress_ms']:.2f} "
                    f"reward_ttc_ms={self._step_timing_last_ms['reward_ttc_ms']:.2f} "
                    f"reward_road_edge_ttc_ms={self._step_timing_last_ms['reward_road_edge_ttc_ms']:.2f} "
                    f"reward_finalize_ms={self._step_timing_last_ms['reward_finalize_ms']:.2f} "
                    f"done_ms={self._step_timing_last_ms['done_ms']:.2f} "
                    f"reset_ms={self._step_timing_last_ms['reset_ms']:.2f} "
                    f"reset_metrics_prep_ms={self._step_timing_last_ms['reset_metrics_prep_ms']:.2f} "
                    f"reset_log_ms={self._step_timing_last_ms['reset_log_ms']:.2f} "
                    f"reset_backend_ms={self._step_timing_last_ms['reset_backend_ms']:.2f} "
                    f"reset_spawn_prep_ms={self._step_timing_last_ms['reset_spawn_prep_ms']:.2f} "
                    f"reset_state_clear_ms={self._step_timing_last_ms['reset_state_clear_ms']:.2f} "
                    f"reset_pose_build_ms={self._step_timing_last_ms['reset_pose_build_ms']:.2f} "
                    f"reset_pose_park_ms={self._step_timing_last_ms['reset_pose_park_ms']:.2f} "
                    f"reset_pose_spawn_ms={self._step_timing_last_ms['reset_pose_spawn_ms']:.2f} "
                    f"reset_pose_finalize_ms={self._step_timing_last_ms['reset_pose_finalize_ms']:.2f} "
                    f"reset_pose_quat_ms={self._step_timing_last_ms['reset_pose_quat_ms']:.2f} "
                    f"reset_pose_goal_ms={self._step_timing_last_ms['reset_pose_goal_ms']:.2f} "
                    f"reset_pose_inactive_ms={self._step_timing_last_ms['reset_pose_inactive_ms']:.2f} "
                    f"reset_pose_joint_defaults_ms={self._step_timing_last_ms['reset_pose_joint_defaults_ms']:.2f} "
                    f"reset_write_ms={self._step_timing_last_ms['reset_write_ms']:.2f} "
                    f"bookkeeping_ms={self._step_timing_last_ms['step_bookkeeping_ms']:.2f} "
                    f"event_ms={self._step_timing_last_ms['step_event_ms']:.2f} "
                    f"obs_noise_ms={self._step_timing_last_ms['step_obs_noise_ms']:.2f} "
                    f"other_ms={self._step_timing_last_ms['step_other_ms']:.2f}",
                    flush=True,
                )
        return extras

    def step(self, actions):
        self._sync_timing_device()
        step_start = perf_counter()
        self._lane_touch_mask_cache_valid = False
        self._collision_force_cache_valid = False
        actions = {agent: action.to(self.device) for agent, action in actions.items()}
        if self.cfg.action_noise_model:
            for agent, action in actions.items():
                if agent in self._action_noise_model:
                    actions[agent] = self._action_noise_model[agent](action)

        self._pre_physics_step(actions)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        apply_action_total_ms = 0.0
        self._step_timing_last_ms["apply_action_math_ms"] = 0.0
        self._step_timing_last_ms["apply_action_target_submit_ms"] = 0.0
        self._step_timing_last_ms["apply_action_wrench_submit_ms"] = 0.0
        physics_write_ms = 0.0
        physics_sim_ms = 0.0
        physics_update_ms = 0.0
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            self._sync_timing_device()
            apply_action_start = perf_counter()
            self._apply_action()
            self._sync_timing_device()
            apply_action_total_ms += (perf_counter() - apply_action_start) * 1000.0

            self._sync_timing_device()
            physics_write_start = perf_counter()
            self.scene.write_data_to_sim()
            self._sync_timing_device()
            physics_write_ms += (perf_counter() - physics_write_start) * 1000.0

            self._sync_timing_device()
            physics_sim_start = perf_counter()
            self.sim.step(render=False)
            self._sync_timing_device()
            physics_sim_ms += (perf_counter() - physics_sim_start) * 1000.0

            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()

            self._sync_timing_device()
            physics_update_start = perf_counter()
            self.scene.update(dt=self.physics_dt)
            self._sync_timing_device()
            physics_update_ms += (perf_counter() - physics_update_start) * 1000.0

        self._step_timing_last_ms["apply_action_ms"] = float(apply_action_total_ms)
        self._step_timing_last_ms["physics_write_ms"] = float(physics_write_ms)
        self._step_timing_last_ms["physics_sim_ms"] = float(physics_sim_ms)
        self._step_timing_last_ms["physics_update_ms"] = float(physics_update_ms)

        self._sync_timing_device()
        step_bookkeeping_start = perf_counter()
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._sync_timing_device()
        step_bookkeeping_ms = (perf_counter() - step_bookkeeping_start) * 1000.0

        terminated, time_outs = self._get_dones()
        self.terminated_dict = terminated
        self.time_out_dict = time_outs
        self.reset_buf[:] = math.prod(self.terminated_dict.values()) | math.prod(self.time_out_dict.values())
        rewards = self._get_rewards()
        self.reward_dict = rewards
        self._step_timing_last_ms["step_bookkeeping_ms"] = float(step_bookkeeping_ms)

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        self._sync_timing_device()
        step_event_start = perf_counter()
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)
        self._sync_timing_device()
        self._step_timing_last_ms["step_event_ms"] = (perf_counter() - step_event_start) * 1000.0

        obs = self._get_observations()
        self.obs_dict = obs
        self.agents = [agent for agent in self.possible_agents if agent in self.obs_dict]

        self._sync_timing_device()
        step_obs_noise_start = perf_counter()
        if self.cfg.observation_noise_model:
            for agent, obs_value in self.obs_dict.items():
                if agent in self._observation_noise_model:
                    self.obs_dict[agent] = self._observation_noise_model[agent](obs_value)
        self._sync_timing_device()
        self._step_timing_last_ms["step_obs_noise_ms"] = (perf_counter() - step_obs_noise_start) * 1000.0

        extras = self.extras
        self._sync_timing_device()
        self._step_timing_last_ms["step_total_ms"] = (perf_counter() - step_start) * 1000.0
        extras = self._inject_step_timing_logs(extras)
        return self.obs_dict, self.reward_dict, self.terminated_dict, self.time_out_dict, extras

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not bool(self.cfg.hide_goal_markers):
                if not hasattr(self, "_goal_marker"):
                    self._goal_marker = build_goal_beacon_marker("/Visuals/MultiGoalMarker")
                self._goal_marker.set_visibility(True)
            if bool(self.cfg.vehicle_proxy_marker_enable):
                if self._vehicle_proxy_marker is None:
                    self._vehicle_proxy_marker = _build_vehicle_proxy_marker(
                        "/Visuals/VehicleProxyMarkers",
                        num_agent_prototypes=self._num_agents,
                        vehicle_length_m=float(self._vehicle_length_m),
                        vehicle_width_m=float(self._vehicle_width_m),
                    )
                self._vehicle_proxy_marker.set_visibility(True)
            if bool(self.cfg.collision_test_debug_markers) and str(self.cfg.test_mode).strip().lower() in {
                "collision_test",
                "scene_factory_collision_test",
            }:
                if not hasattr(self, "_collision_test_vehicle_marker"):
                    self._collision_test_vehicle_marker = build_goal_beacon_marker(
                        "/Visuals/CollisionTestVehicles"
                    )
                self._collision_test_vehicle_marker.set_visibility(True)
        else:
            if hasattr(self, "_goal_marker"):
                self._goal_marker.set_visibility(False)
            if self._vehicle_proxy_marker is not None:
                self._vehicle_proxy_marker.set_visibility(False)
            if hasattr(self, "_collision_test_vehicle_marker"):
                self._collision_test_vehicle_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "scene") or self.scene is None or not hasattr(self, "_vehicles"):
            return
        if hasattr(self, "_goal_marker"):
            goal_positions = self._goal_pos_w.permute(1, 0, 2).reshape(-1, 3)
            marker_positions, marker_indices = goal_beacon_visualization(goal_positions)
            self._goal_marker.visualize(marker_positions, marker_indices=marker_indices)
        if self._vehicle_proxy_marker is not None:
            self._update_vehicle_proxy_markers()
        if hasattr(self, "_collision_test_vehicle_marker"):
            positions = []
            indices = []
            for agent_idx, vehicle in enumerate(self._vehicles):
                pos_w = vehicle.data.root_pos_w.clone()
                pos_w[:, 2] += 2.15
                positions.append(pos_w)
                indices.append(
                    torch.full((self.num_envs,), min(agent_idx, 1), dtype=torch.int32, device=self.device)
                )
            self._collision_test_vehicle_marker.visualize(
                torch.cat(positions, dim=0),
                marker_indices=torch.cat(indices, dim=0),
            )

    @property
    def tunable_config(self) -> StudentTunableConfig:
        return self._tunable_config

    def tunable_config_dict(self) -> dict:
        return asdict(self._tunable_config)

    def close(self):
        if self._capture_camera is not None:
            del self._capture_camera
            self._capture_camera = None
        if self._vehicle_proxy_marker is not None:
            del self._vehicle_proxy_marker
            self._vehicle_proxy_marker = None
        super().close()
