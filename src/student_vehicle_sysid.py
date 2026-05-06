from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
import traceback
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

from src.physx_teacher_patch_track import (
    SurfacePatch,
    VehicleCommand,
    _build_track_and_materials,
    _clamp,
    _ensure_physics_scene,
    _ensure_world_default_prim,
    _set_stage_units,
    select_surface_patch,
)
from src.student_vehicle_sysid_visualizer import write_sysid_report


DEFAULT_STUDENT_USD = "artifacts/student_vehicle_assets/vehicle_only_smoke/student_fwd_vehicle.usd"


def _default_surface_scale_dict() -> dict[str, float]:
    return {
        "dry_asphalt": 1.0,
        "wet_asphalt": 1.0,
        "gravel": 1.0,
    }


@dataclass(frozen=True)
class ReplayLossWeights:
    position_xy: float = 1.0
    yaw: float = 0.4
    speed: float = 0.35
    yaw_rate: float = 0.25
    wheel_speed: float = 0.05
    steer_angle: float = 0.05
    suspension: float = 0.0
    terminal_position_xy: float = 1.5
    terminal_speed: float = 0.75


@dataclass(frozen=True)
class StudentTunableConfig:
    drive_torque_nm: float = 900.0
    brake_front_torque_nm: float = 1400.0
    brake_rear_torque_nm: float = 700.0
    steering_limit_rad: float = math.radians(32.0)
    steering_kp_nm_per_rad: float = 1600.0
    steering_kd_nm_s_per_rad: float = 160.0
    steering_effort_limit_nm: float = 1200.0
    wheel_viscous_friction: float = 0.25
    steering_viscous_friction: float = 1.0
    suspension_viscous_friction: float = 120.0
    wheel_mass_kg: float = 20.0
    wheel_inertia_scale: float = 1.0
    suspension_stiffness_n_m: float = 0.0
    suspension_damping_n_s_m: float = 2200.0
    chassis_com_height_offset_m: float = 0.0
    lateral_velocity_damping_n_per_mps: float = 0.0
    yaw_stability_damping_nm_per_rad_s: float = 0.0
    surface_friction_scale: dict[str, float] = field(default_factory=_default_surface_scale_dict)
    surface_longitudinal_scale: dict[str, float] = field(default_factory=_default_surface_scale_dict)
    surface_lateral_scale: dict[str, float] = field(default_factory=_default_surface_scale_dict)


@dataclass(frozen=True)
class TeacherRollout:
    name: str
    rollout_dir: Path
    metadata: dict[str, Any]
    frames: list[dict[str, Any]]
    patches: list[SurfacePatch]


@dataclass(frozen=True)
class TrialResult:
    name: str
    config: StudentTunableConfig
    loss: dict[str, float]
    num_frames: int
    output_dir: Path


@dataclass(frozen=True)
class TeacherDataset:
    manifest_path: Path
    manifest: dict[str, Any]
    rollouts: list[TeacherRollout]


@dataclass(frozen=True)
class SearchStage:
    name: str
    description: str
    teachers: list[TeacherRollout]
    search_space: dict[str, tuple[float, float]]
    random_trials: int
    search_window_fraction: float | None = None
    search_min_fraction: float | None = None
    seed_from_stage: str | None = None


@dataclass(frozen=True)
class CEMSettings:
    population_size: int = 16
    elite_fraction: float = 0.25
    initial_std_fraction: float = 0.25
    min_std_fraction: float = 0.05


def _surface_patch_from_payload(payload: Mapping[str, Any]) -> SurfacePatch:
    return SurfacePatch(
        name=str(payload["name"]),
        x_center_m=float(payload["x_center_m"]),
        y_center_m=float(payload["y_center_m"]),
        length_m=float(payload["length_m"]),
        width_m=float(payload["width_m"]),
        static_friction=float(payload["static_friction"]),
        dynamic_friction=float(payload["dynamic_friction"]),
        tire_friction=float(payload["tire_friction"]),
        color_srgb=tuple(float(v) for v in payload["color_srgb"]),
    )


def _wrap_angle_rad(angle_rad: float) -> float:
    wrapped = math.fmod(float(angle_rad) + math.pi, 2.0 * math.pi)
    if wrapped < 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def _yaw_from_quat_wxyz(quat_wxyz: Sequence[float]) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _get_nested(mapping: Mapping[str, Any], path: str) -> Any:
    value: Any = mapping
    for key in str(path).split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(path)
        value = value[key]
    return value


def _set_nested(mapping: MutableMapping[str, Any], path: str, value: Any) -> None:
    cursor: MutableMapping[str, Any] = mapping
    keys = str(path).split(".")
    for key in keys[:-1]:
        next_value = cursor.get(key)
        if not isinstance(next_value, MutableMapping):
            next_value = {}
            cursor[key] = next_value
        cursor = next_value
    cursor[keys[-1]] = value


def load_teacher_rollout(rollout_dir: str | Path) -> TeacherRollout:
    root = Path(rollout_dir).expanduser().resolve()
    meta_path = root / "rollout_meta.json"
    frames_path = root / "rollout_frames.jsonl"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    patches = [_surface_patch_from_payload(payload) for payload in metadata.get("surface_patches", [])]
    return TeacherRollout(name=root.name, rollout_dir=root, metadata=metadata, frames=frames, patches=patches)


def load_teacher_dataset_manifest(path: str | Path) -> TeacherDataset:
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Teacher dataset manifest must contain an object.")
    rollout_entries = manifest.get("rollouts", [])
    if not isinstance(rollout_entries, list) or not rollout_entries:
        raise ValueError("Teacher dataset manifest does not contain any rollouts.")
    rollouts: list[TeacherRollout] = []
    for entry in rollout_entries:
        rollout = load_teacher_rollout(entry["rollout_dir"])
        rollouts.append(
            TeacherRollout(
                name=str(entry.get("name", rollout.name)),
                rollout_dir=rollout.rollout_dir,
                metadata=rollout.metadata,
                frames=rollout.frames,
                patches=rollout.patches,
            )
        )
    return TeacherDataset(
        manifest_path=manifest_path,
        manifest=manifest,
        rollouts=rollouts,
    )


def load_tunable_config(path: str | Path) -> StudentTunableConfig:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Tunable config JSON must contain an object.")
    default = asdict(StudentTunableConfig())
    merged = {**default, **payload}
    for field_name in ("surface_friction_scale", "surface_longitudinal_scale", "surface_lateral_scale"):
        merged_surface = dict(default[field_name])
        if isinstance(payload.get(field_name), Mapping):
            merged_surface.update({str(k): float(v) for k, v in payload[field_name].items()})
        merged[field_name] = merged_surface
    return normalize_tunable_config(StudentTunableConfig(**merged))


def load_loss_weights(path: str | Path) -> ReplayLossWeights:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Loss weights JSON must contain an object.")
    default = asdict(ReplayLossWeights())
    return ReplayLossWeights(**{**default, **payload})


def load_search_space(path: str | Path) -> dict[str, tuple[float, float]]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Search space JSON must contain an object.")
    search_space: dict[str, tuple[float, float]] = {}
    for key, value in payload.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Search-space entry '{key}' must be a [lo, hi] pair.")
        lo = float(value[0])
        hi = float(value[1])
        if hi < lo:
            raise ValueError(f"Search-space entry '{key}' has invalid bounds: {value}")
        search_space[str(key)] = (lo, hi)
    return search_space


def default_search_space() -> dict[str, tuple[float, float]]:
    return {
        "drive_torque_nm": (300.0, 1800.0),
        "brake_front_torque_nm": (400.0, 2400.0),
        "brake_rear_torque_nm": (200.0, 1600.0),
        "steering_limit_rad": (0.15, 0.70),
        "steering_kp_nm_per_rad": (200.0, 5000.0),
        "steering_kd_nm_s_per_rad": (20.0, 800.0),
        "wheel_viscous_friction": (0.0, 6.0),
        "wheel_mass_kg": (8.0, 60.0),
        "wheel_inertia_scale": (0.5, 3.0),
        "suspension_stiffness_n_m": (0.0, 30000.0),
        "suspension_damping_n_s_m": (200.0, 12000.0),
        "chassis_com_height_offset_m": (-0.25, 0.25),
        "lateral_velocity_damping_n_per_mps": (0.0, 12000.0),
        "yaw_stability_damping_nm_per_rad_s": (0.0, 12000.0),
        "surface_longitudinal_scale.dry_asphalt": (0.5, 1.5),
        "surface_longitudinal_scale.wet_asphalt": (0.4, 1.4),
        "surface_longitudinal_scale.gravel": (0.4, 1.4),
        "surface_lateral_scale.dry_asphalt": (0.5, 1.5),
        "surface_lateral_scale.wet_asphalt": (0.4, 1.4),
        "surface_lateral_scale.gravel": (0.4, 1.4),
    }


def _surface_search_space(surface_names: Sequence[str]) -> dict[str, tuple[float, float]]:
    surface_bounds = {
        "dry_asphalt": (0.5, 1.5),
        "wet_asphalt": (0.4, 1.4),
        "gravel": (0.4, 1.4),
    }
    search_space: dict[str, tuple[float, float]] = {}
    for surface_name in surface_names:
        if surface_name not in surface_bounds:
            continue
        search_space[f"surface_longitudinal_scale.{surface_name}"] = surface_bounds[surface_name]
        search_space[f"surface_lateral_scale.{surface_name}"] = surface_bounds[surface_name]
    return search_space


def _merge_search_spaces(*spaces: Mapping[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
    merged: dict[str, tuple[float, float]] = {}
    for space in spaces:
        for key, bounds in space.items():
            merged[str(key)] = (float(bounds[0]), float(bounds[1]))
    return merged


def _longitudinal_search_space(surface_names: Sequence[str]) -> dict[str, tuple[float, float]]:
    return _merge_search_spaces(
        {
            "drive_torque_nm": (300.0, 1400.0),
            "brake_front_torque_nm": (500.0, 2200.0),
            "brake_rear_torque_nm": (200.0, 1400.0),
            "wheel_viscous_friction": (0.0, 6.0),
            "wheel_mass_kg": (8.0, 60.0),
            "wheel_inertia_scale": (0.5, 3.0),
        },
        {
            key: bounds
            for key, bounds in _surface_search_space(surface_names).items()
            if key.startswith("surface_longitudinal_scale.")
        },
    )


def _steering_search_space() -> dict[str, tuple[float, float]]:
    return {
        "steering_limit_rad": (0.15, 0.45),
        "steering_kp_nm_per_rad": (200.0, 5000.0),
        "steering_kd_nm_s_per_rad": (20.0, 800.0),
        "suspension_stiffness_n_m": (0.0, 30000.0),
        "suspension_damping_n_s_m": (200.0, 12000.0),
        "chassis_com_height_offset_m": (-0.25, 0.25),
        "lateral_velocity_damping_n_per_mps": (0.0, 12000.0),
        "yaw_stability_damping_nm_per_rad_s": (0.0, 12000.0),
    }


def _joint_refinement_search_space(surface_names: Sequence[str]) -> dict[str, tuple[float, float]]:
    return _merge_search_spaces(
        {
            "drive_torque_nm": (300.0, 1600.0),
            "brake_front_torque_nm": (500.0, 2200.0),
            "brake_rear_torque_nm": (200.0, 1400.0),
            "steering_limit_rad": (0.15, 0.45),
            "steering_kp_nm_per_rad": (200.0, 5000.0),
            "steering_kd_nm_s_per_rad": (20.0, 800.0),
            "wheel_viscous_friction": (0.0, 6.0),
            "wheel_mass_kg": (8.0, 60.0),
            "wheel_inertia_scale": (0.5, 3.0),
            "suspension_stiffness_n_m": (0.0, 30000.0),
            "suspension_damping_n_s_m": (200.0, 12000.0),
            "chassis_com_height_offset_m": (-0.25, 0.25),
            "lateral_velocity_damping_n_per_mps": (0.0, 12000.0),
            "yaw_stability_damping_nm_per_rad_s": (0.0, 12000.0),
        },
        _surface_search_space(surface_names),
    )


def touched_surface_names(teacher: TeacherRollout) -> list[str]:
    names: set[str] = set()
    for frame in teacher.frames:
        for wheel in frame.get("wheels", []):
            surface_name = wheel.get("surface_name")
            if surface_name is None:
                continue
            names.add(str(surface_name))
    return sorted(names)


def infer_maneuver_family(name: str) -> str:
    lowered = str(name).lower()
    if "straight_accel_brake" in lowered or "straight-accel-brake" in lowered:
        return "longitudinal"
    if "surface_transition" in lowered:
        return "surface"
    if "step_steer" in lowered or "sine_steer" in lowered or "constant_steer" in lowered:
        return "steering"
    return "mixed"


def auto_search_space_for_rollout(teacher: TeacherRollout) -> dict[str, tuple[float, float]]:
    maneuver_family = infer_maneuver_family(teacher.name)
    surfaces = touched_surface_names(teacher)
    if maneuver_family == "longitudinal":
        return _longitudinal_search_space(surfaces)
    if maneuver_family == "surface":
        return _surface_search_space(surfaces)
    if maneuver_family == "steering":
        return _steering_search_space()
    return _joint_refinement_search_space(surfaces)


def normalize_tunable_config(config: StudentTunableConfig) -> StudentTunableConfig:
    mutable = asdict(config)
    brake_front = float(mutable["brake_front_torque_nm"])
    brake_rear = float(mutable["brake_rear_torque_nm"])
    if brake_front < brake_rear:
        mutable["brake_front_torque_nm"] = brake_rear
        mutable["brake_rear_torque_nm"] = brake_front
    mutable["wheel_mass_kg"] = max(1.0e-3, float(mutable["wheel_mass_kg"]))
    mutable["wheel_inertia_scale"] = max(1.0e-3, float(mutable["wheel_inertia_scale"]))
    mutable["suspension_stiffness_n_m"] = max(0.0, float(mutable["suspension_stiffness_n_m"]))
    mutable["suspension_damping_n_s_m"] = max(0.0, float(mutable["suspension_damping_n_s_m"]))
    for field_name in ("surface_friction_scale", "surface_longitudinal_scale", "surface_lateral_scale"):
        merged_surface = dict(_default_surface_scale_dict())
        merged_surface.update(
            {
                str(name): float(value)
                for name, value in mutable.get(field_name, {}).items()
            }
        )
        mutable[field_name] = merged_surface
    return StudentTunableConfig(**mutable)


def _surface_scale_value(surface_scale: Mapping[str, float], surface_name: str | None) -> float:
    if surface_name is None:
        return 1.0
    return float(surface_scale.get(str(surface_name), 1.0))


def _config_to_search_vector(
    config: StudentTunableConfig,
    search_space: Mapping[str, tuple[float, float]],
) -> tuple[list[str], list[float], list[tuple[float, float]]]:
    paths = sorted(str(key) for key in search_space.keys())
    payload = asdict(config)
    values = [float(_get_nested(payload, path)) for path in paths]
    bounds = [(float(search_space[path][0]), float(search_space[path][1])) for path in paths]
    return paths, values, bounds


def _vector_to_tunable_config(
    base_config: StudentTunableConfig,
    paths: Sequence[str],
    values: Sequence[float],
) -> StudentTunableConfig:
    mutable = asdict(base_config)
    for path, value in zip(paths, values, strict=True):
        _set_nested(mutable, str(path), float(value))
    return normalize_tunable_config(StudentTunableConfig(**mutable))


def _clip_value(value: float, bounds: tuple[float, float]) -> float:
    return float(min(max(float(value), float(bounds[0])), float(bounds[1])))


def _merge_config_paths(
    base_config: StudentTunableConfig,
    source_config: StudentTunableConfig,
    paths: Iterable[str],
) -> StudentTunableConfig:
    mutable = asdict(base_config)
    source = asdict(source_config)
    for path in paths:
        _set_nested(mutable, str(path), _get_nested(source, str(path)))
    return normalize_tunable_config(StudentTunableConfig(**mutable))


def _localize_search_space(
    base_config: StudentTunableConfig,
    search_space: Mapping[str, tuple[float, float]],
    *,
    window_fraction: float,
    min_fraction: float,
) -> dict[str, tuple[float, float]]:
    payload = asdict(base_config)
    localized: dict[str, tuple[float, float]] = {}
    for path, (lo, hi) in search_space.items():
        lo_f = float(lo)
        hi_f = float(hi)
        span = max(hi_f - lo_f, 1.0e-9)
        center = _clip_value(float(_get_nested(payload, str(path))), (lo_f, hi_f))
        half_width = max(span * 0.5 * float(window_fraction), span * 0.5 * float(min_fraction))
        local_lo = max(lo_f, center - half_width)
        local_hi = min(hi_f, center + half_width)
        if local_hi <= local_lo:
            local_lo = center
            local_hi = center
        localized[str(path)] = (float(local_lo), float(local_hi))
    return localized


def _sample_gaussian_vector(
    mean: Sequence[float],
    std: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    *,
    rng: random.Random,
) -> list[float]:
    return [
        _clip_value(rng.gauss(float(mu), float(sigma)), bound)
        for mu, sigma, bound in zip(mean, std, bounds, strict=True)
    ]


def _mean_std_from_elites(
    vectors: Sequence[Sequence[float]],
    bounds: Sequence[tuple[float, float]],
    *,
    min_std_fraction: float,
) -> tuple[list[float], list[float]]:
    dims = len(bounds)
    if not vectors:
        raise ValueError("At least one elite vector is required.")
    mean: list[float] = []
    std: list[float] = []
    for dim in range(dims):
        values = [float(vector[dim]) for vector in vectors]
        dim_mean = sum(values) / float(len(values))
        variance = sum((value - dim_mean) * (value - dim_mean) for value in values) / float(len(values))
        lo, hi = bounds[dim]
        min_std = max(1.0e-4, float(hi - lo) * float(min_std_fraction))
        mean.append(_clip_value(dim_mean, (lo, hi)))
        std.append(max(math.sqrt(max(0.0, variance)), min_std))
    return mean, std


def sample_tunable_config(
    base_config: StudentTunableConfig,
    search_space: Mapping[str, tuple[float, float]],
    *,
    rng: random.Random,
) -> StudentTunableConfig:
    mutable = asdict(base_config)
    for path, bounds in search_space.items():
        lo, hi = bounds
        _set_nested(mutable, str(path), rng.uniform(float(lo), float(hi)))
    for field_name in ("surface_friction_scale", "surface_longitudinal_scale", "surface_lateral_scale"):
        mutable[field_name] = {
            str(name): float(value)
            for name, value in mutable.get(field_name, {}).items()
        }
    return normalize_tunable_config(StudentTunableConfig(**mutable))


def compute_rollout_loss(
    teacher_frames: Sequence[Mapping[str, Any]],
    student_frames: Sequence[Mapping[str, Any]],
    weights: ReplayLossWeights,
) -> dict[str, float]:
    num_frames = min(len(teacher_frames), len(student_frames))
    if num_frames <= 0:
        raise ValueError("Teacher and student rollouts must contain at least one frame.")

    accum = {
        "position_xy_mse": 0.0,
        "yaw_mse": 0.0,
        "speed_mse": 0.0,
        "yaw_rate_mse": 0.0,
        "wheel_speed_mse": 0.0,
        "steer_angle_mse": 0.0,
        "suspension_mse": 0.0,
        "terminal_position_xy_se": 0.0,
        "terminal_speed_se": 0.0,
    }
    wheel_speed_count = 0
    steer_count = 0
    suspension_count = 0

    for index in range(num_frames):
        teacher = teacher_frames[index]
        student = student_frames[index]

        teacher_vehicle = teacher["vehicle"]
        student_vehicle = student["vehicle"]

        teacher_pos_xy = teacher_vehicle["position_m"][:2]
        student_pos_xy = student_vehicle["position_m"][:2]
        dx = float(student_pos_xy[0]) - float(teacher_pos_xy[0])
        dy = float(student_pos_xy[1]) - float(teacher_pos_xy[1])
        accum["position_xy_mse"] += dx * dx + dy * dy

        yaw_error = _wrap_angle_rad(float(student_vehicle["yaw_rad"]) - float(teacher_vehicle["yaw_rad"]))
        accum["yaw_mse"] += yaw_error * yaw_error

        teacher_lin = teacher_vehicle["linear_velocity_mps"]
        student_lin = student_vehicle["linear_velocity_mps"]
        teacher_speed = math.hypot(float(teacher_lin[0]), float(teacher_lin[1]))
        student_speed = math.hypot(float(student_lin[0]), float(student_lin[1]))
        speed_error = student_speed - teacher_speed
        accum["speed_mse"] += speed_error * speed_error

        teacher_ang = teacher_vehicle["angular_velocity_rad_s"]
        student_ang = student_vehicle["angular_velocity_rad_s"]
        yaw_rate_error = float(student_ang[2]) - float(teacher_ang[2])
        accum["yaw_rate_mse"] += yaw_rate_error * yaw_rate_error

        teacher_wheels = {wheel["label"]: wheel for wheel in teacher.get("wheels", [])}
        student_wheels = {wheel["label"]: wheel for wheel in student.get("wheels", [])}
        shared_labels = sorted(set(teacher_wheels) & set(student_wheels))
        for label in shared_labels:
            teacher_wheel = teacher_wheels[label]
            student_wheel = student_wheels[label]

            wheel_speed_error = float(student_wheel["rotation_speed_rad_s"]) - float(teacher_wheel["rotation_speed_rad_s"])
            accum["wheel_speed_mse"] += wheel_speed_error * wheel_speed_error
            wheel_speed_count += 1

            steer_error = float(student_wheel["steer_angle_rad"]) - float(teacher_wheel["steer_angle_rad"])
            accum["steer_angle_mse"] += steer_error * steer_error
            steer_count += 1

            if "suspension_jounce" in teacher_wheel and "suspension_jounce" in student_wheel:
                suspension_error = float(student_wheel["suspension_jounce"]) - float(teacher_wheel["suspension_jounce"])
                accum["suspension_mse"] += suspension_error * suspension_error
                suspension_count += 1

    inv_frames = 1.0 / float(num_frames)
    accum["position_xy_mse"] *= inv_frames
    accum["yaw_mse"] *= inv_frames
    accum["speed_mse"] *= inv_frames
    accum["yaw_rate_mse"] *= inv_frames
    accum["wheel_speed_mse"] = accum["wheel_speed_mse"] / float(max(1, wheel_speed_count))
    accum["steer_angle_mse"] = accum["steer_angle_mse"] / float(max(1, steer_count))
    accum["suspension_mse"] = accum["suspension_mse"] / float(max(1, suspension_count))

    teacher_final = teacher_frames[num_frames - 1]["vehicle"]
    student_final = student_frames[num_frames - 1]["vehicle"]
    final_dx = float(student_final["position_m"][0]) - float(teacher_final["position_m"][0])
    final_dy = float(student_final["position_m"][1]) - float(teacher_final["position_m"][1])
    accum["terminal_position_xy_se"] = final_dx * final_dx + final_dy * final_dy
    teacher_final_lin = teacher_final["linear_velocity_mps"]
    student_final_lin = student_final["linear_velocity_mps"]
    teacher_final_speed = math.hypot(float(teacher_final_lin[0]), float(teacher_final_lin[1]))
    student_final_speed = math.hypot(float(student_final_lin[0]), float(student_final_lin[1]))
    final_speed_error = student_final_speed - teacher_final_speed
    accum["terminal_speed_se"] = final_speed_error * final_speed_error

    total = (
        float(weights.position_xy) * accum["position_xy_mse"]
        + float(weights.yaw) * accum["yaw_mse"]
        + float(weights.speed) * accum["speed_mse"]
        + float(weights.yaw_rate) * accum["yaw_rate_mse"]
        + float(weights.wheel_speed) * accum["wheel_speed_mse"]
        + float(weights.steer_angle) * accum["steer_angle_mse"]
        + float(weights.suspension) * accum["suspension_mse"]
        + float(weights.terminal_position_xy) * accum["terminal_position_xy_se"]
        + float(weights.terminal_speed) * accum["terminal_speed_se"]
    )

    return {
        "num_frames": float(num_frames),
        **accum,
        "total_loss": float(total),
    }


def _scaled_surface_patches(
    patches: Sequence[SurfacePatch],
    *,
    surface_scale: Mapping[str, float],
    surface_longitudinal_scale: Mapping[str, float],
    surface_lateral_scale: Mapping[str, float],
) -> list[SurfacePatch]:
    scaled: list[SurfacePatch] = []
    for patch in patches:
        isotropic_scale = _surface_scale_value(surface_scale, patch.name)
        longitudinal_scale = _surface_scale_value(surface_longitudinal_scale, patch.name)
        lateral_scale = _surface_scale_value(surface_lateral_scale, patch.name)
        scale = isotropic_scale * math.sqrt(max(1.0e-4, longitudinal_scale * lateral_scale))
        scaled.append(
            SurfacePatch(
                name=patch.name,
                x_center_m=patch.x_center_m,
                y_center_m=patch.y_center_m,
                length_m=patch.length_m,
                width_m=patch.width_m,
                static_friction=patch.static_friction * scale,
                dynamic_friction=patch.dynamic_friction * scale,
                tire_friction=patch.tire_friction * scale,
                color_srgb=patch.color_srgb,
            )
        )
    return scaled


def _spawn_x_from_patches(patches: Sequence[SurfacePatch]) -> float:
    if not patches:
        return 0.0
    total_track_length = float(sum(float(patch.length_m) for patch in patches))
    first_patch_length = float(patches[0].length_m)
    return -0.5 * total_track_length + 0.35 * first_patch_length


def _build_student_wheel_paths(root_path: str) -> dict[str, str]:
    return {
        "front_left": f"{root_path}/front_left_wheel_link",
        "front_right": f"{root_path}/front_right_wheel_link",
        "rear_left": f"{root_path}/rear_left_wheel_link",
        "rear_right": f"{root_path}/rear_right_wheel_link",
    }


def _body_lateral_axis_from_quat_wxyz(quat_wxyz: Sequence[float]) -> tuple[float, float, float]:
    yaw = _yaw_from_quat_wxyz(quat_wxyz)
    return (-math.sin(yaw), math.cos(yaw), 0.0)


def _wheel_surface_names(
    *,
    stage: Any,
    wheel_paths: Mapping[str, str],
    patches: Sequence[SurfacePatch],
) -> dict[str, str | None]:
    import omni.usd

    names: dict[str, str | None] = {}
    for label, wheel_path in wheel_paths.items():
        wheel_prim = stage.GetPrimAtPath(wheel_path)
        if not wheel_prim.IsValid():
            names[label] = None
            continue
        wheel_translation = omni.usd.get_world_transform_matrix(wheel_prim).ExtractTranslation()
        patch = select_surface_patch(patches, float(wheel_translation[0]), float(wheel_translation[1]))
        names[str(label)] = None if patch is None else str(patch.name)
    return names


def _build_dof_name_to_index(dof_names: Sequence[str]) -> dict[str, int]:
    return {str(name): int(index) for index, name in enumerate(dof_names)}


def _required_joint_indices(dof_name_to_index: Mapping[str, int]) -> dict[str, list[int]]:
    required = {
        "steer": ["front_left_steer_joint", "front_right_steer_joint"],
        "drive": ["front_left_wheel_joint", "front_right_wheel_joint"],
        "brake": [
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ],
        "suspension": [
            "front_left_suspension_joint",
            "front_right_suspension_joint",
            "rear_left_suspension_joint",
            "rear_right_suspension_joint",
        ],
    }
    resolved: dict[str, list[int]] = {}
    for group, names in required.items():
        missing = [name for name in names if name not in dof_name_to_index]
        if missing:
            raise RuntimeError(f"Student articulation is missing required joints for '{group}': {missing}")
        resolved[group] = [int(dof_name_to_index[name]) for name in names]
    return resolved


def _set_prim_attr(prim: Any, name: str, value: Any) -> None:
    attribute = prim.GetAttribute(str(name))
    if attribute.IsValid():
        attribute.Set(value)


def _apply_runtime_student_dynamics(
    *,
    stage: Any,
    student_root_path: str,
    config: StudentTunableConfig,
) -> None:
    from pxr import Gf

    chassis_prim = stage.GetPrimAtPath(f"{student_root_path}/base_link")
    if chassis_prim.IsValid():
        _set_prim_attr(
            chassis_prim,
            "physics:centerOfMass",
            Gf.Vec3f(0.0, 0.0, float(config.chassis_com_height_offset_m)),
        )

    wheel_labels = ("front_left", "front_right", "rear_left", "rear_right")
    for label in wheel_labels:
        wheel_prim = stage.GetPrimAtPath(f"{student_root_path}/{label}_wheel_link")
        if not wheel_prim.IsValid():
            continue
        original_mass = float(wheel_prim.GetAttribute("physics:mass").Get() or 20.0)
        original_inertia = wheel_prim.GetAttribute("physics:diagonalInertia").Get()
        if original_inertia is None:
            continue
        mass_ratio = float(config.wheel_mass_kg) / max(1.0e-6, original_mass)
        inertia_scale = mass_ratio * float(config.wheel_inertia_scale)
        _set_prim_attr(wheel_prim, "physics:mass", float(config.wheel_mass_kg))
        _set_prim_attr(
            wheel_prim,
            "physics:diagonalInertia",
            Gf.Vec3f(
                float(original_inertia[0]) * inertia_scale,
                float(original_inertia[1]) * inertia_scale,
                float(original_inertia[2]) * inertia_scale,
            ),
        )

    for label in wheel_labels:
        suspension_prim = stage.GetPrimAtPath(f"{student_root_path}/joints/{label}_suspension_joint")
        if not suspension_prim.IsValid():
            continue
        _set_prim_attr(suspension_prim, "drive:linear:physics:stiffness", float(config.suspension_stiffness_n_m))
        _set_prim_attr(suspension_prim, "drive:linear:physics:damping", float(config.suspension_damping_n_s_m))

    steer_limit_deg = math.degrees(float(config.steering_limit_rad))
    for label in ("front_left", "front_right"):
        steer_prim = stage.GetPrimAtPath(f"{student_root_path}/joints/{label}_steer_joint")
        if not steer_prim.IsValid():
            continue
        _set_prim_attr(steer_prim, "physics:lowerLimit", -steer_limit_deg)
        _set_prim_attr(steer_prim, "physics:upperLimit", steer_limit_deg)


def _sign_with_memory(values: Sequence[float], memory: Sequence[float]) -> list[float]:
    signs: list[float] = []
    for value, prev in zip(values, memory, strict=True):
        if abs(float(value)) > 1.0e-4:
            signs.append(1.0 if float(value) >= 0.0 else -1.0)
        else:
            signs.append(1.0 if float(prev) >= 0.0 else -1.0)
    return signs


def _compute_student_efforts(
    *,
    command: VehicleCommand,
    dof_positions: Sequence[float],
    dof_velocities: Sequence[float],
    joint_ids: Mapping[str, list[int]],
    config: StudentTunableConfig,
    brake_sign_memory: Sequence[float],
    wheel_surface_names: Mapping[str, str | None],
) -> tuple[list[float], list[float]]:
    num_dofs = len(dof_positions)
    efforts = [0.0] * num_dofs

    steer_target = float(command.steering) * float(config.steering_limit_rad)
    for joint_id in joint_ids["steer"]:
        position_error = steer_target - float(dof_positions[joint_id])
        velocity_error = -float(dof_velocities[joint_id])
        steer_effort = (
            float(config.steering_kp_nm_per_rad) * position_error
            + float(config.steering_kd_nm_s_per_rad) * velocity_error
        )
        steer_effort = _clamp(
            steer_effort,
            -float(config.steering_effort_limit_nm),
            float(config.steering_effort_limit_nm),
        )
        efforts[joint_id] = float(steer_effort)

    drive_effort = float(command.accelerator) * float(config.drive_torque_nm)
    for label, joint_id in zip(("front_left", "front_right"), joint_ids["drive"], strict=True):
        surface_name = wheel_surface_names.get(label)
        longitudinal_scale = _surface_scale_value(config.surface_longitudinal_scale, surface_name)
        efforts[joint_id] += drive_effort * longitudinal_scale

    brake_joint_ids = joint_ids["brake"]
    brake_joint_velocities = [float(dof_velocities[joint_id]) for joint_id in brake_joint_ids]
    brake_sign = _sign_with_memory(brake_joint_velocities, brake_sign_memory)
    brake_efforts = [
        float(command.brake) * float(config.brake_front_torque_nm),
        float(command.brake) * float(config.brake_front_torque_nm),
        float(command.brake) * float(config.brake_rear_torque_nm),
        float(command.brake) * float(config.brake_rear_torque_nm),
    ]
    for label, joint_id, sign, brake_effort in zip(
        ("front_left", "front_right", "rear_left", "rear_right"),
        brake_joint_ids,
        brake_sign,
        brake_efforts,
        strict=True,
    ):
        surface_name = wheel_surface_names.get(label)
        longitudinal_scale = _surface_scale_value(config.surface_longitudinal_scale, surface_name)
        efforts[joint_id] -= brake_effort * longitudinal_scale * sign

    return efforts, brake_sign


def _compute_lateral_stabilization_wrench(
    *,
    linear_velocity_mps: Sequence[float],
    angular_velocity_rad_s: Sequence[float],
    orientation_wxyz: Sequence[float],
    config: StudentTunableConfig,
    wheel_surface_names: Mapping[str, str | None],
) -> tuple[list[float], list[float]]:
    surface_values = [
        _surface_scale_value(config.surface_lateral_scale, surface_name)
        for surface_name in wheel_surface_names.values()
    ]
    lateral_scale = sum(surface_values) / float(max(1, len(surface_values)))
    if lateral_scale <= 0.0:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    lateral_axis = _body_lateral_axis_from_quat_wxyz(orientation_wxyz)
    lateral_speed_mps = sum(
        float(linear_velocity_mps[index]) * float(lateral_axis[index])
        for index in range(3)
    )
    lateral_force_mag = -float(config.lateral_velocity_damping_n_per_mps) * lateral_scale * lateral_speed_mps
    yaw_torque_mag = -float(config.yaw_stability_damping_nm_per_rad_s) * lateral_scale * float(angular_velocity_rad_s[2])
    force_global = [lateral_force_mag * float(component) for component in lateral_axis]
    torque_global = [0.0, 0.0, yaw_torque_mag]
    return force_global, torque_global


def _collect_student_frame(
    *,
    step_idx: int,
    sim_time_s: float,
    command: VehicleCommand,
    articulation: Any,
    stage: Any,
    wheel_paths: Mapping[str, str],
    joint_ids: Mapping[str, list[int]],
    patches: Sequence[SurfacePatch],
    dof_names: Sequence[str],
) -> dict[str, Any]:
    import omni.usd

    dof_positions = articulation.get_dof_positions().numpy()[0]
    dof_velocities = articulation.get_dof_velocities().numpy()[0]
    root_positions, root_orientations = articulation.get_world_poses()
    linear_velocities, angular_velocities = articulation.get_velocities()

    position_m = [float(v) for v in root_positions.numpy()[0]]
    orientation_wxyz = [float(v) for v in root_orientations.numpy()[0]]
    linear_velocity_mps = [float(v) for v in linear_velocities.numpy()[0]]
    angular_velocity_rad_s = [float(v) for v in angular_velocities.numpy()[0]]
    steer_joint_positions = joint_ids["steer"]
    suspension_positions = joint_ids["suspension"]

    wheels = []
    for wheel_index, (label, wheel_path) in enumerate(wheel_paths.items()):
        wheel_prim = stage.GetPrimAtPath(wheel_path)
        if not wheel_prim.IsValid():
            raise RuntimeError(f"Student wheel prim does not exist: {wheel_path}")
        wheel_translation = omni.usd.get_world_transform_matrix(wheel_prim).ExtractTranslation()
        wheel_center = [float(wheel_translation[0]), float(wheel_translation[1]), float(wheel_translation[2])]
        patch = select_surface_patch(patches, wheel_center[0], wheel_center[1])
        if label.startswith("front_"):
            steer_angle = float(dof_positions[steer_joint_positions[0 if label.endswith("left") else 1]])
        else:
            steer_angle = 0.0
        if label == "front_left":
            suspension_jounce = float(dof_positions[suspension_positions[0]])
        elif label == "front_right":
            suspension_jounce = float(dof_positions[suspension_positions[1]])
        elif label == "rear_left":
            suspension_jounce = float(dof_positions[suspension_positions[2]])
        else:
            suspension_jounce = float(dof_positions[suspension_positions[3]])
        wheel_joint_name = f"{label}_wheel_joint"
        wheel_joint_index = dof_names.index(wheel_joint_name)
        wheels.append(
            {
                "index": int(wheel_index),
                "label": str(label),
                "path": str(wheel_path),
                "surface_name": None if patch is None else str(patch.name),
                "wheel_center_position_m": wheel_center,
                "rotation_speed_rad_s": float(dof_velocities[wheel_joint_index]),
                "rotation_angle_rad": float(dof_positions[wheel_joint_index]),
                "steer_angle_rad": float(steer_angle),
                "suspension_jounce": float(suspension_jounce),
            }
        )

    return {
        "step": int(step_idx),
        "sim_time_s": float(sim_time_s),
        "command": command.clamped().to_dict(),
        "vehicle": {
            "position_m": position_m,
            "orientation_wxyz": orientation_wxyz,
            "yaw_rad": float(_yaw_from_quat_wxyz(orientation_wxyz)),
            "linear_velocity_mps": linear_velocity_mps,
            "angular_velocity_rad_s": angular_velocity_rad_s,
        },
        "wheels": wheels,
    }


def _load_command_from_frame(frame: Mapping[str, Any]) -> VehicleCommand:
    payload = frame.get("command", {})
    return VehicleCommand(
        accelerator=float(payload.get("accelerator", 0.0)),
        steering=float(payload.get("steering", 0.0)),
        brake=float(payload.get("brake", 0.0)),
    ).clamped()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _aggregate_loss_dicts(losses: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not losses:
        raise ValueError("At least one loss dictionary is required.")
    keys = sorted({str(key) for loss in losses for key in loss.keys()})
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [float(loss[key]) for loss in losses if key in loss]
        if not values:
            continue
        if key == "num_frames":
            aggregated[key] = float(sum(values))
        else:
            aggregated[key] = float(sum(values) / len(values))
    return aggregated


def _run_multi_rollout_trial(
    *,
    app: Any,
    timeline: Any,
    usd_context: Any,
    teachers: Sequence[TeacherRollout],
    student_usd_path: Path,
    output_dir: Path,
    config: StudentTunableConfig,
    weights: ReplayLossWeights,
    warmup_steps: int,
    settle_steps: int,
    spawn_height_m: float,
    max_steps: int,
) -> dict[str, Any]:
    per_rollout: dict[str, dict[str, Any]] = {}
    loss_dicts: list[dict[str, float]] = []
    total_frames = 0
    for teacher in teachers:
        rollout_output_dir = output_dir / teacher.name
        result = _run_single_trial(
            app=app,
            timeline=timeline,
            usd_context=usd_context,
            teacher=teacher,
            student_usd_path=student_usd_path,
            output_dir=rollout_output_dir,
            config=config,
            weights=weights,
            warmup_steps=warmup_steps,
            settle_steps=settle_steps,
            spawn_height_m=spawn_height_m,
            max_steps=max_steps,
            save_stage_usd="",
        )
        per_rollout[teacher.name] = {
            "teacher_rollout_dir": str(teacher.rollout_dir),
            "output_dir": str(rollout_output_dir),
            "loss": result["loss"],
            "num_frames": int(len(result["frames"])),
        }
        loss_dicts.append(result["loss"])
        total_frames += int(len(result["frames"]))
    aggregate_loss = _aggregate_loss_dicts(loss_dicts)
    summary = {
        "teacher_rollout_dirs": {teacher.name: str(teacher.rollout_dir) for teacher in teachers},
        "student_usd_path": str(student_usd_path),
        "student_config": asdict(config),
        "loss_weights": asdict(weights),
        "per_rollout": per_rollout,
        "aggregate_loss": aggregate_loss,
        "num_rollouts": int(len(teachers)),
        "num_frames": int(total_frames),
    }
    _write_json(output_dir / "trial_bundle_summary.json", summary)
    return {
        "loss": aggregate_loss,
        "per_rollout": per_rollout,
        "num_frames": total_frames,
    }


def _teacher_by_name(dataset: TeacherDataset, name: str) -> TeacherRollout | None:
    for teacher in dataset.rollouts:
        if teacher.name == str(name):
            return teacher
    return None


def _select_dataset_teachers(dataset: TeacherDataset, names: Sequence[str]) -> list[TeacherRollout]:
    selected = []
    for name in names:
        teacher = _teacher_by_name(dataset, name)
        if teacher is not None:
            selected.append(teacher)
    return selected


def _select_dataset_teachers_by_prefix(dataset: TeacherDataset, prefixes: Sequence[str]) -> list[TeacherRollout]:
    """Select all rollouts whose name starts with any of the given prefixes."""
    selected = []
    seen: set[str] = set()
    for prefix in prefixes:
        for teacher in dataset.rollouts:
            if teacher.name.startswith(prefix) and teacher.name not in seen:
                selected.append(teacher)
                seen.add(teacher.name)
    return selected


def _select_report_teachers(dataset: TeacherDataset) -> list[TeacherRollout]:
    priority = ["straight_accel_brake", "step_steer_left", "sine_steer"]
    selected = _select_dataset_teachers(dataset, priority)
    if len(selected) >= 3:
        return selected[:3]
    # Prefix fallback for comprehensive suites
    if not selected:
        selected = _select_dataset_teachers_by_prefix(dataset, ["straight_accel_t", "step_steer_left_", "sine_steer_"])
        if len(selected) >= 3:
            return selected[:3]
    seen = {teacher.name for teacher in selected}
    for teacher in dataset.rollouts:
        if teacher.name in seen:
            continue
        selected.append(teacher)
        seen.add(teacher.name)
        if len(selected) >= 3:
            break
    return selected


def _allocate_stage_trials(total_trials: int, num_stages: int) -> list[int]:
    if num_stages <= 0:
        return []
    if total_trials <= 0:
        return [0] * num_stages
    weights = [0.30, 0.20, 0.15, 0.20, 0.15][:num_stages]
    if len(weights) < num_stages:
        weights.extend([1.0 / float(num_stages)] * (num_stages - len(weights)))
    total_weight = sum(weights)
    normalized = [weight / total_weight for weight in weights]
    raw = [float(total_trials) * weight for weight in normalized]
    counts = [int(math.floor(value)) for value in raw]
    remainder = int(total_trials - sum(counts))
    order = sorted(range(num_stages), key=lambda index: raw[index] - counts[index], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def build_staged_search_plan(dataset: TeacherDataset, total_random_trials: int) -> list[SearchStage]:
    # Exact-name matches (legacy suite) with prefix fallback (comprehensive suite)
    straight_rollouts = _select_dataset_teachers(dataset, ["straight_accel_brake"])
    if not straight_rollouts:
        straight_rollouts = _select_dataset_teachers_by_prefix(dataset, ["straight_accel_", "straight_brake_", "ramp_throttle_"])
    steering_rollouts = _select_dataset_teachers(
        dataset,
        ["step_steer_left", "step_steer_right", "constant_steer_left", "constant_steer_right", "sine_steer"],
    )
    if not steering_rollouts:
        steering_rollouts = _select_dataset_teachers_by_prefix(
            dataset, ["step_steer_", "const_steer_", "sine_steer_", "chirp_steer_", "trail_brake_"]
        )
    surface_rollouts = _select_dataset_teachers(dataset, ["surface_transition_s"])
    refinement_rollouts = list(dataset.rollouts)

    stage_teachers: list[
        tuple[
            str,
            str,
            list[TeacherRollout],
            dict[str, tuple[float, float]],
            float | None,
            float | None,
            str | None,
        ]
    ] = []
    if straight_rollouts:
        stage_surfaces = sorted({surface for teacher in straight_rollouts for surface in touched_surface_names(teacher)})
        stage_teachers.append(
            (
                "longitudinal",
                "Fit drive, brake, wheel damping, and touched-surface friction from straight accel/brake.",
                straight_rollouts,
                _longitudinal_search_space(stage_surfaces),
                None,
                None,
                None,
            )
        )
    if steering_rollouts:
        stage_teachers.append(
            (
                "steering",
                "Fit steering response from step, constant, and sine-steer maneuvers.",
                steering_rollouts,
                _steering_search_space(),
                None,
                None,
                None,
            )
        )
    if surface_rollouts:
        stage_surfaces = sorted({surface for teacher in surface_rollouts for surface in touched_surface_names(teacher)})
        stage_teachers.append(
            (
                "surface",
                "Fit only the surface friction scales for the surfaces touched in the transition maneuver.",
                surface_rollouts,
                _surface_search_space(stage_surfaces),
                None,
                None,
                None,
            )
        )
    if refinement_rollouts:
        stage_surfaces = sorted({surface for teacher in refinement_rollouts for surface in touched_surface_names(teacher)})
        stage_teachers.append(
            (
                "refinement",
                "Refine the combined parameters across the full suite, but only in a local window around the staged winners.",
                refinement_rollouts,
                _joint_refinement_search_space(stage_surfaces),
                0.18,
                0.05,
                None,
            )
        )
    if straight_rollouts:
        stage_surfaces = sorted({surface for teacher in straight_rollouts for surface in touched_surface_names(teacher)})
        stage_teachers.append(
            (
                "brake_preservation",
                "Re-anchor straight-line accel/brake distance after joint refinement without reopening steering search.",
                straight_rollouts,
                _longitudinal_search_space(stage_surfaces),
                0.10,
                0.03,
                "longitudinal",
            )
        )

    trial_counts = _allocate_stage_trials(int(total_random_trials), len(stage_teachers))
    return [
        SearchStage(
            name=name,
            description=description,
            teachers=teachers,
            search_space=search_space,
            random_trials=trial_counts[index],
            search_window_fraction=window_fraction,
            search_min_fraction=min_fraction,
            seed_from_stage=seed_from_stage,
        )
        for index, (name, description, teachers, search_space, window_fraction, min_fraction, seed_from_stage) in enumerate(stage_teachers)
    ]


def _run_single_trial(
    *,
    app: Any,
    timeline: Any,
    usd_context: Any,
    teacher: TeacherRollout,
    student_usd_path: Path,
    output_dir: Path,
    config: StudentTunableConfig,
    weights: ReplayLossWeights,
    warmup_steps: int,
    settle_steps: int,
    spawn_height_m: float,
    max_steps: int,
    save_stage_usd: str,
) -> dict[str, Any]:
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    usd_context.new_stage()
    stage = usd_context.get_stage()
    _ensure_world_default_prim(stage)
    _set_stage_units(stage)
    _ensure_physics_scene(stage)

    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/DistantLight"))
    light.CreateIntensityAttr(700.0)

    scaled_patches = _scaled_surface_patches(
        teacher.patches,
        surface_scale=config.surface_friction_scale,
        surface_longitudinal_scale=config.surface_longitudinal_scale,
        surface_lateral_scale=config.surface_lateral_scale,
    )
    _build_track_and_materials(stage, root_path="/World/PatchTrack", patches=scaled_patches)

    student_spawn_path = "/World/PatchTrack/StudentSpawn"
    student_root_path = f"{student_spawn_path}/Student"
    student_spawn_xform = UsdGeom.Xform.Define(stage, student_spawn_path)
    spawn_api = UsdGeom.XformCommonAPI(student_spawn_xform)
    spawn_api.SetTranslate(
        Gf.Vec3d(
            float(_spawn_x_from_patches(scaled_patches)),
            0.0,
            float(spawn_height_m),
        )
    )
    spawn_api.SetRotate(Gf.Vec3f(0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    student_xform = UsdGeom.Xform.Define(stage, student_root_path)
    student_xform.GetPrim().GetReferences().AddReference(str(student_usd_path))

    for _ in range(5):
        app.update()

    _apply_runtime_student_dynamics(stage=stage, student_root_path=student_root_path, config=config)
    for _ in range(2):
        app.update()

    articulation = Articulation(student_root_path)
    base_rigid = SingleRigidPrim(f"{student_root_path}/base_link")
    wheel_paths = _build_student_wheel_paths(student_root_path)

    timeline.play()
    for _ in range(max(1, int(warmup_steps))):
        app.update()
    base_rigid.initialize()

    dof_names = [str(name) for name in articulation.dof_names]
    dof_name_to_index = _build_dof_name_to_index(dof_names)
    joint_ids = _required_joint_indices(dof_name_to_index)
    controlled_ids = joint_ids["steer"] + joint_ids["drive"] + [
        joint_id for joint_id in joint_ids["brake"] if joint_id not in joint_ids["drive"]
    ]
    articulation.switch_dof_control_mode("effort", dof_indices=controlled_ids)

    zero_state = [[0.0] * len(dof_names)]
    articulation.set_dof_positions(zero_state)
    articulation.set_dof_velocities(zero_state)
    articulation.set_velocities([[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]])

    articulation.set_dof_friction_properties(
        viscous_frictions=[[float(config.steering_viscous_friction)] * len(joint_ids["steer"])],
        dof_indices=joint_ids["steer"],
    )
    articulation.set_dof_friction_properties(
        viscous_frictions=[[float(config.wheel_viscous_friction)] * len(joint_ids["brake"])],
        dof_indices=joint_ids["brake"],
    )
    articulation.set_dof_friction_properties(
        viscous_frictions=[[float(config.suspension_viscous_friction)] * len(joint_ids["suspension"])],
        dof_indices=joint_ids["suspension"],
    )

    brake_sign_memory = [1.0, 1.0, 1.0, 1.0]
    zero_command = VehicleCommand(0.0, 0.0, 0.0)
    for _ in range(max(0, int(settle_steps))):
        dof_positions = articulation.get_dof_positions().numpy()[0]
        dof_velocities = articulation.get_dof_velocities().numpy()[0]
        wheel_surface_names = _wheel_surface_names(stage=stage, wheel_paths=wheel_paths, patches=scaled_patches)
        efforts, brake_sign_memory = _compute_student_efforts(
            command=zero_command,
            dof_positions=dof_positions,
            dof_velocities=dof_velocities,
            joint_ids=joint_ids,
            config=config,
            brake_sign_memory=brake_sign_memory,
            wheel_surface_names=wheel_surface_names,
        )
        articulation.set_dof_efforts([efforts])
        base_linear_velocity = base_rigid.get_linear_velocity()
        base_angular_velocity = base_rigid.get_angular_velocity()
        _, base_orientation = base_rigid.get_world_pose()
        force_global, torque_global = _compute_lateral_stabilization_wrench(
            linear_velocity_mps=base_linear_velocity,
            angular_velocity_rad_s=base_angular_velocity,
            orientation_wxyz=base_orientation,
            config=config,
            wheel_surface_names=wheel_surface_names,
        )
        base_rigid._rigid_prim_view.apply_forces_and_torques_at_pos(
            forces=[force_global],
            torques=[torque_global],
            positions=[[0.0, 0.0, 0.0]],
            is_global=True,
        )
        app.update()

    teacher_frames = teacher.frames[: max_steps if max_steps > 0 else len(teacher.frames)]
    student_frames: list[dict[str, Any]] = []
    for step_idx, teacher_frame in enumerate(teacher_frames):
        command = _load_command_from_frame(teacher_frame)
        dof_positions = articulation.get_dof_positions().numpy()[0]
        dof_velocities = articulation.get_dof_velocities().numpy()[0]
        wheel_surface_names = _wheel_surface_names(stage=stage, wheel_paths=wheel_paths, patches=scaled_patches)
        efforts, brake_sign_memory = _compute_student_efforts(
            command=command,
            dof_positions=dof_positions,
            dof_velocities=dof_velocities,
            joint_ids=joint_ids,
            config=config,
            brake_sign_memory=brake_sign_memory,
            wheel_surface_names=wheel_surface_names,
        )
        articulation.set_dof_efforts([efforts])
        base_linear_velocity = base_rigid.get_linear_velocity()
        base_angular_velocity = base_rigid.get_angular_velocity()
        _, base_orientation = base_rigid.get_world_pose()
        force_global, torque_global = _compute_lateral_stabilization_wrench(
            linear_velocity_mps=base_linear_velocity,
            angular_velocity_rad_s=base_angular_velocity,
            orientation_wxyz=base_orientation,
            config=config,
            wheel_surface_names=wheel_surface_names,
        )
        base_rigid._rigid_prim_view.apply_forces_and_torques_at_pos(
            forces=[force_global],
            torques=[torque_global],
            positions=[[0.0, 0.0, 0.0]],
            is_global=True,
        )
        app.update()
        student_frames.append(
            _collect_student_frame(
                step_idx=step_idx,
                sim_time_s=float(teacher_frame.get("sim_time_s", (step_idx + 1) * teacher.metadata["dt_s"])),
                command=command,
                articulation=articulation,
                stage=stage,
                wheel_paths=wheel_paths,
                joint_ids=joint_ids,
                patches=scaled_patches,
                dof_names=dof_names,
            )
        )

    articulation.set_dof_efforts([[0.0] * len(dof_names)])
    for _ in range(3):
        app.update()
    timeline.stop()

    loss = compute_rollout_loss(teacher_frames, student_frames, weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "teacher_rollout_dir": str(teacher.rollout_dir),
        "student_usd_path": str(student_usd_path),
        "student_root_path": student_root_path,
        "student_config": asdict(config),
        "loss_weights": asdict(weights),
        "num_teacher_frames": int(len(teacher_frames)),
        "num_student_frames": int(len(student_frames)),
        "warmup_steps": int(warmup_steps),
        "settle_steps": int(settle_steps),
        "spawn_height_m": float(spawn_height_m),
    }
    _write_json(output_dir / "student_sysid_meta.json", meta)
    _write_json(output_dir / "loss.json", loss)
    with (output_dir / "student_rollout_frames.jsonl").open("w", encoding="utf-8") as stream:
        for frame in student_frames:
            stream.write(json.dumps(frame) + "\n")

    if save_stage_usd:
        save_path = Path(save_stage_usd).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        usd_context.save_as_stage(str(save_path))

    return {
        "frames": student_frames,
        "loss": loss,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded teacher commands on the handcrafted student vehicle and score kinematic error."
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--teacher-rollout-dir", type=str, default="")
    parser.add_argument("--teacher-dataset-manifest", type=str, default="")
    parser.add_argument("--student-usd", type=str, default=DEFAULT_STUDENT_USD)
    parser.add_argument("--output-dir", type=str, default="artifacts/student_vehicle_sysid/default")
    parser.add_argument("--tunable-config-json", type=str, default="")
    parser.add_argument("--loss-weights-json", type=str, default="")
    parser.add_argument("--search-space-json", type=str, default="")
    parser.add_argument("--search-mode", type=str, choices=("auto", "full", "staged"), default="auto")
    parser.add_argument("--optimizer", type=str, choices=("cem", "random"), default="cem")
    parser.add_argument("--random-search-trials", type=int, default=0)
    parser.add_argument("--random-search-seed", type=int, default=0)
    parser.add_argument("--cem-population-size", type=int, default=16)
    parser.add_argument("--cem-elite-fraction", type=float, default=0.25)
    parser.add_argument("--cem-initial-std-fraction", type=float, default=0.25)
    parser.add_argument("--cem-min-std-fraction", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=-1, help="-1 uses the recorded teacher settle_steps.")
    parser.add_argument("--spawn-height-m", type=float, default=1.2)
    parser.add_argument("--max-steps", type=int, default=0, help="0 replays the full teacher rollout.")
    parser.add_argument("--save-stage-usd", type=str, default="")

    parser.add_argument("--drive-torque-nm", type=float, default=None)
    parser.add_argument("--brake-front-torque-nm", type=float, default=None)
    parser.add_argument("--brake-rear-torque-nm", type=float, default=None)
    parser.add_argument("--steering-limit-rad", type=float, default=None)
    parser.add_argument("--steering-kp-nm-per-rad", type=float, default=None)
    parser.add_argument("--steering-kd-nm-s-per-rad", type=float, default=None)
    parser.add_argument("--steering-effort-limit-nm", type=float, default=None)
    parser.add_argument("--wheel-viscous-friction", type=float, default=None)
    parser.add_argument("--steering-viscous-friction", type=float, default=None)
    parser.add_argument("--suspension-viscous-friction", type=float, default=None)
    parser.add_argument("--wheel-mass-kg", type=float, default=None)
    parser.add_argument("--wheel-inertia-scale", type=float, default=None)
    parser.add_argument("--suspension-stiffness-n-m", type=float, default=None)
    parser.add_argument("--suspension-damping-n-s-m", type=float, default=None)
    parser.add_argument("--chassis-com-height-offset-m", type=float, default=None)
    parser.add_argument("--lateral-velocity-damping-n-per-mps", type=float, default=None)
    parser.add_argument("--yaw-stability-damping-nm-per-rad-s", type=float, default=None)

    return parser.parse_args()


def _cli_tunable_config(args: argparse.Namespace) -> StudentTunableConfig:
    config = load_tunable_config(args.tunable_config_json) if str(args.tunable_config_json) else StudentTunableConfig()
    overrides = {
        "drive_torque_nm": args.drive_torque_nm,
        "brake_front_torque_nm": args.brake_front_torque_nm,
        "brake_rear_torque_nm": args.brake_rear_torque_nm,
        "steering_limit_rad": args.steering_limit_rad,
        "steering_kp_nm_per_rad": args.steering_kp_nm_per_rad,
        "steering_kd_nm_s_per_rad": args.steering_kd_nm_s_per_rad,
        "steering_effort_limit_nm": args.steering_effort_limit_nm,
        "wheel_viscous_friction": args.wheel_viscous_friction,
        "steering_viscous_friction": args.steering_viscous_friction,
        "suspension_viscous_friction": args.suspension_viscous_friction,
        "wheel_mass_kg": args.wheel_mass_kg,
        "wheel_inertia_scale": args.wheel_inertia_scale,
        "suspension_stiffness_n_m": args.suspension_stiffness_n_m,
        "suspension_damping_n_s_m": args.suspension_damping_n_s_m,
        "chassis_com_height_offset_m": args.chassis_com_height_offset_m,
        "lateral_velocity_damping_n_per_mps": args.lateral_velocity_damping_n_per_mps,
        "yaw_stability_damping_nm_per_rad_s": args.yaw_stability_damping_nm_per_rad_s,
    }
    if not any(value is not None for value in overrides.values()):
        return config
    mutable = asdict(config)
    for key, value in overrides.items():
        if value is not None:
            mutable[key] = float(value)
    return normalize_tunable_config(StudentTunableConfig(**mutable))


def _cem_settings_from_args(args: argparse.Namespace) -> CEMSettings:
    return CEMSettings(
        population_size=max(2, int(args.cem_population_size)),
        elite_fraction=min(max(float(args.cem_elite_fraction), 0.05), 0.95),
        initial_std_fraction=max(float(args.cem_initial_std_fraction), 1.0e-3),
        min_std_fraction=max(float(args.cem_min_std_fraction), 1.0e-4),
    )


def _resolve_teacher_inputs(args: argparse.Namespace) -> tuple[TeacherRollout | None, TeacherDataset | None]:
    rollout_dir = str(args.teacher_rollout_dir).strip()
    dataset_manifest = str(args.teacher_dataset_manifest).strip()
    if bool(rollout_dir) == bool(dataset_manifest):
        raise ValueError("Specify exactly one of --teacher-rollout-dir or --teacher-dataset-manifest.")
    if rollout_dir:
        return load_teacher_rollout(rollout_dir), None
    return None, load_teacher_dataset_manifest(dataset_manifest)


def _trial_specs(
    *,
    base_config: StudentTunableConfig,
    search_space: Mapping[str, tuple[float, float]],
    trials_dir: Path,
    random_search_trials: int,
    random_seed: int,
) -> list[tuple[str, StudentTunableConfig, Path]]:
    trials: list[tuple[str, StudentTunableConfig, Path]] = [("baseline", base_config, trials_dir / "baseline")]
    rng = random.Random(int(random_seed))
    for trial_index in range(max(0, int(random_search_trials))):
        sampled = sample_tunable_config(base_config, search_space, rng=rng)
        trials.append((f"sample_{trial_index:03d}", sampled, trials_dir / f"sample_{trial_index:03d}"))
    return trials


def _evaluate_trial(
    *,
    trial_name: str,
    trial_config: StudentTunableConfig,
    trial_output_dir: Path,
    app: Any,
    timeline: Any,
    usd_context: Any,
    teachers: Sequence[TeacherRollout],
    student_usd_path: Path,
    weights: ReplayLossWeights,
    warmup_steps: int,
    settle_steps: int,
    spawn_height_m: float,
    max_steps: int,
    history_path: Path,
    stage_name: str | None,
    save_stage_usd: str,
) -> TrialResult:
    if len(teachers) == 1:
        result = _run_single_trial(
            app=app,
            timeline=timeline,
            usd_context=usd_context,
            teacher=teachers[0],
            student_usd_path=student_usd_path,
            output_dir=trial_output_dir,
            config=trial_config,
            weights=weights,
            warmup_steps=warmup_steps,
            settle_steps=settle_steps,
            spawn_height_m=spawn_height_m,
            max_steps=max_steps,
            save_stage_usd=save_stage_usd,
        )
        num_frames = int(len(result["frames"]))
    else:
        result = _run_multi_rollout_trial(
            app=app,
            timeline=timeline,
            usd_context=usd_context,
            teachers=teachers,
            student_usd_path=student_usd_path,
            output_dir=trial_output_dir,
            config=trial_config,
            weights=weights,
            warmup_steps=warmup_steps,
            settle_steps=settle_steps,
            spawn_height_m=spawn_height_m,
            max_steps=max_steps,
        )
        num_frames = int(result["num_frames"])

    trial_result = TrialResult(
        name=trial_name,
        config=trial_config,
        loss=result["loss"],
        num_frames=num_frames,
        output_dir=trial_output_dir,
    )
    _write_history_entry(
        history_path=history_path,
        trial_result=trial_result,
        teachers=teachers,
        stage_name=stage_name,
    )
    return trial_result


def _write_history_entry(
    *,
    history_path: Path,
    trial_result: TrialResult,
    teachers: Sequence[TeacherRollout],
    stage_name: str | None = None,
) -> None:
    with history_path.open("a", encoding="utf-8") as stream:
        payload: dict[str, Any] = {
            "trial_name": trial_result.name,
            "output_dir": str(trial_result.output_dir),
            "num_frames": int(trial_result.num_frames),
            "loss": trial_result.loss,
            "config": asdict(trial_result.config),
            "teacher_rollout_names": [teacher.name for teacher in teachers],
        }
        if stage_name is not None:
            payload["stage_name"] = str(stage_name)
        stream.write(json.dumps(payload) + "\n")


def _run_search(
    *,
    app: Any,
    timeline: Any,
    usd_context: Any,
    teachers: Sequence[TeacherRollout],
    student_usd_path: Path,
    output_root: Path,
    base_config: StudentTunableConfig,
    search_space: Mapping[str, tuple[float, float]],
    weights: ReplayLossWeights,
    warmup_steps: int,
    settle_steps: int,
    spawn_height_m: float,
    max_steps: int,
    random_search_trials: int,
    random_seed: int,
    optimizer: str,
    cem_settings: CEMSettings,
    stage_name: str | None = None,
    save_stage_usd: str = "",
) -> TrialResult:
    history_path = output_root / "search_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        history_path.unlink()

    best_result: TrialResult | None = None
    trials_dir = output_root / "trials"
    budget = int(max(0, random_search_trials))
    if budget <= 0:
        best_result = _evaluate_trial(
            trial_name="single",
            trial_config=base_config,
            trial_output_dir=output_root,
            app=app,
            timeline=timeline,
            usd_context=usd_context,
            teachers=teachers,
            student_usd_path=student_usd_path,
            weights=weights,
            warmup_steps=warmup_steps,
            settle_steps=settle_steps,
            spawn_height_m=spawn_height_m,
            max_steps=max_steps,
            history_path=history_path,
            stage_name=stage_name,
            save_stage_usd=save_stage_usd,
        )
    elif str(optimizer) == "random":
        trials = _trial_specs(
            base_config=base_config,
            search_space=search_space,
            trials_dir=trials_dir,
            random_search_trials=budget,
            random_seed=random_seed,
        )
        for trial_name, trial_config, trial_output_dir in trials:
            trial_result = _evaluate_trial(
                trial_name=trial_name,
                trial_config=trial_config,
                trial_output_dir=trial_output_dir,
                app=app,
                timeline=timeline,
                usd_context=usd_context,
                teachers=teachers,
                student_usd_path=student_usd_path,
                weights=weights,
                warmup_steps=warmup_steps,
                settle_steps=settle_steps,
                spawn_height_m=spawn_height_m,
                max_steps=max_steps,
                history_path=history_path,
                stage_name=stage_name,
                save_stage_usd=save_stage_usd if len(trials) == 1 else "",
            )
            if best_result is None or float(trial_result.loss["total_loss"]) < float(best_result.loss["total_loss"]):
                best_result = trial_result
    else:
        rng = random.Random(int(random_seed))
        paths, mean_vector, bounds = _config_to_search_vector(base_config, search_space)
        if not paths:
            best_result = _evaluate_trial(
                trial_name="single",
                trial_config=base_config,
                trial_output_dir=output_root,
                app=app,
                timeline=timeline,
                usd_context=usd_context,
                teachers=teachers,
                student_usd_path=student_usd_path,
                weights=weights,
                warmup_steps=warmup_steps,
                settle_steps=settle_steps,
                spawn_height_m=spawn_height_m,
                max_steps=max_steps,
                history_path=history_path,
                stage_name=stage_name,
                save_stage_usd=save_stage_usd,
            )
        else:
            population_size = max(2, int(cem_settings.population_size))
            elite_count = max(1, min(population_size, int(math.ceil(population_size * float(cem_settings.elite_fraction)))))
            std_vector = [
                max(1.0e-4, float(hi - lo) * float(cem_settings.initial_std_fraction))
                for lo, hi in bounds
            ]
            baseline_result = _evaluate_trial(
                trial_name="baseline",
                trial_config=base_config,
                trial_output_dir=trials_dir / "baseline",
                app=app,
                timeline=timeline,
                usd_context=usd_context,
                teachers=teachers,
                student_usd_path=student_usd_path,
                weights=weights,
                warmup_steps=warmup_steps,
                settle_steps=settle_steps,
                spawn_height_m=spawn_height_m,
                max_steps=max_steps,
                history_path=history_path,
                stage_name=stage_name,
                save_stage_usd="",
            )
            best_result = baseline_result
            remaining = budget
            trial_index = 0
            while remaining > 0:
                batch_size = min(population_size, remaining)
                batch_results: list[tuple[TrialResult, list[float]]] = []
                for _ in range(batch_size):
                    candidate_vector = _sample_gaussian_vector(mean_vector, std_vector, bounds, rng=rng)
                    trial_config = _vector_to_tunable_config(base_config, paths, candidate_vector)
                    trial_result = _evaluate_trial(
                        trial_name=f"sample_{trial_index:03d}",
                        trial_config=trial_config,
                        trial_output_dir=trials_dir / f"sample_{trial_index:03d}",
                        app=app,
                        timeline=timeline,
                        usd_context=usd_context,
                        teachers=teachers,
                        student_usd_path=student_usd_path,
                        weights=weights,
                        warmup_steps=warmup_steps,
                        settle_steps=settle_steps,
                        spawn_height_m=spawn_height_m,
                        max_steps=max_steps,
                        history_path=history_path,
                        stage_name=stage_name,
                        save_stage_usd="",
                    )
                    batch_results.append((trial_result, candidate_vector))
                    if float(trial_result.loss["total_loss"]) < float(best_result.loss["total_loss"]):
                        best_result = trial_result
                    trial_index += 1
                elites = sorted(batch_results, key=lambda item: float(item[0].loss["total_loss"]))[:elite_count]
                mean_vector, std_vector = _mean_std_from_elites(
                    [vector for _, vector in elites],
                    bounds,
                    min_std_fraction=float(cem_settings.min_std_fraction),
                )
                remaining -= batch_size

    if best_result is None:
        raise RuntimeError("No replay trials were executed.")
    summary_payload = {
        "stage_name": None if stage_name is None else str(stage_name),
        "teacher_rollout_names": [teacher.name for teacher in teachers],
        "student_usd_path": str(student_usd_path),
        "best_trial_name": str(best_result.name),
        "best_trial_output_dir": str(best_result.output_dir),
        "best_loss": best_result.loss,
        "best_config": asdict(best_result.config),
        "num_trials": int(max(1, budget + 1 if budget > 0 else 1)),
        "optimizer": str(optimizer),
    }
    if len(teachers) == 1:
        summary_payload["teacher_rollout_dir"] = str(teachers[0].rollout_dir)
    _write_json(output_root / "best_result.json", summary_payload)
    _write_json(output_root / "best_config.json", asdict(best_result.config))
    return best_result


def main() -> int:
    args = _parse_args()
    teacher, dataset = _resolve_teacher_inputs(args)
    student_usd_path = Path(args.student_usd).expanduser().resolve()
    if not student_usd_path.exists():
        raise FileNotFoundError(f"Student USD does not exist: {student_usd_path}")

    base_config = _cli_tunable_config(args)
    weights = load_loss_weights(args.loss_weights_json) if str(args.loss_weights_json) else ReplayLossWeights()
    cem_settings = _cem_settings_from_args(args)
    max_steps = int(args.max_steps)
    output_root = Path(args.output_dir).expanduser().resolve()
    random_search_trials = int(args.random_search_trials)

    if teacher is not None:
        settle_steps = (
            int(args.settle_steps)
            if int(args.settle_steps) >= 0
            else int(teacher.metadata.get("settle_steps", 60))
        )
    elif dataset is not None:
        dataset_settle_steps = int(dataset.manifest.get("teacher_recording", {}).get("settle_steps", 60))
        settle_steps = int(args.settle_steps) if int(args.settle_steps) >= 0 else dataset_settle_steps
    else:
        raise RuntimeError("No teacher input was resolved.")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        import omni.kit.app
        import omni.timeline
        import omni.usd

        app = omni.kit.app.get_app()
        timeline = omni.timeline.get_timeline_interface()
        usd_context = omni.usd.get_context()

        if teacher is not None:
            if str(args.search_space_json):
                search_space = load_search_space(args.search_space_json)
            elif str(args.search_mode) == "full":
                search_space = default_search_space()
            else:
                search_space = auto_search_space_for_rollout(teacher)

            best_result = _run_search(
                app=app,
                timeline=timeline,
                usd_context=usd_context,
                teachers=[teacher],
                student_usd_path=student_usd_path,
                output_root=output_root,
                base_config=base_config,
                search_space=search_space,
                weights=weights,
                warmup_steps=int(args.warmup_steps),
                settle_steps=settle_steps,
                spawn_height_m=float(args.spawn_height_m),
                max_steps=max_steps,
                random_search_trials=random_search_trials,
                random_seed=int(args.random_search_seed),
                optimizer=str(args.optimizer),
                cem_settings=cem_settings,
                save_stage_usd=args.save_stage_usd,
            )
            summary = {
                "teacher_rollout_dir": str(teacher.rollout_dir),
                "student_usd_path": str(student_usd_path),
                "best_trial_name": str(best_result.name),
                "best_trial_output_dir": str(best_result.output_dir),
                "best_loss": best_result.loss,
                "best_config": asdict(best_result.config),
                "num_trials": int(max(1, random_search_trials + 1 if random_search_trials > 0 else 1)),
                "search_mode": str(args.search_mode),
                "optimizer": str(args.optimizer),
                "search_space": {key: [bounds[0], bounds[1]] for key, bounds in search_space.items()},
            }
            _write_json(output_root / "best_result.json", summary)
            _write_json(output_root / "best_config.json", asdict(best_result.config))
            report_path = write_sysid_report(output_root)
            summary["sysid_report_html"] = str(report_path)
            _write_json(output_root / "best_result.json", summary)
            print(
                f"[student_vehicle_sysid] best trial={best_result.name} total_loss={best_result.loss['total_loss']:.6f} "
                f"output_dir={best_result.output_dir}",
                flush=True,
            )
        else:
            if dataset is None:
                raise RuntimeError("No teacher rollout or dataset was provided.")
            if str(args.search_space_json) and str(args.search_mode) in {"auto", "staged"}:
                raise ValueError("Custom search-space JSON is only supported with --search-mode full in dataset mode.")

            if str(args.search_mode) == "full":
                all_surfaces = sorted({surface for rollout in dataset.rollouts for surface in touched_surface_names(rollout)})
                stages = [
                    SearchStage(
                        name="full_suite",
                        description="Single-stage joint refinement across the full teacher dataset.",
                        teachers=list(dataset.rollouts),
                        search_space=(
                            load_search_space(args.search_space_json)
                            if str(args.search_space_json)
                            else _joint_refinement_search_space(all_surfaces)
                        ),
                        random_trials=random_search_trials,
                    )
                ]
            else:
                stages = build_staged_search_plan(dataset, random_search_trials)

            stage_summaries: list[dict[str, Any]] = []
            current_config = base_config
            stage_best_by_name: dict[str, TrialResult] = {}
            representative_report_path = ""
            representative_rollout_dir = ""
            final_bundle_report_dir = ""
            for stage_index, stage in enumerate(stages, start=1):
                stage_output_dir = output_root / "stages" / f"{stage_index:02d}_{stage.name}"
                stage_base_config = current_config
                if stage.seed_from_stage is not None and stage.seed_from_stage in stage_best_by_name:
                    stage_base_config = _merge_config_paths(
                        current_config,
                        stage_best_by_name[stage.seed_from_stage].config,
                        stage.search_space.keys(),
                    )
                stage_search_space = dict(stage.search_space)
                if stage.search_window_fraction is not None:
                    stage_search_space = _localize_search_space(
                        stage_base_config,
                        stage_search_space,
                        window_fraction=float(stage.search_window_fraction),
                        min_fraction=float(
                            stage.search_min_fraction
                            if stage.search_min_fraction is not None
                            else stage.search_window_fraction
                        ),
                    )
                stage_best = _run_search(
                    app=app,
                    timeline=timeline,
                    usd_context=usd_context,
                    teachers=stage.teachers,
                    student_usd_path=student_usd_path,
                    output_root=stage_output_dir,
                    base_config=stage_base_config,
                    search_space=stage_search_space,
                    weights=weights,
                    warmup_steps=int(args.warmup_steps),
                    settle_steps=settle_steps,
                    spawn_height_m=float(args.spawn_height_m),
                    max_steps=max_steps,
                    random_search_trials=stage.random_trials,
                    random_seed=int(args.random_search_seed) + stage_index,
                    optimizer=str(args.optimizer),
                    cem_settings=cem_settings,
                    stage_name=stage.name,
                )
                current_config = stage_best.config
                stage_best_by_name[stage.name] = stage_best
                stage_summaries.append(
                    {
                        "name": stage.name,
                        "description": stage.description,
                        "teacher_rollout_names": [teacher_rollout.name for teacher_rollout in stage.teachers],
                        "random_trials": int(stage.random_trials),
                        "search_window_fraction": stage.search_window_fraction,
                        "search_min_fraction": stage.search_min_fraction,
                        "seed_from_stage": stage.seed_from_stage,
                        "best_trial_name": str(stage_best.name),
                        "best_trial_output_dir": str(stage_best.output_dir),
                        "best_loss": stage_best.loss,
                        "best_config": asdict(stage_best.config),
                        "search_space": {key: [bounds[0], bounds[1]] for key, bounds in stage_search_space.items()},
                    }
                )

            # Run a representative final replay on straight accel/brake if it exists, otherwise on the first rollout.
            representative_teacher = _teacher_by_name(dataset, "straight_accel_brake") or dataset.rollouts[0]
            representative_output_dir = output_root / "final_report"
            _run_search(
                app=app,
                timeline=timeline,
                usd_context=usd_context,
                teachers=[representative_teacher],
                student_usd_path=student_usd_path,
                output_root=representative_output_dir,
                base_config=current_config,
                search_space=auto_search_space_for_rollout(representative_teacher),
                weights=weights,
                warmup_steps=int(args.warmup_steps),
                settle_steps=settle_steps,
                spawn_height_m=float(args.spawn_height_m),
                max_steps=max_steps,
                random_search_trials=0,
                random_seed=int(args.random_search_seed),
                optimizer=str(args.optimizer),
                cem_settings=cem_settings,
            )
            representative_report_path = str(write_sysid_report(representative_output_dir))
            representative_rollout_dir = str(representative_teacher.rollout_dir)

            report_teachers = _select_report_teachers(dataset)
            if report_teachers:
                final_bundle_output_dir = output_root / "final_bundle_report"
                _run_multi_rollout_trial(
                    app=app,
                    timeline=timeline,
                    usd_context=usd_context,
                    teachers=report_teachers,
                    student_usd_path=student_usd_path,
                    output_dir=final_bundle_output_dir,
                    config=current_config,
                    weights=weights,
                    warmup_steps=int(args.warmup_steps),
                    settle_steps=settle_steps,
                    spawn_height_m=float(args.spawn_height_m),
                    max_steps=max_steps,
                )
                final_bundle_report_dir = str(final_bundle_output_dir)

            summary = {
                "teacher_dataset_manifest": str(dataset.manifest_path),
                "student_usd_path": str(student_usd_path),
                "search_mode": str(args.search_mode),
                "optimizer": str(args.optimizer),
                "best_config": asdict(current_config),
                "stages": stage_summaries,
                "representative_rollout_dir": representative_rollout_dir,
                "representative_report_html": representative_report_path,
                "final_bundle_report_dir": final_bundle_report_dir,
                "final_bundle_rollout_names": [teacher.name for teacher in report_teachers],
            }
            _write_json(output_root / "best_result.json", summary)
            _write_json(output_root / "best_config.json", asdict(current_config))
            aggregated_report_path = str(write_sysid_report(output_root))
            summary["sysid_report_html"] = aggregated_report_path
            _write_json(output_root / "best_result.json", summary)
            print(
                f"[student_vehicle_sysid] completed {len(stages)} staged search phases "
                f"output_dir={output_root}",
                flush=True,
            )
        return 0
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
