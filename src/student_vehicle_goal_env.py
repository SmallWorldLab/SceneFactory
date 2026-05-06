from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import re
from typing import Sequence

import gymnasium as gym
import numpy as np
import torch

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths
from src.student_vehicle_sysid import (
    StudentTunableConfig,
    _apply_runtime_student_dynamics,
    load_tunable_config,
    normalize_tunable_config,
)

ensure_isaaclab_source_paths()

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.markers import CUBOID_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, sample_uniform, subtract_frame_transforms


DEFAULT_STUDENT_VEHICLE_USD = "artifacts/student_vehicle_assets/vehicle_student/student_fwd_vehicle.usd"
_DEFAULT_TUNABLE_CONFIG_CANDIDATES = (
    "artifacts/student_vehicle_sysid/comprehensive_fwd_v1_cem_v4/best_config.json",
    "artifacts/student_vehicle_sysid/fwd_v1_staged_cem_anchor_overnight/best_config.json",
    "artifacts/student_vehicle_sysid/fwd_v1_staged_cem_big/best_config.json",
)
_GOAL_MARKER_POLE_SIZE = (0.34, 0.34, 2.20)
_GOAL_MARKER_POLE_CENTER_Z_M = 1.10
_GOAL_MARKER_CAP_RADIUS_M = 0.52
_GOAL_MARKER_CAP_CENTER_Z_M = 2.15


def _default_tunable_config_json() -> str:
    for candidate in _DEFAULT_TUNABLE_CONFIG_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return ""


def build_goal_beacon_marker(prim_path: str) -> VisualizationMarkers:
    marker_cfg = CUBOID_MARKER_CFG.copy()
    marker_cfg.markers = {
        "pole": sim_utils.CuboidCfg(
            size=_GOAL_MARKER_POLE_SIZE,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.48, 0.08),
                emissive_color=(0.35, 0.12, 0.02),
                roughness=0.18,
                metallic=0.0,
            ),
        ),
        "cap": sim_utils.SphereCfg(
            radius=_GOAL_MARKER_CAP_RADIUS_M,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 1.0, 0.28),
                emissive_color=(0.05, 0.42, 0.10),
                roughness=0.10,
                metallic=0.0,
            ),
        ),
    }
    marker_cfg.prim_path = str(prim_path)
    return VisualizationMarkers(marker_cfg)


def goal_beacon_visualization(goal_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return marker positions and prototype indices for a tall goal beacon."""
    pole_positions = goal_positions.clone()
    cap_positions = goal_positions.clone()
    pole_positions[:, 2] += float(_GOAL_MARKER_POLE_CENTER_Z_M)
    cap_positions[:, 2] += float(_GOAL_MARKER_CAP_CENTER_Z_M)
    positions = torch.cat([pole_positions, cap_positions], dim=0)
    marker_indices = torch.cat(
        [
            torch.zeros(goal_positions.shape[0], dtype=torch.int32, device=goal_positions.device),
            torch.ones(goal_positions.shape[0], dtype=torch.int32, device=goal_positions.device),
        ],
        dim=0,
    )
    return positions, marker_indices


def build_student_vehicle_articulation_cfg(
    usd_path: str,
    spawn_height_m: float = 1.6,
    prim_path: str = "/World/envs/env_.*/Vehicle",
) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=str(prim_path),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(usd_path).expanduser().resolve()),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=200.0,
                max_angular_velocity=200.0,
                max_depenetration_velocity=20.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(max_effort=5000.0, max_velocity=2000.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, float(spawn_height_m)),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "steering": ImplicitActuatorCfg(
                joint_names_expr=[".*_steer_joint"],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=5000.0,
                velocity_limit_sim=200.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=[".*_wheel_joint"],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=5000.0,
                velocity_limit_sim=2000.0,
            ),
        },
    )


def _source_env_vehicle_root_path(prim_path: str) -> str:
    return re.sub(r"env_\.\*", "env_0", str(prim_path))


def _dry_ground_material_cfg(config: StudentTunableConfig) -> sim_utils.RigidBodyMaterialCfg:
    dry_surface_scale = float(config.surface_friction_scale.get("dry_asphalt", 1.0))
    dry_longitudinal_scale = float(config.surface_longitudinal_scale.get("dry_asphalt", 1.0))
    dry_lateral_scale = float(config.surface_lateral_scale.get("dry_asphalt", 1.0))
    effective_scale = dry_surface_scale * math.sqrt(max(1.0e-4, dry_longitudinal_scale * dry_lateral_scale))
    static_friction = min(1.0, max(1.0e-3, 1.00 * effective_scale))
    dynamic_friction = min(static_friction, max(1.0e-3, 0.95 * effective_scale))
    return sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="min",
        restitution_combine_mode="min",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=0.0,
    )


def _spawn_local_ground_plane(prim_path: str, physics_material: sim_utils.RigidBodyMaterialCfg):
    ground_cfg = sim_utils.CuboidCfg(
        size=(1000.0, 1000.0, 1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        physics_material=physics_material,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.24, 0.24), roughness=1.0),
    )
    ground_cfg.func(prim_path, ground_cfg, translation=(0.0, 0.0, -0.5))


def _spawn_ground(
    prim_path: str,
    physics_material: sim_utils.RigidBodyMaterialCfg,
    mode: str = "cuboid",
):
    mode = str(mode).strip().lower()
    if mode == "plane":
        ground_cfg = sim_utils.GroundPlaneCfg(
            color=(0.02, 0.02, 0.02),
            size=(500.0, 500.0),
            physics_material=physics_material,
        )
        ground_cfg.func(prim_path, ground_cfg)
        return
    if mode != "cuboid":
        raise ValueError(f"Unsupported ground mode: {mode!r}")
    _spawn_local_ground_plane(prim_path, physics_material)


def _hide_ground_visuals(prim_path: str) -> None:
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(str(prim_path))
    if not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        try:
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
        except Exception:
            continue


@configclass
class StudentVehicleGoalEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 4
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    observation_space = 17
    state_space = 0
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
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=12.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    vehicle: ArticulationCfg = build_student_vehicle_articulation_cfg(DEFAULT_STUDENT_VEHICLE_USD)
    tunable_config_json: str = _default_tunable_config_json()

    spawn_height_m: float = 1.6
    ground_mode: str = "plane"
    start_radius_m: float = 0.75
    goal_radius_min_m: float = 4.0
    goal_radius_max_m: float = 7.0
    goal_height_m: float = 0.05
    goal_reached_threshold_m: float = 0.75
    fall_height_threshold_m: float = 0.18
    bad_tilt_gravity_threshold: float = -0.15
    max_distance_from_origin_m: float = 12.0

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
    reward_goal_bonus: float = 20.0
    reward_crash_penalty: float = -10.0


class StudentVehicleGoalEnv(DirectRLEnv):
    cfg: StudentVehicleGoalEnvCfg

    def __init__(self, cfg: StudentVehicleGoalEnvCfg, render_mode: str | None = None, **kwargs):
        self._tunable_config = normalize_tunable_config(
            load_tunable_config(cfg.tunable_config_json) if str(cfg.tunable_config_json) else StudentTunableConfig()
        )
        super().__init__(cfg, render_mode, **kwargs)

        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._semantic_actions = torch.zeros_like(self._raw_actions)
        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._current_goal_distance = torch.zeros(self.num_envs, device=self.device)

        self._steer_joint_ids, _ = self.vehicle.find_joints(
            ["front_left_steer_joint", "front_right_steer_joint"], preserve_order=True
        )
        self._drive_joint_ids, _ = self.vehicle.find_joints(
            ["front_left_wheel_joint", "front_right_wheel_joint"], preserve_order=True
        )
        self._brake_joint_ids, _ = self.vehicle.find_joints(
            [
                "front_left_wheel_joint",
                "front_right_wheel_joint",
                "rear_left_wheel_joint",
                "rear_right_wheel_joint",
            ],
            preserve_order=True,
        )
        self._wheel_joint_ids = list(self._brake_joint_ids)
        self._suspension_joint_ids, _ = self.vehicle.find_joints(
            [
                "front_left_suspension_joint",
                "front_right_suspension_joint",
                "rear_left_suspension_joint",
                "rear_right_suspension_joint",
            ],
            preserve_order=True,
        )
        self._base_body_id, _ = self.vehicle.find_bodies("base_link")
        self._base_body_ids = torch.tensor(self._base_body_id, dtype=torch.int32, device=self.device)

        self._steer_limit = float(self._tunable_config.steering_limit_rad)
        self._dry_longitudinal_scale = float(self._tunable_config.surface_longitudinal_scale.get("dry_asphalt", 1.0))
        self._dry_lateral_scale = float(self._tunable_config.surface_lateral_scale.get("dry_asphalt", 1.0))

        self._brake_sign_memory = torch.ones(self.num_envs, len(self._brake_joint_ids), device=self.device)
        self._joint_effort_targets = torch.zeros(self.num_envs, self.vehicle.num_joints, device=self.device)
        self._external_forces = torch.zeros(self.num_envs, len(self._base_body_id), 3, device=self.device)
        self._external_torques = torch.zeros_like(self._external_forces)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            for key in (
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
                "goal_bonus",
                "crash_penalty",
            )
        }

        self.vehicle.write_joint_viscous_friction_coefficient_to_sim(
            joint_viscous_friction_coeff=torch.full(
                (self.num_envs, len(self._steer_joint_ids)),
                float(self._tunable_config.steering_viscous_friction),
                device=self.device,
            ),
            joint_ids=self._steer_joint_ids,
        )
        self.vehicle.write_joint_viscous_friction_coefficient_to_sim(
            joint_viscous_friction_coeff=torch.full(
                (self.num_envs, len(self._wheel_joint_ids)),
                float(self._tunable_config.wheel_viscous_friction),
                device=self.device,
            ),
            joint_ids=self._wheel_joint_ids,
        )
        self.vehicle.write_joint_viscous_friction_coefficient_to_sim(
            joint_viscous_friction_coeff=torch.full(
                (self.num_envs, len(self._suspension_joint_ids)),
                float(self._tunable_config.suspension_viscous_friction),
                device=self.device,
            ),
            joint_ids=self._suspension_joint_ids,
        )

        self.set_debug_vis(bool(self.cfg.debug_vis))

    def _setup_scene(self):
        import omni.usd

        self.vehicle = Articulation(self.cfg.vehicle)
        self.scene.articulations["vehicle"] = self.vehicle

        stage = omni.usd.get_context().get_stage()
        _apply_runtime_student_dynamics(
            stage=stage,
            student_root_path=_source_env_vehicle_root_path(self.cfg.vehicle.prim_path),
            config=self._tunable_config,
        )

        _spawn_ground("/World/ground", _dry_ground_material_cfg(self._tunable_config), mode=self.cfg.ground_mode)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._raw_actions = actions.clone().clamp_(-1.0, 1.0)
        # Rectify throttle and brake so a zero policy output maps to a true neutral command.
        self._semantic_actions[:, 0] = torch.clamp(self._raw_actions[:, 0], min=0.0, max=1.0)
        self._semantic_actions[:, 1] = self._raw_actions[:, 1]
        self._semantic_actions[:, 2] = torch.clamp(self._raw_actions[:, 2], min=0.0, max=1.0)

    def _apply_action(self):
        joint_pos = self.vehicle.data.joint_pos
        joint_vel = self.vehicle.data.joint_vel

        self._joint_effort_targets.zero_()

        steer_target = self._semantic_actions[:, 1:2] * self._steer_limit
        steer_pos_error = steer_target - joint_pos[:, self._steer_joint_ids]
        steer_vel_error = -joint_vel[:, self._steer_joint_ids]
        steer_effort = (
            float(self._tunable_config.steering_kp_nm_per_rad) * steer_pos_error
            + float(self._tunable_config.steering_kd_nm_s_per_rad) * steer_vel_error
        )
        steer_effort.clamp_(
            -float(self._tunable_config.steering_effort_limit_nm),
            float(self._tunable_config.steering_effort_limit_nm),
        )
        self._joint_effort_targets[:, self._steer_joint_ids] = steer_effort

        drive_effort = (
            self._semantic_actions[:, 0:1]
            * float(self._tunable_config.drive_torque_nm)
            * float(self._dry_longitudinal_scale)
        )
        self._joint_effort_targets[:, self._drive_joint_ids] += drive_effort

        brake_joint_vel = joint_vel[:, self._brake_joint_ids]
        moving_mask = torch.abs(brake_joint_vel) > 1.0e-4
        current_sign = torch.sign(brake_joint_vel)
        current_sign = torch.where(current_sign == 0.0, self._brake_sign_memory, current_sign)
        self._brake_sign_memory = torch.where(moving_mask, current_sign, self._brake_sign_memory)
        brake_sign = torch.where(moving_mask, current_sign, self._brake_sign_memory)

        brake = self._semantic_actions[:, 2:3]
        front_brake_effort = (
            brake * float(self._tunable_config.brake_front_torque_nm) * float(self._dry_longitudinal_scale)
        )
        rear_brake_effort = (
            brake * float(self._tunable_config.brake_rear_torque_nm) * float(self._dry_longitudinal_scale)
        )
        self._joint_effort_targets[:, self._brake_joint_ids[0:2]] -= front_brake_effort * brake_sign[:, 0:2]
        self._joint_effort_targets[:, self._brake_joint_ids[2:4]] -= rear_brake_effort * brake_sign[:, 2:4]

        self.vehicle.set_joint_effort_target(self._joint_effort_targets)

        self._external_forces.zero_()
        self._external_torques.zero_()
        self._external_forces[:, 0, 1] = (
            -float(self._tunable_config.lateral_velocity_damping_n_per_mps)
            * float(self._dry_lateral_scale)
            * self.vehicle.data.root_lin_vel_b[:, 1]
        )
        self._external_torques[:, 0, 2] = (
            -float(self._tunable_config.yaw_stability_damping_nm_per_rad_s)
            * float(self._dry_lateral_scale)
            * self.vehicle.data.root_ang_vel_b[:, 2]
        )
        self.vehicle.permanent_wrench_composer.set_forces_and_torques(
            forces=self._external_forces,
            torques=self._external_torques,
            body_ids=self._base_body_ids,
            is_global=False,
        )

    def _compute_goal_position_body(self) -> torch.Tensor:
        goal_pos_b, _ = subtract_frame_transforms(
            self.vehicle.data.root_pos_w,
            self.vehicle.data.root_quat_w,
            self._goal_pos_w,
        )
        return goal_pos_b

    def _compute_goal_distance(self) -> tuple[torch.Tensor, torch.Tensor]:
        goal_pos_b = self._compute_goal_position_body()
        distance = torch.linalg.norm(goal_pos_b[:, :2], dim=1)
        return goal_pos_b, distance

    def _get_observations(self) -> dict:
        goal_pos_b, goal_distance = self._compute_goal_distance()
        self._current_goal_distance[:] = goal_distance

        obs = torch.cat(
            [
                goal_pos_b[:, :2] / float(max(1.0e-6, self.cfg.goal_radius_max_m)),
                goal_distance.unsqueeze(-1) / float(max(1.0e-6, self.cfg.goal_radius_max_m)),
                self.vehicle.data.root_lin_vel_b[:, :2] / 10.0,
                self.vehicle.data.root_ang_vel_b[:, 2:3] / 10.0,
                self.vehicle.data.projected_gravity_b[:, :2],
                self.vehicle.data.joint_pos[:, self._steer_joint_ids] / float(max(1.0e-6, self._steer_limit)),
                self.vehicle.data.joint_vel[:, self._wheel_joint_ids] / 50.0,
                self._raw_actions,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        goal_pos_b, goal_distance = self._compute_goal_distance()
        self._current_goal_distance[:] = goal_distance

        goal_dir_b = goal_pos_b[:, :2] / goal_distance.unsqueeze(-1).clamp_min(1.0e-6)
        progress = self._previous_goal_distance - goal_distance
        heading_alignment = goal_dir_b[:, 0]
        speed_to_goal = torch.sum(self.vehicle.data.root_lin_vel_b[:, :2] * goal_dir_b, dim=1).clamp_min(0.0)
        lateral_velocity = torch.abs(self.vehicle.data.root_lin_vel_b[:, 1])
        yaw_rate = torch.abs(self.vehicle.data.root_ang_vel_b[:, 2])
        action_rate = torch.sum(torch.square(self._raw_actions - self._previous_raw_actions), dim=1)
        action_magnitude = torch.sum(torch.square(self._semantic_actions), dim=1)
        throttle_brake_conflict = self._semantic_actions[:, 0] * self._semantic_actions[:, 2]
        goal_shaping = 1.0 - torch.tanh(goal_distance / float(max(1.0, self.cfg.goal_radius_max_m * 0.5)))
        goal_bonus = (goal_distance <= float(self.cfg.goal_reached_threshold_m)).float() * float(self.cfg.reward_goal_bonus)

        crash_mask = self._crash_mask()
        crash_penalty = crash_mask.float() * float(self.cfg.reward_crash_penalty)

        rewards = {
            "alive": torch.full_like(goal_distance, float(self.cfg.reward_scale_alive)),
            "progress": progress * float(self.cfg.reward_scale_progress),
            "goal_shaping": goal_shaping * float(self.cfg.reward_scale_goal_shaping),
            "heading": heading_alignment * float(self.cfg.reward_scale_heading),
            "speed_to_goal": speed_to_goal * float(self.cfg.reward_scale_speed_to_goal),
            "lateral_velocity": lateral_velocity * float(self.cfg.reward_scale_lateral_velocity),
            "yaw_rate": yaw_rate * float(self.cfg.reward_scale_yaw_rate),
            "action_rate": action_rate * float(self.cfg.reward_scale_action_rate),
            "action_magnitude": action_magnitude * float(self.cfg.reward_scale_action_magnitude),
            "throttle_brake_conflict": throttle_brake_conflict * float(self.cfg.reward_scale_throttle_brake_conflict),
            "goal_bonus": goal_bonus,
            "crash_penalty": crash_penalty,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        for key, value in rewards.items():
            self._episode_sums[key] += value

        self._previous_goal_distance[:] = goal_distance
        self._previous_raw_actions[:] = self._raw_actions

        return reward

    def _crash_mask(self) -> torch.Tensor:
        root_pos_rel = self.vehicle.data.root_pos_w - self.scene.env_origins
        too_low = root_pos_rel[:, 2] < float(self.cfg.fall_height_threshold_m)
        too_far = torch.linalg.norm(root_pos_rel[:, :2], dim=1) > float(self.cfg.max_distance_from_origin_m)
        bad_tilt = self.vehicle.data.projected_gravity_b[:, 2] > float(self.cfg.bad_tilt_gravity_threshold)
        return too_low | too_far | bad_tilt

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, goal_distance = self._compute_goal_distance()
        self._current_goal_distance[:] = goal_distance
        reached_goal = goal_distance <= float(self.cfg.goal_reached_threshold_m)
        crashed = self._crash_mask()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = reached_goal | crashed
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.vehicle._ALL_INDICES

        final_goal_distance = torch.mean(self._current_goal_distance[env_ids]).item() if len(env_ids) > 0 else 0.0
        goal_reached_count = torch.count_nonzero(
            self._current_goal_distance[env_ids] <= float(self.cfg.goal_reached_threshold_m)
        ).item()
        crash_count = torch.count_nonzero(self._crash_mask()[env_ids]).item() if len(env_ids) > 0 else 0

        self.extras["log"] = {}
        for key, value in self._episode_sums.items():
            self.extras["log"][f"Episode_Reward/{key}"] = torch.mean(value[env_ids]).item() / max(
                1.0, float(self.max_episode_length_s)
            )
            value[env_ids] = 0.0
        self.extras["log"]["Metrics/final_distance_to_goal"] = final_goal_distance
        self.extras["log"]["Metrics/goal_reached_count"] = float(goal_reached_count)
        self.extras["log"]["Metrics/crash_count"] = float(crash_count)

        self.vehicle.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        num_resets = len(env_ids)
        zero_actions = torch.zeros(num_resets, 3, device=self.device)
        self._raw_actions[env_ids] = zero_actions
        self._semantic_actions[env_ids] = zero_actions
        self._previous_raw_actions[env_ids] = zero_actions
        self._joint_effort_targets[env_ids] = 0.0
        self._external_forces[env_ids] = 0.0
        self._external_torques[env_ids] = 0.0
        self._brake_sign_memory[env_ids] = 1.0

        env_origins = self.scene.env_origins[env_ids]
        root_state = self.vehicle.data.default_root_state[env_ids].clone()
        root_state[:, 0:2] = env_origins[:, 0:2] + sample_uniform(
            -float(self.cfg.start_radius_m),
            float(self.cfg.start_radius_m),
            (num_resets, 2),
            self.device,
        )
        root_state[:, 2] = env_origins[:, 2] + float(self.cfg.spawn_height_m)
        yaw = sample_uniform(-math.pi, math.pi, (num_resets,), self.device)
        zeros = torch.zeros_like(yaw)
        root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        root_state[:, 7:] = 0.0

        goal_radius = sample_uniform(
            float(self.cfg.goal_radius_min_m),
            float(self.cfg.goal_radius_max_m),
            (num_resets,),
            self.device,
        )
        goal_heading = sample_uniform(-math.pi, math.pi, (num_resets,), self.device)
        env_goal_offset = torch.stack(
            [
                goal_radius * torch.cos(goal_heading),
                goal_radius * torch.sin(goal_heading),
                torch.full_like(goal_radius, float(self.cfg.goal_height_m)),
            ],
            dim=-1,
        )
        self._goal_pos_w[env_ids] = self.scene.env_origins[env_ids] + env_goal_offset
        self._previous_goal_distance[env_ids] = torch.linalg.norm(
            self._goal_pos_w[env_ids, :2] - root_state[:, :2],
            dim=1,
        )
        self._current_goal_distance[env_ids] = self._previous_goal_distance[env_ids]

        joint_pos = self.vehicle.data.default_joint_pos[env_ids].clone()
        joint_vel = self.vehicle.data.default_joint_vel[env_ids].clone()

        self.vehicle.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.vehicle.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.vehicle.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.vehicle.set_joint_effort_target(self._joint_effort_targets[env_ids], env_ids=env_ids)
        self.vehicle.permanent_wrench_composer.set_forces_and_torques(
            forces=self._external_forces[env_ids],
            torques=self._external_torques[env_ids],
            body_ids=self._base_body_ids,
            env_ids=env_ids,
            is_global=False,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_goal_marker"):
                self._goal_marker = build_goal_beacon_marker("/Visuals/GoalMarker")
            self._goal_marker.set_visibility(True)
        else:
            if hasattr(self, "_goal_marker"):
                self._goal_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "_goal_marker"):
            marker_positions, marker_indices = goal_beacon_visualization(self._goal_pos_w)
            self._goal_marker.visualize(marker_positions, marker_indices=marker_indices)

    @property
    def tunable_config(self) -> StudentTunableConfig:
        return self._tunable_config

    def tunable_config_dict(self) -> dict:
        return asdict(self._tunable_config)
