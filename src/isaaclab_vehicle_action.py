from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING, dataclass
import re
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor


def _unique_preserve_order(values: Sequence[int]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        ordered.append(int(value))
        seen.add(int(value))
    return ordered


def _resolve_parameter(
    values: float | dict[str, float] | Sequence[float],
    joint_names: Sequence[str],
    *,
    device: str,
    default_value: float,
) -> torch.Tensor:
    """Resolve scalar, regex-dict, or per-joint sequences to a tensor aligned with resolved joint names."""
    if isinstance(values, (float, int)):
        return torch.full((len(joint_names),), float(values), device=device)

    if isinstance(values, dict):
        resolved = torch.full((len(joint_names),), float(default_value), device=device)
        for pattern, value in values.items():
            regex = re.compile(pattern)
            for joint_index, joint_name in enumerate(joint_names):
                if regex.fullmatch(joint_name):
                    resolved[joint_index] = float(value)
        return resolved

    resolved_values = [float(value) for value in values]
    if len(resolved_values) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} values, but received {len(resolved_values)} for joints: {list(joint_names)}"
        )
    return torch.tensor(resolved_values, dtype=torch.float32, device=device)


class VehicleActionTerm(ActionTerm):
    r"""Semantic vehicle action term for articulated cars.

    The policy action is a 3D command with the following layout:

    1. ``throttle`` in ``[0, 1]``
    2. ``steering`` in ``[-1, 1]``
    3. ``brake`` in ``[0, 1]``

    The term decodes this semantic command into:

    - steering joint position targets
    - drive-wheel effort targets
    - brake-wheel effort targets that oppose wheel angular velocity
    """

    cfg: "VehicleActionTermCfg"
    _asset: "Articulation"

    def __init__(self, cfg: "VehicleActionTermCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._steer_joint_ids, self._steer_joint_names = self._asset.find_joints(
            cfg.steering_joint_names, preserve_order=cfg.preserve_order
        )
        self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(
            cfg.drive_joint_names, preserve_order=cfg.preserve_order
        )
        self._brake_joint_ids, self._brake_joint_names = self._asset.find_joints(
            cfg.brake_joint_names, preserve_order=cfg.preserve_order
        )

        if len(self._steer_joint_ids) == 0:
            raise ValueError("VehicleActionTerm requires at least one steering joint.")
        if len(self._drive_joint_ids) == 0:
            raise ValueError("VehicleActionTerm requires at least one drive joint.")
        if len(self._brake_joint_ids) == 0:
            raise ValueError("VehicleActionTerm requires at least one brake joint.")

        self._effort_joint_ids = _unique_preserve_order([*self._drive_joint_ids, *self._brake_joint_ids])
        effort_local_index = {joint_id: i for i, joint_id in enumerate(self._effort_joint_ids)}
        self._drive_effort_local_ids = [effort_local_index[joint_id] for joint_id in self._drive_joint_ids]
        self._brake_effort_local_ids = [effort_local_index[joint_id] for joint_id in self._brake_joint_ids]

        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._steer_position_targets = torch.zeros(self.num_envs, len(self._steer_joint_ids), device=self.device)
        self._effort_targets = torch.zeros(self.num_envs, len(self._effort_joint_ids), device=self.device)
        self._brake_direction = torch.full(
            (self.num_envs, len(self._brake_joint_ids)),
            float(cfg.brake_sign_fallback),
            dtype=torch.float32,
            device=self.device,
        )

        self._action_scale = torch.tensor(cfg.action_scale, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._action_offset = torch.tensor(cfg.action_offset, dtype=torch.float32, device=self.device).unsqueeze(0)

        self._steering_scale = _resolve_parameter(
            cfg.steering_scale, self._steer_joint_names, device=self.device, default_value=0.0
        )
        self._steering_offset = _resolve_parameter(
            cfg.steering_offset, self._steer_joint_names, device=self.device, default_value=0.0
        )
        self._drive_effort_scale = _resolve_parameter(
            cfg.drive_effort_scale, self._drive_joint_names, device=self.device, default_value=0.0
        )
        self._brake_effort_scale = _resolve_parameter(
            cfg.brake_effort_scale, self._brake_joint_names, device=self.device, default_value=0.0
        )

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "vehicle actions"
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Invalid action shape '{tuple(actions.shape)}'. Expected {(self.num_envs, self.action_dim)}."
            )

        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._action_scale + self._action_offset

        self._processed_actions[:, 0].clamp_(*self.cfg.throttle_clip)
        self._processed_actions[:, 1].clamp_(*self.cfg.steering_clip)
        self._processed_actions[:, 2].clamp_(*self.cfg.brake_clip)

    def apply_actions(self):
        throttle = self._processed_actions[:, 0:1]
        steering = self._processed_actions[:, 1:2]
        brake = self._processed_actions[:, 2:3]

        self._steer_position_targets[:] = steering * self._steering_scale.unsqueeze(0) + self._steering_offset.unsqueeze(0)
        self._asset.set_joint_position_target(self._steer_position_targets, joint_ids=self._steer_joint_ids)

        self._effort_targets.zero_()
        self._effort_targets[:, self._drive_effort_local_ids] += throttle * self._drive_effort_scale.unsqueeze(0)

        brake_joint_vel = self._asset.data.joint_vel[:, self._brake_joint_ids]
        moving_mask = torch.abs(brake_joint_vel) > float(self.cfg.brake_velocity_threshold)
        current_sign = torch.sign(brake_joint_vel)
        current_sign = torch.where(current_sign == 0.0, self._brake_direction, current_sign)
        self._brake_direction = torch.where(moving_mask, current_sign, self._brake_direction)
        brake_sign = torch.where(moving_mask, current_sign, self._brake_direction)

        self._effort_targets[:, self._brake_effort_local_ids] -= (
            brake * self._brake_effort_scale.unsqueeze(0) * brake_sign
        )
        self._asset.set_joint_effort_target(self._effort_targets, joint_ids=self._effort_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._steer_position_targets[env_ids] = 0.0
        self._effort_targets[env_ids] = 0.0
        self._brake_direction[env_ids] = float(self.cfg.brake_sign_fallback)


@dataclass(kw_only=True)
class VehicleActionTermCfg(ActionTermCfg):
    """Configuration for :class:`VehicleActionTerm`."""

    class_type: type[ActionTerm] = VehicleActionTerm

    steering_joint_names: list[str] = MISSING
    drive_joint_names: list[str] = MISSING
    brake_joint_names: list[str] = MISSING

    action_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    action_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    throttle_clip: tuple[float, float] = (0.0, 1.0)
    steering_clip: tuple[float, float] = (-1.0, 1.0)
    brake_clip: tuple[float, float] = (0.0, 1.0)

    steering_scale: float | dict[str, float] | Sequence[float] = 0.0
    steering_offset: float | dict[str, float] | Sequence[float] = 0.0
    drive_effort_scale: float | dict[str, float] | Sequence[float] = 0.0
    brake_effort_scale: float | dict[str, float] | Sequence[float] = 0.0

    brake_velocity_threshold: float = 1.0e-3
    brake_sign_fallback: float = 1.0
    preserve_order: bool = False
