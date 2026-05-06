from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


WEATHER_CONTEXT_DIM = 4


@dataclass(frozen=True)
class ObservationContract:
    ego_dim: int
    weather_dim: int
    road_point_dim: int
    road_point_k: int
    vehicle_dim: int
    vehicle_k: int

    @property
    def total_dim(self) -> int:
        return int(self.ego_dim + self.weather_dim + self.road_point_dim * self.road_point_k + self.vehicle_dim * self.vehicle_k)


def _cfg_bool(cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    return bool(cfg.get(key, default))


def _cfg_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def chocolate_observation_contract(*, road_points_enable: bool = True, road_points_k: int = 200, road_points_include_dirs: bool = False, vehicle_obs_enable: bool = True, vehicle_obs_k: int = 63, vehicle_obs_include_ttc: bool = False) -> ObservationContract:
    return ObservationContract(
        ego_dim=7,
        weather_dim=int(WEATHER_CONTEXT_DIM),
        road_point_dim=(5 if bool(road_points_include_dirs) else 3) if bool(road_points_enable) else 0,
        road_point_k=int(road_points_k) if bool(road_points_enable) else 0,
        vehicle_dim=(7 if bool(vehicle_obs_include_ttc) else 6) if bool(vehicle_obs_enable) else 0,
        vehicle_k=int(vehicle_obs_k) if bool(vehicle_obs_enable) else 0,
    )


def scene_factory_observation_contract(observation_cfg: Mapping[str, Any]) -> ObservationContract:
    return ObservationContract(
        ego_dim=7,
        weather_dim=int(WEATHER_CONTEXT_DIM) if _cfg_bool(observation_cfg, "weather_context_enable", True) else 0,
        road_point_dim=(5 if _cfg_bool(observation_cfg, "road_points_include_dirs", False) else 3)
        if _cfg_bool(observation_cfg, "road_points_enable", True)
        else 0,
        road_point_k=_cfg_int(observation_cfg, "road_points_k", 200)
        if _cfg_bool(observation_cfg, "road_points_enable", True)
        else 0,
        vehicle_dim=(7 if _cfg_bool(observation_cfg, "neighbor_include_ttc", False) else 6)
        + (1 if _cfg_bool(observation_cfg, "neighbor_include_index", False) else 0)
        if _cfg_bool(observation_cfg, "neighbor_enable", True)
        else 0,
        vehicle_k=_cfg_int(observation_cfg, "neighbor_k", 63)
        if _cfg_bool(observation_cfg, "neighbor_enable", True)
        else 0,
    )


def compare_to_chocolate_contract(observation_cfg: Mapping[str, Any]) -> dict[str, Any]:
    chocolate = chocolate_observation_contract(
        road_points_enable=_cfg_bool(observation_cfg, "road_points_enable", True),
        road_points_k=_cfg_int(observation_cfg, "road_points_k", 200),
        road_points_include_dirs=_cfg_bool(observation_cfg, "road_points_include_dirs", False),
        vehicle_obs_enable=_cfg_bool(observation_cfg, "neighbor_enable", True),
        vehicle_obs_k=_cfg_int(observation_cfg, "neighbor_k", 63),
        vehicle_obs_include_ttc=_cfg_bool(observation_cfg, "neighbor_include_ttc", False),
    )
    scene_factory = scene_factory_observation_contract(observation_cfg)
    return {
        "chocolate": chocolate,
        "scene_factory": scene_factory,
        "matches": chocolate == scene_factory,
        "differences": {
            field: {"chocolate": getattr(chocolate, field), "scene_factory": getattr(scene_factory, field)}
            for field in ObservationContract.__dataclass_fields__
            if getattr(chocolate, field) != getattr(scene_factory, field)
        },
    }