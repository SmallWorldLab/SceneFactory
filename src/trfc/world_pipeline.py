"""Helpers for wiring wet-road friction estimates into stage/world configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

from .friction_api import FrictionEstimate, FrictionInput, estimate_friction


_FRICTION_INPUT_ALIASES = {
    "v_ref_mps": "reference_speed_mps",
    "s_ref_static": "slip_static",
    "s_ref_dynamic": "slip_dynamic",
}

_FRICTION_INPUT_FIELDS = set(FrictionInput.__dataclass_fields__)


@dataclass(frozen=True)
class StageWorldSpec:
    world_index: int
    scene_json_path: Path
    scene_json_name: str
    friction_request: FrictionInput | None = None
    friction_estimate: FrictionEstimate | None = None


def _as_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _coerce_scene_json_name(scene_json: Any) -> str:
    if scene_json is None:
        raise ValueError("scene_json must be provided")
    text = str(scene_json).strip()
    if not text:
        raise ValueError("scene_json must be a non-empty string")
    if not text.endswith(".json"):
        text = f"{text}.json"
    return text


def resolve_scene_json_path(scene_dir: str | Path, scene_json: Any) -> Path:
    root = Path(scene_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"scene_json_dir does not exist: {root}")

    scene_name = _coerce_scene_json_name(scene_json)
    raw_path = Path(scene_name).expanduser()
    path = raw_path.resolve() if raw_path.is_absolute() else (root / scene_name).resolve()
    if not path.exists():
        raise FileNotFoundError(f"scene_json does not exist: {path}")
    return path


def friction_input_from_mapping(
    raw: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any] | None = None,
) -> FrictionInput:
    merged: dict[str, Any] = {}

    for source in (defaults or {}, raw):
        source_mapping = _as_mapping("friction config", source)
        for key, value in source_mapping.items():
            if value is None:
                continue
            canonical_key = _FRICTION_INPUT_ALIASES.get(str(key), str(key))
            if canonical_key not in _FRICTION_INPUT_FIELDS:
                raise ValueError(f"Unsupported friction input key: {key!r}")
            merged[canonical_key] = value

    return FrictionInput(**merged)


def _assignment_fill_mode(world_cfg: Mapping[str, Any]) -> str:
    raw = world_cfg.get("assignment_fill_mode", world_cfg.get("assignments_fill_mode", "strict"))
    return str(raw).strip().lower().replace("-", "_")


def expand_world_assignments(
    assignments: Sequence[Any],
    *,
    world_count: int,
    world_cfg: Mapping[str, Any],
) -> list[Any]:
    assignment_list = list(assignments)
    fill_mode = _assignment_fill_mode(world_cfg)

    if fill_mode == "strict":
        if len(assignment_list) != int(world_count):
            raise ValueError(
                "world.assignments must have exactly world_count entries "
                f"({len(assignment_list)} != {world_count})"
            )
        return assignment_list

    if fill_mode != "random_fill":
        raise ValueError(
            "Unsupported world.assignment_fill_mode. "
            f"Expected 'strict' or 'random_fill', got {fill_mode!r}"
        )

    if not assignment_list:
        raise ValueError("world.assignments must contain at least one entry when assignment_fill_mode=random_fill")

    rng = random.Random(
        int(world_cfg.get("assignment_fill_seed", world_cfg.get("assignments_fill_seed", 42)))
    )
    expanded: list[Any] = []
    target_count = int(world_count)

    while len(expanded) < target_count:
        cycle = list(assignment_list)
        rng.shuffle(cycle)
        remaining = target_count - len(expanded)
        expanded.extend(cycle[:remaining])

    return expanded


def prepare_stage_world_specs(cfg: Mapping[str, Any]) -> list[StageWorldSpec]:
    io_cfg = _as_mapping("io", cfg.get("io", {}))
    world_cfg = _as_mapping("world", cfg.get("world", {}))
    ground_cfg = _as_mapping("ground", cfg.get("ground", {}))

    scene_dir = Path(str(io_cfg["scene_json_dir"])).expanduser().resolve()
    world_count = int(world_cfg["world_count"])
    assignments = world_cfg.get("assignments")
    pipeline_cfg = _as_mapping(
        "ground.friction_pipeline",
        ground_cfg.get("friction_pipeline", {}) or {},
    )
    pipeline_defaults = _as_mapping(
        "ground.friction_pipeline.defaults",
        pipeline_cfg.get("defaults", {}) or {},
    )
    require_friction = bool(pipeline_cfg.get("enable", False))

    if assignments:
        if not isinstance(assignments, Sequence) or isinstance(assignments, (str, bytes)):
            raise ValueError("world.assignments must be a sequence of mappings")
        assignments = expand_world_assignments(assignments, world_count=world_count, world_cfg=world_cfg)

        specs: list[StageWorldSpec] = []
        for world_index, entry in enumerate(assignments):
            entry_mapping = _as_mapping(f"world.assignments[{world_index}]", entry)
            scene_json_path = resolve_scene_json_path(
                scene_dir,
                entry_mapping.get("scene_json"),
            )

            friction_request = None
            friction_estimate = None
            friction_cfg = entry_mapping.get("friction")
            if friction_cfg is not None:
                friction_request = friction_input_from_mapping(
                    _as_mapping(f"world.assignments[{world_index}].friction", friction_cfg),
                    defaults=pipeline_defaults,
                )
                friction_estimate = estimate_friction(friction_request)
            elif require_friction:
                raise ValueError(
                    "ground.friction_pipeline.enable=true requires a friction block "
                    f"for world.assignments[{world_index}]"
                )

            specs.append(
                StageWorldSpec(
                    world_index=world_index,
                    scene_json_path=scene_json_path,
                    scene_json_name=scene_json_path.name,
                    friction_request=friction_request,
                    friction_estimate=friction_estimate,
                )
            )
        return specs

    explicit_scene_jsons = io_cfg.get("scene_jsons")
    if explicit_scene_jsons:
        if not isinstance(explicit_scene_jsons, Sequence) or isinstance(
            explicit_scene_jsons, (str, bytes)
        ):
            raise ValueError("io.scene_jsons must be a sequence of scene JSON filenames")
        if len(explicit_scene_jsons) < world_count:
            raise ValueError(
                "io.scene_jsons must contain at least world_count entries "
                f"({len(explicit_scene_jsons)} < {world_count})"
            )
        specs: list[StageWorldSpec] = []
        for world_index in range(world_count):
            scene_json_path = resolve_scene_json_path(scene_dir, explicit_scene_jsons[world_index])
            specs.append(
                StageWorldSpec(
                    world_index=world_index,
                    scene_json_path=scene_json_path,
                    scene_json_name=scene_json_path.name,
                )
            )
        return specs

    all_jsons = sorted(scene_dir.glob("scene_*.json"))
    k = int(io_cfg.get("take_first_k_scenes", world_count))
    json_paths = all_jsons[:k]
    if len(json_paths) < k:
        raise RuntimeError(
            f"Found only {len(json_paths)} scene_*.json files in {scene_dir}, wanted {k}"
        )
    if len(json_paths) < world_count:
        raise RuntimeError(
            f"Need at least {world_count} scene_*.json files in {scene_dir}, found {len(json_paths)}"
        )
    return [
        StageWorldSpec(
            world_index=world_index,
            scene_json_path=json_paths[world_index],
            scene_json_name=json_paths[world_index].name,
        )
        for world_index in range(world_count)
    ]
