from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.isaaclab_bootstrap import ensure_isaaclab_source_paths
from src.trfc.lane_center_sampler import compute_scene_center_from_road

ensure_isaaclab_source_paths()

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp_cache")

from isaaclab.app import AppLauncher

from src.scene_factory_multiworld_scene import (
    _build_single_world_roads_only,
    _load_yaml,
    _spawn_goal_marker,
    extract_vehicle_spawns_from_json,
)


DEFAULT_TEMPLATE_PATH = "configs/scene_factory/multiworld_scene_curated_lane_safety_worldreset_32.yaml"
DEFAULT_TRAINING_PRESET_TEMPLATE = (
    "configs/scene_factory/"
    "goal_reaching_roads_choco_obs_curated32_agent_slots_goal3_late_fusion_oldobs_oldppo_"
    "routeprogress_candidate_working_v1.yaml"
)
DEFAULT_EVAL_PRESET_TEMPLATE = (
    "configs/scene_factory/goal_reaching_roads_policy_eval_unseen32_candidate_working_v1.yaml"
)
DEFAULT_OUTPUT_DIR = "configs/scene_factory/generated"
PREVIEW_ROOT = "/World/SceneFactoryConfigWizard"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively build SceneFactory scene configs by prompting for base parameters and then "
            "visually curating training/testing scene assignments."
        )
    )
    parser.add_argument(
        "--template",
        type=str,
        default=DEFAULT_TEMPLATE_PATH,
        help="Template SceneFactory scene config to clone and edit.",
    )
    parser.add_argument(
        "--training_preset_template",
        type=str,
        default=DEFAULT_TRAINING_PRESET_TEMPLATE,
        help="Training preset YAML to clone when writing a runnable training preset.",
    )
    parser.add_argument(
        "--eval_preset_template",
        type=str,
        default=DEFAULT_EVAL_PRESET_TEMPLATE,
        help="Eval preset YAML to clone when writing a runnable eval preset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated SceneFactory YAMLs will be written.",
    )
    parser.add_argument(
        "--scene_json_dir",
        type=str,
        default="",
        help="Optional override for io.scene_json_dir before prompting.",
    )
    parser.add_argument(
        "--preview_marker_limit",
        type=int,
        default=16,
        help="Maximum number of valid controllable start/goal pairs to draw in preview.",
    )
    parser.add_argument(
        "--viz_mode",
        type=str,
        default="matplotlib",
        choices=("matplotlib", "isaacsim"),
        help="Scene review UI. 'matplotlib' matches the old lightweight curator style.",
    )
    parser.add_argument(
        "--lane_types",
        type=str,
        default="1,2",
        help="Comma-separated road types to draw as lane center in the matplotlib visualizer.",
    )
    parser.add_argument(
        "--edge_types",
        type=str,
        default="15,16",
        help="Comma-separated road types to draw as forbidden edge in the matplotlib visualizer.",
    )
    parser.add_argument(
        "--figsize",
        type=str,
        default="11,11",
        help="Matplotlib scene-review figure size as 'W,H'.",
    )
    parser.add_argument(
        "--show_od_lines",
        action="store_true",
        default=True,
        help="Show start-to-goal connectors for valid controllable spawns in the matplotlib visualizer.",
    )
    parser.add_argument(
        "--hide_od_lines",
        dest="show_od_lines",
        action="store_false",
        help="Hide start-to-goal connectors in the matplotlib visualizer.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _prompt_text(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw or str(default)


def _prompt_int(label: str, default: int, *, minimum: int | None = None) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            value = int(default)
        else:
            try:
                value = int(raw)
            except ValueError:
                print("  Enter an integer.")
                continue
        if minimum is not None and value < minimum:
            print(f"  Must be >= {minimum}.")
            continue
        return value


def _prompt_float(label: str, default: float) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("  Enter a number.")


def _prompt_bool(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{suffix}]: ").strip().lower()
        if not raw:
            return bool(default)
        if raw in {"y", "yes", "1", "true", "t"}:
            return True
        if raw in {"n", "no", "0", "false", "f"}:
            return False
        print("  Enter y or n.")


# ---------------------------------------------------------------------------
# Friction randomization
# ---------------------------------------------------------------------------
# Each entry: (weight, road_type, precip_type, precip_intensity_mmph, water_film_mm)
# Weights are relative; they will be normalised to sum to 1.0.
# Roughly: 50% dry AC, 20% light-rain AC, 15% moderate-rain AC,
#           10% wet SMA, 5% heavy-rain OGFC.
_FRICTION_DISTRIBUTION: list[tuple[float, str, str, float, float]] = [
    (0.50, "AC",   "clear", 0.0,  0.00),
    (0.20, "AC",   "rain",  2.0,  0.30),
    (0.15, "AC",   "rain",  5.0,  1.00),
    (0.10, "SMA",  "rain",  5.0,  2.00),
    (0.05, "OGFC", "rain", 10.0,  4.00),
]


def _sample_friction_cfg(rng: np.random.Generator) -> dict[str, Any]:
    """Draw one friction config from _FRICTION_DISTRIBUTION."""
    weights = np.array([w for w, *_ in _FRICTION_DISTRIBUTION], dtype=float)
    weights /= weights.sum()
    idx = int(rng.choice(len(_FRICTION_DISTRIBUTION), p=weights))
    _, road_type, precip_type, precip_intensity_mmph, water_film_mm = _FRICTION_DISTRIBUTION[idx]
    return {
        "road_type": road_type,
        "precip_type": precip_type,
        "precip_intensity_mmph": float(precip_intensity_mmph),
        "water_film_mm": float(water_film_mm),
    }



def _infer_grid_cols(world_count: int) -> int:
    return max(1, int(math.ceil(math.sqrt(float(max(world_count, 1))))))


def _infer_rows(world_count: int, grid_cols: int) -> int:
    return max(1, int(math.ceil(float(max(world_count, 1)) / float(max(grid_cols, 1)))))


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _parse_float_pair(text: str) -> tuple[float, float]:
    parts = [x.strip() for x in str(text).split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated floats, got {text!r}")
    return float(parts[0]), float(parts[1])


def _discover_scene_paths(scene_json_dir: Path) -> list[Path]:
    paths = sorted(p for p in scene_json_dir.glob("*.json") if p.is_file())
    dedup: dict[str, Path] = {}
    for path in paths:
        dedup.setdefault(path.name, path)
    return list(dedup.values())


def _load_scene_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Scene JSON root must be a mapping, got {type(payload).__name__}: {path}")
    return payload


def _safe_xy(block: Any) -> np.ndarray | None:
    if not isinstance(block, Mapping):
        return None
    try:
        x = float(block.get("x", None))
        y = float(block.get("y", None))
    except Exception:
        return None
    return np.asarray([x, y], dtype=np.float32)


def _iter_polylines(scene_cfg: Mapping[str, Any]):
    road = scene_cfg.get("road", {}) or {}
    polylines = list((road.get("polylines", []) or []))
    for polyline in polylines:
        try:
            road_type = int(polyline.get("type", -1))
        except Exception:
            road_type = -1
        pts = np.asarray(polyline.get("xyz", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
            continue
        yield road_type, pts[:, :2]


def _compute_scene_center_xy(scene_cfg: Mapping[str, Any], *, center_mode: str = "mean") -> np.ndarray:
    center_mode = str(center_mode).strip().lower()
    if center_mode != "bbox":
        center = compute_scene_center_from_road(dict(scene_cfg))
        return np.asarray(center[:2], dtype=np.float32)

    road = scene_cfg.get("road", {}) or {}
    polylines = road.get("polylines", []) or []
    all_points: list[np.ndarray] = []
    for polyline in polylines:
        pts = np.asarray(polyline.get("xyz", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
            continue
        all_points.append(pts[:, :2])
    if not all_points:
        return np.zeros((2,), dtype=np.float32)
    points = np.concatenate(all_points, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return 0.5 * (mins + maxs)


def _count_raw_agents(scene_cfg: Mapping[str, Any]) -> int:
    agents = scene_cfg.get("agents", {}) or {}
    return len(list((agents.get("items", []) or [])))


def _plot_scene_matplotlib(
    *,
    scene_cfg: Mapping[str, Any],
    scene_name: str,
    phase_label: str,
    progress_label: str,
    valid_spawns: list,
    origin_mode: str,
    origin_center_mode: str,
    lane_types: list[int],
    edge_types: list[int],
    show_od_lines: bool,
    figsize: tuple[float, float],
) -> Any:
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax = plt.subplots(figsize=figsize)
    lane_set = {int(x) for x in lane_types}
    edge_set = {int(x) for x in edge_types}
    scene_center_xy = (
        _compute_scene_center_xy(scene_cfg, center_mode=origin_center_mode)
        if str(origin_mode).strip().lower() == "center"
        else np.zeros((2,), dtype=np.float32)
    )

    n_lane = 0
    n_edge = 0
    n_other = 0
    for road_type, pts in _iter_polylines(scene_cfg):
        pts = np.asarray(pts, dtype=np.float32) - scene_center_xy[None, :]
        if road_type in lane_set:
            color = "#2E86DE"
            lw = 0.9
            alpha = 0.80
            n_lane += 1
        elif road_type in edge_set:
            color = "#E74C3C"
            lw = 1.0
            alpha = 0.90
            n_edge += 1
        else:
            color = "#B0B0B0"
            lw = 0.6
            alpha = 0.36
            n_other += 1
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, alpha=alpha, zorder=1)

    if valid_spawns:
        starts = np.asarray([[float(s.start_local_xyz[0]), float(s.start_local_xyz[1])] for s in valid_spawns], dtype=np.float32)
        goals = np.asarray([[float(s.goal_local_xyz[0]), float(s.goal_local_xyz[1])] for s in valid_spawns], dtype=np.float32)
        if show_od_lines:
            for start_xy, goal_xy in zip(starts, goals):
                ax.plot(
                    [float(start_xy[0]), float(goal_xy[0])],
                    [float(start_xy[1]), float(goal_xy[1])],
                    color="#C218D4",
                    linewidth=2.0,
                    alpha=0.58,
                    solid_capstyle="round",
                    zorder=4,
                )
            vec = goals - starts
            ax.quiver(
                starts[:, 0],
                starts[:, 1],
                vec[:, 0],
                vec[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.0020,
                color="#C218D4",
                alpha=0.32,
                zorder=5,
            )
        ax.scatter(
            starts[:, 0],
            starts[:, 1],
            s=52.0,
            c="#2ECC71",
            alpha=0.95,
            edgecolors="#0b0b0b",
            linewidths=0.4,
            zorder=6,
        )
        ax.scatter(
            goals[:, 0],
            goals[:, 1],
            s=96.0,
            c="#FFD34E",
            marker="*",
            alpha=0.95,
            edgecolors="#0b0b0b",
            linewidths=0.45,
            zorder=7,
        )

    count_valid = (scene_cfg.get("agents", {}) or {}).get("count_valid", None)
    count_valid_text = f"{count_valid}" if count_valid is not None else "n/a"
    ax.set_title(
        f"{phase_label} | {progress_label}\n"
        f"{scene_name} | raw_agents={_count_raw_agents(scene_cfg)} count_valid={count_valid_text} "
        f"scene_factory_valid={len(valid_spawns)}\n"
        "Keys: y=accept  n=reject  b=back  q=finish"
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    return fig


def _wait_for_matplotlib_decision(fig: Any) -> str:
    import matplotlib.pyplot as plt  # type: ignore

    state: dict[str, str | None] = {"decision": None}

    def _on_key(event: Any) -> None:
        key = str(getattr(event, "key", "") or "").lower()
        if key == "y":
            state["decision"] = "y"
            return
        if key == "n":
            state["decision"] = "n"
            return
        if key == "b":
            state["decision"] = "b"
            return
        if key in ("q", "escape"):
            state["decision"] = "q"

    def _on_close(_event: Any) -> None:
        if state["decision"] is None:
            state["decision"] = "q"

    key_cid = fig.canvas.mpl_connect("key_press_event", _on_key)
    close_cid = fig.canvas.mpl_connect("close_event", _on_close)
    try:
        while state["decision"] is None:
            plt.pause(0.05)
    finally:
        try:
            fig.canvas.mpl_disconnect(key_cid)
            fig.canvas.mpl_disconnect(close_cid)
        except Exception:
            pass
    return str(state["decision"] or "q")


def _spawn_start_marker(prim_path: str, xyz: tuple[float, float, float]) -> None:
    import isaaclab.sim as sim_utils

    marker_cfg = sim_utils.SphereCfg(
        radius=0.65,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        rigid_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.10, 0.42, 1.0),
            emissive_color=(0.02, 0.10, 0.28),
            roughness=0.12,
            metallic=0.0,
        ),
    )
    marker_cfg.func(
        prim_path,
        marker_cfg,
        translation=(float(xyz[0]), float(xyz[1]), float(xyz[2]) + 0.35),
    )


def _preview_camera(sim: Any, bounds_size_m: float) -> None:
    extent = max(120.0, float(bounds_size_m) * 1.25)
    sim.set_camera_view(eye=(0.0, 0.0, extent), target=(0.0, 0.0, 0.0))


def _preview_scene(
    *,
    stage: Any,
    sim: Any,
    preview_cfg: Mapping[str, Any],
    scene_path: Path,
    preview_marker_limit: int,
) -> int:
    from pxr import UsdGeom

    stage.RemovePrim(PREVIEW_ROOT)
    UsdGeom.Xform.Define(stage, PREVIEW_ROOT)
    world_root = f"{PREVIEW_ROOT}/world_000"
    _build_single_world_roads_only(stage=stage, cfg=preview_cfg, json_path=scene_path, world_root=world_root)

    world_cfg = dict(preview_cfg.get("world", {}) or {})
    vehicles_cfg = dict(preview_cfg.get("vehicles", {}) or {})
    spawns = extract_vehicle_spawns_from_json(
        scene_path,
        bounds_size_m=float(world_cfg.get("bounds_size_m", 200.0)),
        origin_mode=str(world_cfg.get("origin_mode", "center")),
        origin_center_mode=str(world_cfg.get("origin_center_mode", "mean")),
        max_controllable=max(1, int(preview_marker_limit)),
        require_goal_in_bounds=bool(vehicles_cfg.get("require_goal_in_bounds", True)),
        skip_if_start_in_goal=bool(vehicles_cfg.get("skip_if_start_in_goal", True)),
        goal_radius_m=float(vehicles_cfg.get("goal_radius_m", 3.0)),
        start_goal_thresh_m=vehicles_cfg.get("start_goal_thresh_m"),
    )

    UsdGeom.Xform.Define(stage, f"{world_root}/PreviewStarts")
    UsdGeom.Xform.Define(stage, f"{world_root}/PreviewGoals")
    goal_radius_m = float((preview_cfg.get("goal_markers", {}) or {}).get("radius_m", 1.0))
    goal_height_m = float((preview_cfg.get("goal_markers", {}) or {}).get("height_m", 0.12))

    for spawn_idx, spawn in enumerate(spawns):
        _spawn_start_marker(
            f"{world_root}/PreviewStarts/Start_{spawn_idx:03d}",
            spawn.start_local_xyz,
        )
        _spawn_goal_marker(
            f"{world_root}/PreviewGoals/Goal_{spawn_idx:03d}",
            goal_local_xyz=spawn.goal_local_xyz,
            radius_m=goal_radius_m,
            height_m=goal_height_m,
        )

    _preview_camera(sim, float(world_cfg.get("bounds_size_m", 200.0)))
    sim.reset()
    for _ in range(18):
        sim.step(render=True)
    return len(spawns)


def _wait_for_preview_decision(simulation_app: Any, sim: Any, prompt: str) -> str:
    response_queue: queue.Queue[str] = queue.Queue(maxsize=1)

    def _reader() -> None:
        try:
            response_queue.put(input(prompt).strip().lower())
        except EOFError:
            response_queue.put("q")

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    while simulation_app.is_running():
        try:
            return response_queue.get_nowait()
        except queue.Empty:
            sim.step(render=True)
    return "q"


def _pick_scenes_isaacsim(
    *,
    label: str,
    simulation_app: Any,
    sim: Any,
    stage: Any,
    preview_cfg: Mapping[str, Any],
    candidate_paths: list[Path],
    target_count: int,
    preview_marker_limit: int,
) -> tuple[list[Path], set[str]]:
    selected: list[Path] = []
    rejected: set[str] = set()
    history: list[tuple[str, Path]] = []
    idx = 0

    while idx < len(candidate_paths) and len(selected) < target_count and simulation_app.is_running():
        scene_path = candidate_paths[idx]
        spawn_count = _preview_scene(
            stage=stage,
            sim=sim,
            preview_cfg=preview_cfg,
            scene_path=scene_path,
            preview_marker_limit=preview_marker_limit,
        )
        print(
            f"\n[{label}] {len(selected)}/{target_count} selected | "
            f"scene {idx + 1}/{len(candidate_paths)} | {scene_path.name} | valid_spawns={spawn_count}"
        )
        print("Commands: [y] accept  [n] reject  [b] back  [q] finish now")
        command = _wait_for_preview_decision(simulation_app, sim, "> ")

        if command == "y":
            selected.append(scene_path)
            history.append(("selected", scene_path))
            idx += 1
        elif command == "n":
            rejected.add(scene_path.name)
            history.append(("rejected", scene_path))
            idx += 1
        elif command == "b":
            if not history:
                print("Nothing to undo.")
                continue
            last_action, last_path = history.pop()
            idx = max(0, idx - 1)
            if last_action == "selected":
                if selected and selected[-1].name == last_path.name:
                    selected.pop()
                else:
                    selected = [item for item in selected if item.name != last_path.name]
            else:
                rejected.discard(last_path.name)
        elif command == "q":
            break
        else:
            print("Unknown command.")

    return selected, rejected


def _pick_scenes_matplotlib(
    *,
    label: str,
    preview_cfg: Mapping[str, Any],
    candidate_paths: list[Path],
    target_count: int,
    preview_marker_limit: int,
    lane_types: list[int],
    edge_types: list[int],
    figsize: tuple[float, float],
    show_od_lines: bool,
) -> tuple[list[Path], set[str]]:
    import matplotlib.pyplot as plt  # type: ignore

    selected: list[Path] = []
    rejected: set[str] = set()
    history: list[tuple[str, Path]] = []
    idx = 0
    world_cfg = dict(preview_cfg.get("world", {}) or {})
    vehicles_cfg = dict(preview_cfg.get("vehicles", {}) or {})

    while idx < len(candidate_paths) and len(selected) < target_count:
        scene_path = candidate_paths[idx]
        scene_cfg = _load_scene_json(scene_path)
        valid_spawns = extract_vehicle_spawns_from_json(
            scene_path,
            bounds_size_m=float(world_cfg.get("bounds_size_m", 200.0)),
            origin_mode=str(world_cfg.get("origin_mode", "center")),
            origin_center_mode=str(world_cfg.get("origin_center_mode", "mean")),
            max_controllable=max(1, int(preview_marker_limit)),
            require_goal_in_bounds=bool(vehicles_cfg.get("require_goal_in_bounds", True)),
            skip_if_start_in_goal=bool(vehicles_cfg.get("skip_if_start_in_goal", True)),
            goal_radius_m=float(vehicles_cfg.get("goal_radius_m", 3.0)),
            start_goal_thresh_m=vehicles_cfg.get("start_goal_thresh_m"),
        )
        fig = _plot_scene_matplotlib(
            scene_cfg=scene_cfg,
            scene_name=scene_path.name,
            phase_label=f"{label.title()} Scene Review",
            progress_label=f"{len(selected)}/{target_count} selected | scene {idx + 1}/{len(candidate_paths)}",
            valid_spawns=valid_spawns,
            origin_mode=str(world_cfg.get("origin_mode", "center")),
            origin_center_mode=str(world_cfg.get("origin_center_mode", "mean")),
            lane_types=lane_types,
            edge_types=edge_types,
            show_od_lines=show_od_lines,
            figsize=figsize,
        )
        plt.show(block=False)
        command = _wait_for_matplotlib_decision(fig)
        plt.close(fig)

        if command == "y":
            selected.append(scene_path)
            history.append(("selected", scene_path))
            idx += 1
        elif command == "n":
            rejected.add(scene_path.name)
            history.append(("rejected", scene_path))
            idx += 1
        elif command == "b":
            if not history:
                print("Nothing to undo.")
                continue
            last_action, last_path = history.pop()
            idx = max(0, idx - 1)
            if last_action == "selected":
                if selected and selected[-1].name == last_path.name:
                    selected.pop()
                else:
                    selected = [item for item in selected if item.name != last_path.name]
            else:
                rejected.discard(last_path.name)
        elif command == "q":
            break
        else:
            print("Unknown command.")

    return selected, rejected


def _make_assignment(scene_name: str, friction_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {"scene_json": scene_name, "friction": dict(friction_cfg)}


def _compute_viewer(world_cfg: Mapping[str, Any]) -> dict[str, Any]:
    grid_cols = int(world_cfg.get("grid_cols", 1))
    rows = int(world_cfg.get("rows", 1))
    world_size = tuple(map(float, world_cfg.get("world_size_m", [200.0, 200.0])))
    padding_m = float(world_cfg.get("padding_m", 200.0))
    span_x = max(world_size[0], grid_cols * world_size[0] + max(0, grid_cols - 1) * padding_m)
    span_y = max(world_size[1], rows * world_size[1] + max(0, rows - 1) * padding_m)
    center_x = 0.5 * max(0.0, (grid_cols - 1) * (world_size[0] + padding_m))
    center_y = 0.5 * max(0.0, (rows - 1) * (world_size[1] + padding_m))
    height = 0.95 * max(span_x, span_y) + 60.0
    return {
        "eye": [center_x, center_y, height],
        "lookat": [center_x, center_y, 0.0],
        "light_path": "/World/Light",
        "light_intensity": 3200.0,
        "light_color": [0.78, 0.78, 0.78],
    }


def _apply_common_values(
    cfg: dict[str, Any],
    *,
    scene_json_dir: Path,
    world_count: int,
    grid_cols: int,
    bounds_size_m: float,
    padding_m: float,
    base_z_m: float,
    max_controllable_per_world: int,
    spawn_height_m: float,
    require_goal_in_bounds: bool,
    skip_if_start_in_goal: bool,
    goal_radius_m: float,
    start_goal_thresh_m: float,
    road_render_mode: str,
    jump_break_m: float,
    seg_width: float,
    seg_height: float,
    z_lift: float,
    flatten_road_z: bool,
    road_z_m: float,
    road_polyline_reduction_area: float,
    min_points_for_reduction: int,
    friction_cfg: Mapping[str, Any],
    assignments: list[dict[str, Any]],
    randomize_friction: bool = False,
) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    rows = _infer_rows(world_count, grid_cols)

    out.setdefault("io", {})["scene_json_dir"] = str(scene_json_dir)

    world_cfg = out.setdefault("world", {})
    world_cfg["world_count"] = int(world_count)
    world_cfg["grid_cols"] = int(grid_cols)
    world_cfg["rows"] = int(rows)
    world_cfg["world_size_m"] = [float(bounds_size_m), float(bounds_size_m)]
    world_cfg["padding_m"] = float(padding_m)
    world_cfg["base_z_m"] = float(base_z_m)
    world_cfg["bounds_size_m"] = float(bounds_size_m)
    world_cfg["origin_mode"] = str(world_cfg.get("origin_mode", "center"))
    world_cfg["origin_center_mode"] = str(world_cfg.get("origin_center_mode", "mean"))
    world_cfg["assignments"] = list(assignments)

    road_cfg = out.setdefault("road", {})
    road_cfg["render_mode"] = str(road_render_mode)
    road_cfg["jump_break_m"] = float(jump_break_m)
    road_cfg["seg_width"] = float(seg_width)
    road_cfg["seg_height"] = float(seg_height)
    road_cfg["z_lift"] = float(z_lift)
    road_cfg["flatten_road_z"] = bool(flatten_road_z)
    road_cfg["road_z_m"] = float(road_z_m)
    road_cfg["polyline_reduction_area"] = float(road_polyline_reduction_area)
    road_cfg["min_points_for_reduction"] = int(min_points_for_reduction)
    road_cfg["enable_segment_collision"] = False
    road_cfg["trigger_enable"] = False

    vehicles_cfg = out.setdefault("vehicles", {})
    vehicles_cfg["max_controllable_per_world"] = int(max_controllable_per_world)
    vehicles_cfg["spawn_height_m"] = float(spawn_height_m)
    vehicles_cfg["require_goal_in_bounds"] = bool(require_goal_in_bounds)
    vehicles_cfg["skip_if_start_in_goal"] = bool(skip_if_start_in_goal)
    vehicles_cfg["goal_radius_m"] = float(goal_radius_m)
    vehicles_cfg["start_goal_thresh_m"] = float(start_goal_thresh_m)

    out["viewer"] = _compute_viewer(world_cfg)
    out.setdefault("goal_markers", {})["enable"] = True
    out["goal_markers"]["radius_m"] = float(out["goal_markers"].get("radius_m", 1.0))
    out["goal_markers"]["height_m"] = float(out["goal_markers"].get("height_m", 0.12))

    ground_cfg = out.setdefault("ground", {})
    friction_values = ground_cfg.get("friction_values", [0.50])
    if not isinstance(friction_values, list) or not friction_values:
        ground_cfg["friction_values"] = [0.50]

    # Stamp friction into assignments.
    # When randomize_friction=True the assignments already carry per-world friction
    # drawn from _FRICTION_DISTRIBUTION; do not overwrite them.
    if not randomize_friction:
        for entry in out["world"]["assignments"]:
            entry["friction"] = dict(friction_cfg)

    return out


def _prompt_settings(template_cfg: Mapping[str, Any], scene_json_dir_override: str) -> dict[str, Any]:
    io_cfg = dict(template_cfg.get("io", {}) or {})
    world_cfg = dict(template_cfg.get("world", {}) or {})
    road_cfg = dict(template_cfg.get("road", {}) or {})
    vehicles_cfg = dict(template_cfg.get("vehicles", {}) or {})

    default_scene_dir = scene_json_dir_override or str(io_cfg.get("scene_json_dir", "data/processed/waymo_scenes_json"))
    base_name = _prompt_text("Config base name", "scene_factory_custom")
    scene_json_dir = Path(_prompt_text("Scene JSON directory", default_scene_dir)).expanduser().resolve()
    if not scene_json_dir.is_dir():
        raise FileNotFoundError(f"scene_json_dir does not exist: {scene_json_dir}")

    world_count = _prompt_int("Training world count", int(world_cfg.get("world_count", 32)), minimum=1)
    grid_cols = _prompt_int("Grid columns", _infer_grid_cols(world_count), minimum=1)
    bounds_size_m = _prompt_float("Bounds size (m)", float(world_cfg.get("bounds_size_m", 200.0)))
    padding_m = _prompt_float("Padding between worlds (m)", float(world_cfg.get("padding_m", 200.0)))
    base_z_m = _prompt_float("Base Z (m)", float(world_cfg.get("base_z_m", 0.0)))
    max_controllable_per_world = _prompt_int(
        "Max controllable vehicles per world",
        int(vehicles_cfg.get("max_controllable_per_world", 8)),
        minimum=1,
    )
    spawn_height_m = _prompt_float("Vehicle spawn height (m)", float(vehicles_cfg.get("spawn_height_m", 1.0)))
    goal_radius_m = _prompt_float("Goal radius (m)", float(vehicles_cfg.get("goal_radius_m", 3.0)))
    start_goal_thresh_m = _prompt_float(
        "Minimum start-goal distance (m)",
        float(vehicles_cfg.get("start_goal_thresh_m", goal_radius_m)),
    )
    jump_break_m = _prompt_float("Road jump break (m)", float(road_cfg.get("jump_break_m", 3.0)))
    seg_width = _prompt_float("Road segment width (m)", float(road_cfg.get("seg_width", 0.10)))
    seg_height = _prompt_float("Road segment height (m)", float(road_cfg.get("seg_height", 0.01)))
    z_lift = _prompt_float("Road Z lift (m)", float(road_cfg.get("z_lift", 0.02)))
    road_z_m = _prompt_float("Flattened road Z (m)", float(road_cfg.get("road_z_m", 0.0)))
    polyline_reduction_area = _prompt_float(
        "Polyline reduction area",
        float(road_cfg.get("polyline_reduction_area", 0.0)),
    )
    min_points_for_reduction = _prompt_int(
        "Minimum points for polyline reduction",
        int(road_cfg.get("min_points_for_reduction", 10)),
        minimum=2,
    )

    require_goal_in_bounds = _prompt_bool(
        "Require goals to be inside bounds",
        bool(vehicles_cfg.get("require_goal_in_bounds", True)),
    )
    skip_if_start_in_goal = _prompt_bool(
        "Skip agents already in goal",
        bool(vehicles_cfg.get("skip_if_start_in_goal", True)),
    )
    flatten_road_z = _prompt_bool("Flatten road Z", bool(road_cfg.get("flatten_road_z", True)))
    road_render_mode = _prompt_text("Road render mode", str(road_cfg.get("render_mode", "explicit_prims")))

    print("\nDefault friction to stamp into selected assignments:")
    friction_cfg = {
        "road_type": _prompt_text("  road_type", "AC"),
        "precip_type": _prompt_text("  precip_type", "clear"),
        "precip_intensity_mmph": _prompt_float("  precip_intensity_mmph", 0.0),
        "water_film_mm": _prompt_float("  water_film_mm", 0.0),
    }
    randomize_friction = _prompt_bool(
        "Randomize friction per world (draws from built-in AC/SMA/OGFC distribution)",
        False,
    )
    friction_random_seed = None
    if randomize_friction:
        friction_random_seed = _prompt_int("  Friction randomization seed", 42, minimum=0)

    return {
        "base_name": base_name,
        "scene_json_dir": scene_json_dir,
        "world_count": world_count,
        "grid_cols": grid_cols,
        "bounds_size_m": bounds_size_m,
        "padding_m": padding_m,
        "base_z_m": base_z_m,
        "max_controllable_per_world": max_controllable_per_world,
        "spawn_height_m": spawn_height_m,
        "goal_radius_m": goal_radius_m,
        "start_goal_thresh_m": start_goal_thresh_m,
        "jump_break_m": jump_break_m,
        "seg_width": seg_width,
        "seg_height": seg_height,
        "z_lift": z_lift,
        "road_z_m": road_z_m,
        "road_polyline_reduction_area": polyline_reduction_area,
        "min_points_for_reduction": min_points_for_reduction,
        "require_goal_in_bounds": require_goal_in_bounds,
        "skip_if_start_in_goal": skip_if_start_in_goal,
        "flatten_road_z": flatten_road_z,
        "road_render_mode": road_render_mode,
        "friction_cfg": friction_cfg,
        "randomize_friction": randomize_friction,
        "friction_random_seed": friction_random_seed,
    }


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_summary(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _repo_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path.resolve())


def _build_preset_from_template(
    *,
    template_path: Path,
    generated_scene_cfg_path: Path,
    world_count: int,
    max_controllable_per_world: int,
    generated_run_name: str,
) -> dict[str, Any]:
    preset_cfg = _load_yaml(template_path)
    env_cfg = preset_cfg.setdefault("env", {})
    env_cfg["num_envs"] = int(world_count)
    if "num_agents_per_env" in env_cfg:
        env_cfg["num_agents_per_env"] = min(int(env_cfg["num_agents_per_env"]), int(max_controllable_per_world))

    scene_factory_cfg = preset_cfg.setdefault("scene_factory", {})
    scene_factory_cfg["config_path"] = _repo_relative_or_absolute(generated_scene_cfg_path)
    if int(world_count) > 1:
        scene_factory_cfg["world_selection_mode"] = "random_envs"

    runner_cfg = preset_cfg.setdefault("runner", {})
    if runner_cfg:
        runner_cfg["run_name"] = str(generated_run_name)
    return preset_cfg


def main() -> None:
    args_cli = _build_parser().parse_args()
    template_cfg = _load_yaml(args_cli.template)

    print("SceneFactory Config Wizard")
    print("Press Enter to accept defaults.\n")
    settings = _prompt_settings(template_cfg, args_cli.scene_json_dir)

    all_scene_paths = _discover_scene_paths(settings["scene_json_dir"])
    if not all_scene_paths:
        raise RuntimeError(f"No .json scenes found in {settings['scene_json_dir']}")

    print(f"\nDiscovered {len(all_scene_paths)} unique scene files.")
    print(f"Launching {args_cli.viz_mode} scene review for training-scene selection...\n")

    lane_types = _parse_int_list(args_cli.lane_types)
    edge_types = _parse_int_list(args_cli.edge_types)
    figsize = _parse_float_pair(args_cli.figsize)

    simulation_app = None
    sim = None
    stage = None
    if str(args_cli.viz_mode).strip().lower() == "isaacsim":
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        from isaaclab.sim import SimulationCfg, SimulationContext

        sim_device = str(args_cli.device or "cuda:0").strip() or "cuda:0"
        sim = SimulationContext(SimulationCfg(dt=1.0 / 120.0, device=sim_device))
        import omni.usd

        stage = omni.usd.get_context().get_stage()

    preview_cfg = _apply_common_values(
        template_cfg,
        scene_json_dir=settings["scene_json_dir"],
        world_count=1,
        grid_cols=1,
        bounds_size_m=settings["bounds_size_m"],
        padding_m=settings["padding_m"],
        base_z_m=settings["base_z_m"],
        max_controllable_per_world=settings["max_controllable_per_world"],
        spawn_height_m=settings["spawn_height_m"],
        require_goal_in_bounds=settings["require_goal_in_bounds"],
        skip_if_start_in_goal=settings["skip_if_start_in_goal"],
        goal_radius_m=settings["goal_radius_m"],
        start_goal_thresh_m=settings["start_goal_thresh_m"],
        road_render_mode=settings["road_render_mode"],
        jump_break_m=settings["jump_break_m"],
        seg_width=settings["seg_width"],
        seg_height=settings["seg_height"],
        z_lift=settings["z_lift"],
        flatten_road_z=settings["flatten_road_z"],
        road_z_m=settings["road_z_m"],
        road_polyline_reduction_area=settings["road_polyline_reduction_area"],
        min_points_for_reduction=settings["min_points_for_reduction"],
        friction_cfg=settings["friction_cfg"],
        assignments=[],
    )

    if str(args_cli.viz_mode).strip().lower() == "isaacsim":
        train_selected, train_rejected = _pick_scenes_isaacsim(
            label="train",
            simulation_app=simulation_app,
            sim=sim,
            stage=stage,
            preview_cfg=preview_cfg,
            candidate_paths=all_scene_paths,
            target_count=int(settings["world_count"]),
            preview_marker_limit=int(args_cli.preview_marker_limit),
        )
    else:
        train_selected, train_rejected = _pick_scenes_matplotlib(
            label="train",
            preview_cfg=preview_cfg,
            candidate_paths=all_scene_paths,
            target_count=int(settings["world_count"]),
            preview_marker_limit=int(args_cli.preview_marker_limit),
            lane_types=lane_types,
            edge_types=edge_types,
            figsize=figsize,
            show_od_lines=bool(args_cli.show_od_lines),
        )

    actual_train_count = len(train_selected)
    if actual_train_count <= 0:
        if simulation_app is not None:
            simulation_app.close()
        raise RuntimeError("No training scenes were selected.")

    _randomize = bool(settings.get("randomize_friction", False))
    _seed = settings.get("friction_random_seed") or 42
    _rng = np.random.default_rng(int(_seed))
    if _randomize:
        train_assignments = [
            _make_assignment(path.name, _sample_friction_cfg(_rng)) for path in train_selected
        ]
    else:
        train_assignments = [_make_assignment(path.name, settings["friction_cfg"]) for path in train_selected]
    train_cfg = _apply_common_values(
        template_cfg,
        scene_json_dir=settings["scene_json_dir"],
        world_count=actual_train_count,
        grid_cols=settings["grid_cols"],
        bounds_size_m=settings["bounds_size_m"],
        padding_m=settings["padding_m"],
        base_z_m=settings["base_z_m"],
        max_controllable_per_world=settings["max_controllable_per_world"],
        spawn_height_m=settings["spawn_height_m"],
        require_goal_in_bounds=settings["require_goal_in_bounds"],
        skip_if_start_in_goal=settings["skip_if_start_in_goal"],
        goal_radius_m=settings["goal_radius_m"],
        start_goal_thresh_m=settings["start_goal_thresh_m"],
        road_render_mode=settings["road_render_mode"],
        jump_break_m=settings["jump_break_m"],
        seg_width=settings["seg_width"],
        seg_height=settings["seg_height"],
        z_lift=settings["z_lift"],
        flatten_road_z=settings["flatten_road_z"],
        road_z_m=settings["road_z_m"],
        road_polyline_reduction_area=settings["road_polyline_reduction_area"],
        min_points_for_reduction=settings["min_points_for_reduction"],
        friction_cfg=settings["friction_cfg"],
        assignments=train_assignments,
        randomize_friction=_randomize,
    )

    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    train_path = output_dir / f"{settings['base_name']}_train.yaml"
    _write_yaml(train_path, train_cfg)
    print(f"\nWrote training SceneFactory config: {train_path}")

    training_preset_template_path = Path(args_cli.training_preset_template).expanduser().resolve()
    training_preset_cfg = _build_preset_from_template(
        template_path=training_preset_template_path,
        generated_scene_cfg_path=train_path,
        world_count=actual_train_count,
        max_controllable_per_world=int(settings["max_controllable_per_world"]),
        generated_run_name=f"{settings['base_name']}_train",
    )
    training_preset_path = output_dir / f"{settings['base_name']}_train_preset.yaml"
    _write_yaml(training_preset_path, training_preset_cfg)
    print(f"Wrote runnable training preset: {training_preset_path}")

    summary_payload: dict[str, Any] = {
        "training_scene_config_path": str(train_path),
        "training_preset_path": str(training_preset_path),
        "train_selected": [path.name for path in train_selected],
        "train_rejected": sorted(train_rejected),
    }

    if _prompt_bool("\nGenerate a testing SceneFactory config from remaining eligible scenes", True):
        train_selected_names = {path.name for path in train_selected}
        test_candidates = [
            path
            for path in all_scene_paths
            if path.name not in train_selected_names and path.name not in train_rejected
        ]
        needed_test_count = len(train_selected)
        print(
            f"\nTesting selection uses the same config as training and only changes assignments.\n"
            f"Eligible scenes remaining: {len(test_candidates)} | target testing count: {needed_test_count}\n"
        )
        if str(args_cli.viz_mode).strip().lower() == "isaacsim":
            test_selected, test_rejected = _pick_scenes_isaacsim(
                label="test",
                simulation_app=simulation_app,
                sim=sim,
                stage=stage,
                preview_cfg=preview_cfg,
                candidate_paths=test_candidates,
                target_count=needed_test_count,
                preview_marker_limit=int(args_cli.preview_marker_limit),
            )
        else:
            test_selected, test_rejected = _pick_scenes_matplotlib(
                label="test",
                preview_cfg=preview_cfg,
                candidate_paths=test_candidates,
                target_count=needed_test_count,
                preview_marker_limit=int(args_cli.preview_marker_limit),
                lane_types=lane_types,
                edge_types=edge_types,
                figsize=figsize,
                show_od_lines=bool(args_cli.show_od_lines),
            )
        if not test_selected:
            print("No testing scenes selected; skipping test config generation.")
            summary_payload["test_selected"] = []
            summary_payload["test_rejected"] = sorted(test_rejected)
        else:
            # Test configs always use dry/uniform friction so evaluation results are comparable.
            test_assignments = [_make_assignment(path.name, settings["friction_cfg"]) for path in test_selected]
            test_cfg = _apply_common_values(
                template_cfg,
                scene_json_dir=settings["scene_json_dir"],
                world_count=len(test_selected),
                grid_cols=settings["grid_cols"],
                bounds_size_m=settings["bounds_size_m"],
                padding_m=settings["padding_m"],
                base_z_m=settings["base_z_m"],
                max_controllable_per_world=settings["max_controllable_per_world"],
                spawn_height_m=settings["spawn_height_m"],
                require_goal_in_bounds=settings["require_goal_in_bounds"],
                skip_if_start_in_goal=settings["skip_if_start_in_goal"],
                goal_radius_m=settings["goal_radius_m"],
                start_goal_thresh_m=settings["start_goal_thresh_m"],
                road_render_mode=settings["road_render_mode"],
                jump_break_m=settings["jump_break_m"],
                seg_width=settings["seg_width"],
                seg_height=settings["seg_height"],
                z_lift=settings["z_lift"],
                flatten_road_z=settings["flatten_road_z"],
                road_z_m=settings["road_z_m"],
                road_polyline_reduction_area=settings["road_polyline_reduction_area"],
                min_points_for_reduction=settings["min_points_for_reduction"],
                friction_cfg=settings["friction_cfg"],
                assignments=test_assignments,
                randomize_friction=False,  # eval always uniform
            )
            test_path = output_dir / f"{settings['base_name']}_test.yaml"
            _write_yaml(test_path, test_cfg)
            print(f"Wrote testing SceneFactory config: {test_path}")
            eval_preset_template_path = Path(args_cli.eval_preset_template).expanduser().resolve()
            eval_preset_cfg = _build_preset_from_template(
                template_path=eval_preset_template_path,
                generated_scene_cfg_path=test_path,
                world_count=len(test_selected),
                max_controllable_per_world=int(settings["max_controllable_per_world"]),
                generated_run_name=f"{settings['base_name']}_test_eval",
            )
            eval_preset_path = output_dir / f"{settings['base_name']}_test_eval_preset.yaml"
            _write_yaml(eval_preset_path, eval_preset_cfg)
            print(f"Wrote runnable eval preset: {eval_preset_path}")
            summary_payload["test_selected"] = [path.name for path in test_selected]
            summary_payload["test_rejected"] = sorted(test_rejected)
            summary_payload["testing_scene_config_path"] = str(test_path)
            summary_payload["testing_eval_preset_path"] = str(eval_preset_path)

    summary_path = output_dir / f"{settings['base_name']}_selection_summary.json"
    _write_summary(summary_path, summary_payload)
    print(f"Wrote selection summary: {summary_path}")

    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
