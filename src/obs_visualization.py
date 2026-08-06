"""
Observation visualization for SceneFactory.

Parses a choco_reference observation vector from the multi-agent env and produces
a 2-panel matplotlib figure:
  Left  — top-down map: road points, car heading arrow, goal star, neighbor markers
  Right — text panel: ego scalars, weather token, road/neighbor counts

Usage (via train script test_mode):
    run_obs_visualization(env, run_dir, env_idx=0, agent_idx=0, num_warmup_steps=10)
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ── Observation segment sizes (must match choco_reference build logic) ─────────

def _road_feat_dim(include_dirs: bool) -> int:
    """Number of features per road point: 3 (x, y, type) or 5 with directions."""
    return 5 if bool(include_dirs) else 3


def _neighbor_feat_dim(include_ttc: bool, include_index: bool) -> int:
    """Number of features per neighbor vehicle slot."""
    dim = 6  # relx, rely, length, width, yaw_diff, speed
    if bool(include_ttc):
        dim += 1
    if bool(include_index):
        dim += 1
    return dim


def _compute_obs_segment_offsets(cfg: Any) -> dict[str, tuple[int, int]]:
    """Return {segment_name: (start_idx, end_idx)} for each obs block.

    Segment order must exactly match the torch.cat sequence in
    StudentVehicleMultiAgentGoalEnv._get_observations (choco_reference mode):
      ego (7) | weather (4) | road_points (k * feat) | neighbor (k * feat)
    """
    segments: dict[str, tuple[int, int]] = {}
    idx = 0

    # Ego: goal_pos_b_x, goal_pos_b_y, sin_heading, cos_heading, dist, vel_x, vel_y
    ego_end = idx + 7
    segments["ego"] = (idx, ego_end)
    idx = ego_end

    # Weather: [h_w_norm, AC, SMA, OGFC]
    if bool(getattr(cfg, "obs_weather_context_enable", True)):
        weather_end = idx + 4
        segments["weather"] = (idx, weather_end)
        idx = weather_end

    # Road points: k * feat_dim
    if bool(getattr(cfg, "obs_road_points_enable", True)):
        k = int(getattr(cfg, "obs_road_points_k", 200))
        feat = _road_feat_dim(bool(getattr(cfg, "obs_road_points_include_dirs", False)))
        road_end = idx + k * feat
        segments["road_points"] = (idx, road_end)
        segments["road_k"] = k           # type: ignore[assignment]
        segments["road_feat_dim"] = feat  # type: ignore[assignment]
        idx = road_end

    # Neighbor vehicles: k * feat_dim
    if bool(getattr(cfg, "obs_neighbor_enable", True)):
        k_n = int(getattr(cfg, "obs_neighbor_k", 63))
        feat_n = _neighbor_feat_dim(
            bool(getattr(cfg, "obs_neighbor_include_ttc", False)),
            bool(getattr(cfg, "obs_neighbor_include_index", False)),
        )
        nbr_end = idx + k_n * feat_n
        segments["neighbor"] = (idx, nbr_end)
        segments["neighbor_k"] = k_n          # type: ignore[assignment]
        segments["neighbor_feat_dim"] = feat_n  # type: ignore[assignment]
        idx = nbr_end

    segments["total_dim"] = idx  # type: ignore[assignment]
    return segments


def _parse_obs_vector(obs_vec: np.ndarray, cfg: Any) -> dict[str, Any]:
    """Decompose a flat 1-D observation vector into named segments.

    Returns a dict with keys:
      ego         — dict of scalar ego features (goal_dx, goal_dy, sin_h, cos_h, dist, vx, vy)
      weather     — 4-element array [h_w, AC, SMA, OGFC] or None
      road_points — (N, feat_dim) float array of road-point features in ego frame, or None
      neighbor    — (K, feat_dim) float array of neighbor features in ego frame, or None
      segments    — raw segment offset dict
    """
    segs = _compute_obs_segment_offsets(cfg)

    # Ego
    s, e = segs["ego"]
    ego_raw = obs_vec[s:e]
    bounds_scale = float(getattr(cfg, "_scene_factory_bounds_size_m", 100.0))
    result: dict[str, Any] = {
        "ego": {
            "goal_dx_norm": float(ego_raw[0]),
            "goal_dy_norm": float(ego_raw[1]),
            "sin_heading_err": float(ego_raw[2]),
            "cos_heading_err": float(ego_raw[3]),
            "goal_dist_norm": float(ego_raw[4]),
            "vel_x_norm": float(ego_raw[5]),
            "vel_y_norm": float(ego_raw[6]),
        },
        "weather": None,
        "road_points": None,
        "neighbor": None,
        "segments": segs,
    }

    # Weather
    if "weather" in segs:
        ws, we = segs["weather"]
        result["weather"] = obs_vec[ws:we].copy()

    # Road points
    if "road_points" in segs:
        rs, re = segs["road_points"]
        k = int(segs["road_k"])  # type: ignore[arg-type]
        fd = int(segs["road_feat_dim"])  # type: ignore[arg-type]
        road_raw = obs_vec[rs:re].reshape(k, fd)
        # Keep only non-zero rows (padded zeros = no road point there)
        valid_mask = np.any(road_raw != 0.0, axis=1)
        result["road_points"] = road_raw[valid_mask]

    # Neighbor vehicles
    if "neighbor" in segs:
        ns, ne = segs["neighbor"]
        k_n = int(segs["neighbor_k"])  # type: ignore[arg-type]
        fd_n = int(segs["neighbor_feat_dim"])  # type: ignore[arg-type]
        nbr_raw = obs_vec[ns:ne].reshape(k_n, fd_n)
        valid_mask_n = np.any(nbr_raw != 0.0, axis=1)
        result["neighbor"] = nbr_raw[valid_mask_n]

    return result


def _make_figure(parsed: dict[str, Any], cfg: Any, env_idx: int, agent_idx: int) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Build the 2-panel figure from parsed observation data.

    Left panel  — top-down map in ego-centered coordinates (meters, de-normalized).
    Right panel — text display of scalar obs values.
    """
    import matplotlib  # lazy import so module loads without matplotlib on import
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # De-normalisation scales
    bounds_scale = float(getattr(cfg, "_scene_factory_bounds_size_m", 100.0))
    distance_scale = bounds_scale * math.sqrt(2.0)
    road_radius_m = float(getattr(cfg, "obs_road_points_radius_m", 35.0))
    road_norm = road_radius_m if road_radius_m > 0.0 else bounds_scale
    neighbor_scale = float(getattr(cfg, "agent_neighbor_obs_scale_m", 100.0))
    speed_scale = 10.0  # m/s
    type_norm = float(getattr(cfg, "obs_road_points_type_norm", 20.0))

    ego = parsed["ego"]
    weather = parsed["weather"]
    road_pts = parsed["road_points"]
    neighbors = parsed["neighbor"]

    # Ego scalars (de-normalised)
    goal_dx_m = ego["goal_dx_norm"] * bounds_scale
    goal_dy_m = ego["goal_dy_norm"] * bounds_scale
    goal_dist_m = ego["goal_dist_norm"] * distance_scale
    heading_err_rad = math.atan2(ego["sin_heading_err"], ego["cos_heading_err"])
    vx_mps = ego["vel_x_norm"] * speed_scale
    vy_mps = ego["vel_y_norm"] * speed_scale
    speed_mps = math.hypot(vx_mps, vy_mps)

    fig, (ax_map, ax_text) = plt.subplots(
        1, 2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )
    fig.patch.set_facecolor("#1a1a2e")

    # ── Left: top-down map ─────────────────────────────────────────────────────
    ax_map.set_facecolor("#16213e")
    ax_map.set_aspect("equal")
    ax_map.tick_params(colors="white")
    ax_map.spines[:].set_color("#4a4e69")
    ax_map.set_xlabel("Ego-frame X (m, forward →)", color="white")
    ax_map.set_ylabel("Ego-frame Y (m, left ←)", color="white")
    ax_map.set_title(
        f"Top-down observation map  [env={env_idx}, agent={agent_idx}]",
        color="white", fontsize=11,
    )

    # Road points
    if road_pts is not None and len(road_pts) > 0:
        rx = road_pts[:, 0] * road_norm
        ry = road_pts[:, 1] * road_norm
        rtype = road_pts[:, 2] * type_norm if type_norm > 0.0 else road_pts[:, 2]
        rtype_i = np.rint(rtype).astype(int)

        type_styles = [
            (rtype_i == 1, "#4ade80", 10, 0.55, "lane_center (1)"),
            (np.isin(rtype_i, [2, 3]), "#9ca3af", 10, 0.55, "lane/boundary (2/3)"),
            (rtype_i == 6, "#facc15", 12, 0.65, "divider (6)"),
            (np.isin(rtype_i, [15, 16]), "#fb923c", 14, 0.75, "forbidden edge (15/16)"),
            (rtype_i == 10, "#22d3ee", 42, 0.95, "cone obs (10)"),
            (rtype_i == 21, "#ef4444", 55, 1.0, "workzone boundary (21)"),
        ]
        styled = np.zeros_like(rtype_i, dtype=bool)
        for mask, color, size, alpha, label in type_styles:
            if np.any(mask):
                styled |= mask
                ax_map.scatter(rx[mask], ry[mask], c=color, s=size, alpha=alpha, zorder=4 if label.startswith(("cone", "workzone")) else 2, label=label)
        other = ~styled
        if np.any(other):
            ax_map.scatter(rx[other], ry[other], c="#c084fc", s=8, alpha=0.45, zorder=2, label="other road type")

    # Actual forbidden workzone boxes, transformed from env-local coordinates into ego frame.
    for box in parsed.get("keepout_boxes_ego", []) or []:
        corners = np.asarray(box.get("corners_ego_m", []), dtype=np.float32)
        if corners.shape == (4, 2):
            poly = mpatches.Polygon(
                corners, closed=True, fill=True, facecolor="#dc2626", edgecolor="#ffffff",
                alpha=0.22, linewidth=1.3, zorder=3, label="forbidden box footprint",
            )
            ax_map.add_patch(poly)

    # Car at origin with heading arrow
    ax_map.scatter([0], [0], c="#f472b6", s=120, zorder=5, marker="s", label="ego vehicle")
    arrow_len = max(3.0, road_norm * 0.08)
    ax_map.annotate(
        "", xy=(arrow_len, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#f472b6", lw=2.5),
        zorder=6,
    )

    # Goal
    ax_map.scatter([goal_dx_m], [goal_dy_m], c="#fbbf24", s=180, marker="*", zorder=5, label=f"goal ({goal_dist_m:.1f} m)")

    # Neighbor vehicles
    if neighbors is not None and len(neighbors) > 0:
        nbr_rx = neighbors[:, 0] * neighbor_scale
        nbr_ry = neighbors[:, 1] * neighbor_scale
        ax_map.scatter(
            nbr_rx, nbr_ry,
            c="#f87171", s=80, marker="^", zorder=4,
            label=f"neighbors ({len(nbr_rx)})",
        )
        for i, (nx, ny) in enumerate(zip(nbr_rx, nbr_ry)):
            ax_map.annotate(str(i), (nx, ny), color="#fca5a5", fontsize=7, zorder=5,
                            xytext=(2, 2), textcoords="offset points")

    # Axes limits: centred on scene with a bit of margin
    view_r = max(road_norm * 1.1, abs(goal_dx_m) * 1.1, abs(goal_dy_m) * 1.1, 15.0)
    ax_map.set_xlim(-view_r, view_r)
    ax_map.set_ylim(-view_r, view_r)

    ax_map.axhline(0, color="#4a4e69", lw=0.5, zorder=1)
    ax_map.axvline(0, color="#4a4e69", lw=0.5, zorder=1)
    legend = ax_map.legend(
        loc="lower left", fontsize=7.5,
        facecolor="#0f3460", edgecolor="#4a4e69", labelcolor="white",
    )

    # ── Right: text panel ──────────────────────────────────────────────────────
    ax_text.set_facecolor("#0f3460")
    ax_text.axis("off")
    ax_text.set_title("Observation values", color="white", fontsize=11)

    lines: list[str] = []

    lines.append("━━━  EGO SCALARS  ━━━")
    lines.append(f"  speed           : {speed_mps:.3f} m/s")
    lines.append(f"  vel_x (fwd)     : {vx_mps:.3f} m/s")
    lines.append(f"  vel_y (lat)     : {vy_mps:.3f} m/s")
    lines.append(f"  heading_err     : {math.degrees(heading_err_rad):+.1f} deg  ({heading_err_rad:+.3f} rad)")
    lines.append(f"  goal_dx         : {goal_dx_m:.2f} m")
    lines.append(f"  goal_dy         : {goal_dy_m:.2f} m")
    lines.append(f"  goal_dist       : {goal_dist_m:.2f} m")
    lines.append("")

    if weather is not None:
        lines.append("━━━  WEATHER TOKEN  ━━━")
        lines.append(f"  h_w (norm)      : {weather[0]:.4f}")
        lines.append(f"  road_AC         : {weather[1]:.1f}")
        lines.append(f"  road_SMA        : {weather[2]:.1f}")
        lines.append(f"  road_OGFC       : {weather[3]:.1f}")
        road_types = ["AC", "SMA", "OGFC"]
        active = [t for t, v in zip(road_types, weather[1:]) if v > 0.5]
        lines.append(f"  surface type    : {active[0] if active else 'dry/unset'}")
        lines.append("")

    n_road = len(road_pts) if road_pts is not None else 0
    n_nbr = len(neighbors) if neighbors is not None else 0
    lines.append("━━━  SEGMENT COUNTS  ━━━")
    lines.append(f"  road pts (valid): {n_road}  of  k={int(parsed['segments'].get('road_k', 0))}")
    lines.append(f"  neighbors (live): {n_nbr}  of  k={int(parsed['segments'].get('neighbor_k', 0))}")
    lines.append(f"  obs vector dim  : {int(parsed['segments']['total_dim'])}")
    if road_pts is not None and len(road_pts) > 0:
        rtype = road_pts[:, 2] * type_norm if type_norm > 0.0 else road_pts[:, 2]
        rtype_i = np.rint(rtype).astype(int)
        unique, counts = np.unique(rtype_i, return_counts=True)
        count_map = {int(k): int(v) for k, v in zip(unique, counts)}
        lines.append(f"  cones type10    : {count_map.get(10, 0)}")
        lines.append(f"  wz boundary t21 : {count_map.get(21, 0)}")
        for type_id, label in [(10, "nearest cone"), (21, "nearest t21")]:
            mask = rtype_i == type_id
            if np.any(mask):
                dx_m = road_pts[mask, 0] * road_norm
                dy_m = road_pts[mask, 1] * road_norm
                lines.append(f"  {label:<15}: {np.hypot(dx_m, dy_m).min():.2f} m")
            else:
                lines.append(f"  {label:<15}: not in obs")
    lines.append("")

    if neighbors is not None and len(neighbors) > 0:
        lines.append("━━━  NEIGHBOR DETAIL (top 5)  ━━━")
        for i, row in enumerate(neighbors[:5]):
            nx_m = row[0] * neighbor_scale
            ny_m = row[1] * neighbor_scale
            dist_m = math.hypot(nx_m, ny_m)
            spd_n = row[5] * speed_scale
            yaw_n = math.degrees(row[4] * math.pi)
            extra = ""
            if len(row) > 6:
                ttc = row[6] * float(getattr(cfg, "obs_neighbor_ttc_max_s", 10.0))
                extra = f"  TTC={ttc:.1f}s"
            lines.append(f"  [{i}] d={dist_m:.1f}m  spd={spd_n:.1f}m/s  yaw={yaw_n:+.0f}°{extra}")

    ax_text.text(
        0.04, 0.97,
        "\n".join(lines),
        transform=ax_text.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        fontfamily="monospace",
        color="#e2e8f0",
    )

    fig.tight_layout(pad=1.5)
    return fig


# ── Runtime reconstruction helpers ─────────────────────────────────────────────

def _yaw_from_quat_wxyz(q: np.ndarray) -> float:
    qw, qx, qy, qz = [float(v) for v in q[:4]]
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _world_delta_to_ego_xy(dx: np.ndarray, dy: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    cy = math.cos(float(yaw))
    sy = math.sin(float(yaw))
    return cy * dx + sy * dy, -sy * dx + cy * dy


def _workzone_boxes_to_ego(env: Any, env_idx: int, agent_idx: int) -> list[dict[str, Any]]:
    if not hasattr(env, "_keepout_boxes_m") or env._keepout_boxes_m.shape[1] == 0:
        return []
    vehicle = env._vehicles[agent_idx]
    root_pos_w = vehicle.data.root_pos_w[env_idx].detach().cpu().numpy()
    root_quat_w = vehicle.data.root_quat_w[env_idx].detach().cpu().numpy()
    ego_yaw = _yaw_from_quat_wxyz(root_quat_w)
    env_origin_xy = env.scene.env_origins[env_idx, :2].detach().cpu().numpy()
    ego_local_xy = root_pos_w[:2] - env_origin_xy

    boxes = env._keepout_boxes_m[env_idx].detach().cpu().numpy()
    n_boxes = int(env._keepout_boxes_count[env_idx].detach().cpu().item()) if hasattr(env, "_keepout_boxes_count") else boxes.shape[0]
    result: list[dict[str, Any]] = []
    for b in boxes[:n_boxes]:
        cx, cy, half_len, half_wid, yaw = [float(v) for v in b]
        fwd = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        right = np.array([math.sin(yaw), -math.cos(yaw)], dtype=np.float32)
        center = np.array([cx, cy], dtype=np.float32)
        corners_local = np.stack([
            center + fwd * half_len + right * half_wid,
            center + fwd * half_len - right * half_wid,
            center - fwd * half_len - right * half_wid,
            center - fwd * half_len + right * half_wid,
        ], axis=0)
        rel = corners_local - ego_local_xy.reshape(1, 2)
        ex, ey = _world_delta_to_ego_xy(rel[:, 0], rel[:, 1], ego_yaw)
        center_rel = center - ego_local_xy
        cx_ego, cy_ego = _world_delta_to_ego_xy(
            np.asarray([center_rel[0]], dtype=np.float32),
            np.asarray([center_rel[1]], dtype=np.float32),
            ego_yaw,
        )
        result.append({
            "center_ego_m": [float(cx_ego[0]), float(cy_ego[0])],
            "half_len_m": float(half_len),
            "half_wid_m": float(half_wid),
            "yaw_env_rad": float(yaw),
            "corners_ego_m": [[float(x), float(y)] for x, y in zip(ex, ey)],
        })
    return result


def _road_type_summary(road_pts: np.ndarray | None, type_norm: float) -> dict[str, Any]:
    if road_pts is None or len(road_pts) == 0:
        return {"counts": {}, "nearest_m": {}}
    rtype = road_pts[:, 2] * type_norm if type_norm > 0.0 else road_pts[:, 2]
    rtype_i = np.rint(rtype).astype(int)
    unique, counts = np.unique(rtype_i, return_counts=True)
    nearest: dict[str, float] = {}
    road_norm = float(type_norm)  # placeholder, overwritten by caller for distances where needed
    return {"counts": {str(int(k)): int(v) for k, v in zip(unique, counts)}, "type_ids": rtype_i.tolist()}

# ── Main entry point called from train script ──────────────────────────────────

def run_obs_visualization(
    env: Any,
    run_dir: Path,
    *,
    env_idx: int = 0,
    agent_idx: int = 0,
    num_warmup_steps: int = 10,
    warmup_action: Any | None = None,
) -> None:
    """Run the observation visualization test mode.

    Steps the env for `num_warmup_steps` with zero actions so vehicles have
    valid physics state, then extracts the observation for (env_idx, agent_idx),
    parses it according to the obs contract, and saves a figure to
    artifacts/obs_debug/<timestamp>_obs_viz.png.

    Args:
        env:             Initialized StudentVehicleMultiAgentGoalEnv (already reset).
        run_dir:         Logging directory for this run (used for metadata, not the figure).
        env_idx:         Which parallel environment instance to visualize.
        agent_idx:       Which vehicle slot within that environment to visualize.
        num_warmup_steps: Steps to simulate before capturing the observation.
    """
    cfg = env.cfg
    observation_mode = str(getattr(cfg, "observation_mode", "choco_reference")).strip().lower()
    if observation_mode != "choco_reference":
        print(
            f"[ObsViz] WARNING: observation_mode={observation_mode!r}, "
            "expected 'choco_reference'. The obs parser targets choco_reference layout; "
            "results may be incorrect for other modes.",
            flush=True,
        )

    # Validate indices
    if env_idx >= env.num_envs:
        raise ValueError(f"env_idx={env_idx} out of range [0, {env.num_envs})")
    num_agents = int(getattr(cfg, "num_agents_per_env", 1))
    if agent_idx >= num_agents:
        raise ValueError(f"agent_idx={agent_idx} out of range [0, {num_agents})")

    agent_id = f"vehicle_{agent_idx}"

    print(f"[ObsViz] Resetting env ({env.num_envs} worlds)...", flush=True)
    env.reset()

    device = env.device
    n = env.num_envs
    if warmup_action is None:
        action_vec = torch.zeros((3,), dtype=torch.float32, device=device)
        action_desc = "zero action"
    else:
        action_vec = torch.as_tensor(warmup_action, dtype=torch.float32, device=device).reshape(3).clamp(-1.0, 1.0)
        action_desc = f"fixed action {action_vec.detach().cpu().tolist()}"
    print(f"[ObsViz] Warming up for {num_warmup_steps} steps ({action_desc})...", flush=True)

    def _warmup_actions() -> dict[str, torch.Tensor]:
        return {
            f"vehicle_{ai}": action_vec.unsqueeze(0).repeat(n, 1)
            for ai in range(num_agents)
        }

    step_output = None
    for step in range(num_warmup_steps):
        step_output = env.step(_warmup_actions())
        if (step + 1) % max(1, num_warmup_steps // 3) == 0:
            print(f"  step {step + 1}/{num_warmup_steps}", flush=True)

    # Extract observations from last step output
    # env.step returns (obs_dict, rewards, terminated, truncated, info)
    # obs_dict is a dict[agent_id, Tensor[num_envs, obs_dim]]
    if step_output is None:
        # Edge case: warmup_steps=0, call _get_observations directly
        obs_dict = env._get_observations()
    else:
        obs_dict = step_output[0]

    if agent_id not in obs_dict:
        raise KeyError(
            f"agent_id={agent_id!r} not in obs_dict. "
            f"Available: {list(obs_dict.keys())}"
        )

    obs_tensor = obs_dict[agent_id]  # [num_envs, obs_dim]
    obs_vec_np = obs_tensor[env_idx].detach().cpu().numpy()

    print(
        f"[ObsViz] Captured obs for env={env_idx} agent={agent_idx}: "
        f"shape={obs_vec_np.shape}  non-zero={int(np.count_nonzero(obs_vec_np))}",
        flush=True,
    )

    # Expose normalisation scale on cfg for de-normalisation in the figure
    # (the env stores this as a private attribute after scene load)
    bounds_size = float(getattr(env, "_scene_factory_bounds_size_m", 100.0))
    cfg._scene_factory_bounds_size_m = bounds_size  # type: ignore[attr-defined]

    parsed = _parse_obs_vector(obs_vec_np, cfg)
    parsed["keepout_boxes_ego"] = _workzone_boxes_to_ego(env, env_idx, agent_idx)

    print("[ObsViz] Building figure...", flush=True)
    fig = _make_figure(parsed, cfg, env_idx, agent_idx)

    # Save output
    out_dir = Path("artifacts/obs_debug")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{timestamp}_env{env_idx}_agent{agent_idx}_obs_viz.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[ObsViz] Figure saved → {out_path.resolve()}", flush=True)

    # Also save a JSON dump of the parsed obs for debugging
    json_path = out_dir / f"{timestamp}_env{env_idx}_agent{agent_idx}_obs_parsed.json"
    import json
    road_pts = parsed["road_points"]
    type_norm = float(getattr(cfg, "obs_road_points_type_norm", 20.0))
    road_norm = float(getattr(cfg, "obs_road_points_radius_m", 35.0))
    road_type_counts: dict[str, int] = {}
    nearest_by_type_m: dict[str, float] = {}
    if road_pts is not None and len(road_pts) > 0:
        rtype_i = np.rint((road_pts[:, 2] * type_norm) if type_norm > 0.0 else road_pts[:, 2]).astype(int)
        unique, counts = np.unique(rtype_i, return_counts=True)
        road_type_counts = {str(int(k)): int(v) for k, v in zip(unique, counts)}
        for tid in sorted(set(rtype_i.tolist())):
            mask = rtype_i == int(tid)
            dist = np.hypot(road_pts[mask, 0] * road_norm, road_pts[mask, 1] * road_norm)
            nearest_by_type_m[str(int(tid))] = float(dist.min()) if len(dist) else float("nan")
    dump: dict[str, Any] = {
        "env_idx": env_idx,
        "agent_idx": agent_idx,
        "obs_dim": int(obs_vec_np.shape[0]),
        "ego": parsed["ego"],
        "weather": parsed["weather"].tolist() if parsed["weather"] is not None else None,
        "road_points_valid_count": len(road_pts) if road_pts is not None else 0,
        "road_point_type_counts": road_type_counts,
        "nearest_road_point_by_type_m": nearest_by_type_m,
        "keepout_boxes_ego": parsed.get("keepout_boxes_ego", []),
        "neighbor_valid_count": len(parsed["neighbor"]) if parsed["neighbor"] is not None else 0,
    }
    json_path.write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")
    print(f"[ObsViz] Parsed obs JSON → {json_path.resolve()}", flush=True)

    if road_pts is not None and len(road_pts) > 0:
        csv_path = out_dir / f"{timestamp}_env{env_idx}_agent{agent_idx}_road_points.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["slot", "x_ego_m", "y_ego_m", "type_id"]
            if road_pts.shape[1] >= 5:
                header += ["dir_x_ego", "dir_y_ego"]
            writer.writerow(header)
            rtype_i = np.rint((road_pts[:, 2] * type_norm) if type_norm > 0.0 else road_pts[:, 2]).astype(int)
            for slot, row in enumerate(road_pts):
                values = [slot, float(row[0] * road_norm), float(row[1] * road_norm), int(rtype_i[slot])]
                if road_pts.shape[1] >= 5:
                    values += [float(row[3]), float(row[4])]
                writer.writerow(values)
        print(f"[ObsViz] Road-point CSV → {csv_path.resolve()}", flush=True)

    import matplotlib.pyplot as plt
    plt.close(fig)
