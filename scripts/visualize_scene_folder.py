"""Contact-sheet visualizer: tile every scene JSON in a folder into one grid image.

Renders a small top-down panel per scene (lanes, road edges, workzone boxes,
type-21 workzone edges, cones, and agent start/goal markers when present) and
lays them out in a grid so a whole folder can be eyeballed at a glance. Each
panel auto-frames its own scene extent.

Usage:
    PYTHONPATH=. python scripts/visualize_scene_folder.py \
        --scene-dir data/processed/workzone_variants_preview \
        --out artifacts/scene_factory/variant_contact_sheet.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORKZONE_EDGE_TYPE = 21
LANE_TYPES = {1, 2}
ROAD_EDGE_TYPES = {15, 16}


def _scene_bounds(scene: dict[str, Any]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for pl in (scene.get("road") or {}).get("polylines") or []:
        typ = int(pl.get("type", -1))
        # Frame on drivable/edge geometry, not stray markings far away.
        if typ not in LANE_TYPES and typ not in ROAD_EDGE_TYPES and typ != WORKZONE_EDGE_TYPE:
            continue
        for pt in pl.get("xyz") or []:
            if len(pt) >= 2:
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
    if not xs:
        return -100.0, 100.0, -100.0, 100.0
    return min(xs), max(xs), min(ys), max(ys)


def _draw_scene(ax: "plt.Axes", scene_path: Path) -> None:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    xmin, xmax, ymin, ymax = _scene_bounds(scene)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    ax.set_facecolor("#f7f7f4")

    def lx(pt: Any) -> float:
        return float(pt[0]) - cx

    def ly(pt: Any) -> float:
        return float(pt[1]) - cy

    # Multilane merge contract (Piece 2): highlight the closed lane vs the open
    # neighbor(s) when present. Single-lane variants have neither, so this is inert.
    boxes = (scene.get("zones") or {}).get("keepout_boxes") or []
    closed_id = boxes[0].get("closed_lane_id") if boxes else None
    open_ids = set(boxes[0].get("open_lane_ids") or []) if boxes else set()

    for pl in (scene.get("road") or {}).get("polylines") or []:
        pts = pl.get("xyz") or []
        if len(pts) < 2:
            continue
        typ = int(pl.get("type", -1))
        pid = pl.get("id")
        xs = [lx(p) for p in pts]
        ys = [ly(p) for p in pts]
        if typ == WORKZONE_EDGE_TYPE:
            ax.plot(xs, ys, color="#d7191c", linewidth=2.2, zorder=6)
        elif closed_id is not None and pid == closed_id:
            ax.plot(xs, ys, color="#c53030", linewidth=2.6, alpha=0.95, zorder=4)  # CLOSED lane
        elif pid in open_ids:
            ax.plot(xs, ys, color="#2f855a", linewidth=2.6, alpha=0.95, zorder=4)  # OPEN neighbor
        elif typ in LANE_TYPES:
            ax.plot(xs, ys, color="#2b6cb0", linewidth=0.8, alpha=0.65, zorder=2)
        elif typ in ROAD_EDGE_TYPES:
            ax.plot(xs, ys, color="#ec4899", linewidth=1.6, alpha=0.9, zorder=3)
        else:
            ax.plot(xs, ys, color="#a0aec0", linewidth=0.4, alpha=0.25, zorder=0)

    wz = scene.get("zones") or {}
    for box in wz.get("keepout_boxes") or []:
        bx, by = float(box["cx"]) - cx, float(box["cy"]) - cy
        hl, hw, yaw = float(box["half_len"]), float(box["half_wid"]), float(box["yaw_rad"])
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [
            (bx + c * dx - s * dy, by + s * dx + c * dy)
            for dx, dy in [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw), (hl, hw)]
        ]
        ax.fill([p[0] for p in corners], [p[1] for p in corners], color="#e53e3e", alpha=0.20, zorder=5)
        ax.plot([p[0] for p in corners], [p[1] for p in corners], color="#c53030", linewidth=1.3, zorder=7)

    cones = wz.get("obstacles") or []
    if cones:
        cxs = [float(c["x"]) - cx for c in cones]
        cys = [float(c["y"]) - cy for c in cones]
        ax.scatter(cxs, cys, marker="^", s=18, c="#f6ad55", edgecolors="#7b341e", linewidths=0.5, zorder=8)

    for agent in (scene.get("agents") or {}).get("items") or []:
        start = agent.get("start") or {}
        end = agent.get("end") or agent.get("goal") or {}
        if "x" in start and "y" in start:
            ax.scatter(float(start["x"]) - cx, float(start["y"]) - cy, s=14, c="#2f855a", edgecolors="white", linewidths=0.4, zorder=9)
        if "x" in end and "y" in end:
            ax.scatter(float(end["x"]) - cx, float(end["y"]) - cy, marker="*", s=40, c="#805ad5", edgecolors="white", linewidths=0.4, zorder=9)

    half = 0.5 * max(xmax - xmin, ymax - ymin, 20.0) + 8.0
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=5, length=2)
    n_box = len(wz.get("keepout_boxes") or [])
    ax.set_title(f"{scene_path.stem}  ({n_box} box, {len(cones)} cones)", fontsize=6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile every scene JSON in a folder into one grid image.")
    parser.add_argument("--scene-dir", type=Path, required=True, help="Folder of scene JSONs")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--glob", type=str, default="*.json", help="Filename glob within scene-dir")
    parser.add_argument("--cols", type=int, default=0, help="Columns (0 = auto ~sqrt)")
    parser.add_argument("--limit", type=int, default=0, help="Max scenes (0 = all)")
    parser.add_argument("--panel-in", type=float, default=2.6, help="Panel size (inches)")
    args = parser.parse_args()

    paths = sorted(p for p in args.scene_dir.glob(args.glob) if p.name != "variants_manifest.json")
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        print(f"[ERROR] no scene JSONs matching {args.glob} in {args.scene_dir}")
        return

    n = len(paths)
    cols = args.cols if args.cols > 0 else max(1, round(math.sqrt(n * 1.4)))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * args.panel_in, rows * args.panel_in), dpi=150, squeeze=False
    )
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx < n:
            try:
                _draw_scene(ax, paths[idx])
            except Exception as exc:  # keep the sheet rendering even if one scene is malformed
                ax.text(0.5, 0.5, f"{paths[idx].name}\n{exc}", fontsize=5, ha="center", va="center", color="#c53030")
                ax.set_aspect("equal")
        else:
            ax.axis("off")

    fig.suptitle(f"{args.scene_dir}  —  {n} scenes", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"Rendered {n} scenes -> {args.out}  ({cols}x{rows} grid)")


if __name__ == "__main__":
    main()
