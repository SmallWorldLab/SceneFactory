"""Generate per-world friction reports from chocolate curriculum YAML configs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import yaml

try:
    from . import ROAD_TYPE_ORDER, estimate_friction, friction_input_from_mapping
    from .world_pipeline import expand_world_assignments
except ImportError:  # pragma: no cover - supports direct script execution.
    from src.trfc import ROAD_TYPE_ORDER, estimate_friction, friction_input_from_mapping
    from src.trfc.world_pipeline import expand_world_assignments


ROAD_COLOR_MAP = {
    "AC": "#4C78A8",
    "SMA": "#59A14F",
    "OGFC": "#E15759",
    "unknown": "#9C755F",
}


@dataclass(frozen=True)
class WorldReportRow:
    world_index: int
    scene_json_name: str
    friction_source: str
    road_type: str | None
    precip_type: str | None
    precip_intensity_mmph: float | None
    water_film_mm: float | None
    tire_model_id: str | None
    reference_speed_mps: float | None
    slip_static: float | None
    slip_dynamic: float | None
    mu_eff_mode: str | None
    theta_texture: float | None
    texture_amplitude_mm: float | None
    model_mu_static: float | None
    model_mu_dynamic: float | None
    applied_static_friction: float
    applied_dynamic_friction: float
    effective_friction: float
    params_source: str | None

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def _as_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _coerce_scene_json_name(scene_json: Any) -> str:
    if scene_json is None:
        return ""
    text = str(scene_json).strip()
    if not text:
        return ""
    if not text.endswith(".json"):
        text = f"{text}.json"
    return text


def _sequence_or_empty(value: Any) -> Sequence[Any]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Expected a sequence")
    return value


def _legacy_row(
    *,
    world_index: int,
    scene_json_name: str,
    friction_value: float,
) -> WorldReportRow:
    friction_value = float(friction_value)
    return WorldReportRow(
        world_index=int(world_index),
        scene_json_name=scene_json_name,
        friction_source="legacy_friction_values",
        road_type=None,
        precip_type=None,
        precip_intensity_mmph=None,
        water_film_mm=None,
        tire_model_id=None,
        reference_speed_mps=None,
        slip_static=None,
        slip_dynamic=None,
        mu_eff_mode=None,
        theta_texture=None,
        texture_amplitude_mm=None,
        model_mu_static=friction_value,
        model_mu_dynamic=friction_value,
        applied_static_friction=friction_value,
        applied_dynamic_friction=friction_value,
        effective_friction=friction_value,
        params_source=None,
    )


def _estimated_row(
    *,
    world_index: int,
    scene_json_name: str,
    friction_cfg: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> WorldReportRow:
    request = friction_input_from_mapping(friction_cfg, defaults=defaults)
    estimate = estimate_friction(request)
    model_mu_static = float(estimate.mu_static)
    model_mu_dynamic = float(estimate.mu_dynamic)
    applied_static_friction = max(model_mu_static, model_mu_dynamic)
    applied_dynamic_friction = min(model_mu_static, model_mu_dynamic)
    return WorldReportRow(
        world_index=int(world_index),
        scene_json_name=scene_json_name,
        friction_source="pipeline_estimate",
        road_type=estimate.road_type,
        precip_type=getattr(request, "precip_type", None),
        precip_intensity_mmph=getattr(request, "precip_intensity_mmph", None),
        water_film_mm=float(estimate.water_film_mm),
        tire_model_id=estimate.tire_model_id,
        reference_speed_mps=float(request.reference_speed_mps),
        slip_static=float(request.slip_static),
        slip_dynamic=float(request.slip_dynamic),
        mu_eff_mode=str(request.mu_eff_mode),
        theta_texture=float(estimate.theta_texture),
        texture_amplitude_mm=float(estimate.texture_amplitude_mm),
        model_mu_static=model_mu_static,
        model_mu_dynamic=model_mu_dynamic,
        applied_static_friction=applied_static_friction,
        applied_dynamic_friction=applied_dynamic_friction,
        effective_friction=float(estimate.mu_eff),
        params_source=estimate.params_source,
    )


def build_world_report_rows(cfg: Mapping[str, Any]) -> list[WorldReportRow]:
    world_cfg = _as_mapping("world", cfg.get("world", {}))
    ground_cfg = _as_mapping("ground", cfg.get("ground", {}))
    io_cfg = _as_mapping("io", cfg.get("io", {}))

    world_count = int(world_cfg["world_count"])
    assignments = _sequence_or_empty(world_cfg.get("assignments"))
    explicit_scene_jsons = _sequence_or_empty(io_cfg.get("scene_jsons"))

    pipeline_cfg = _as_mapping(
        "ground.friction_pipeline",
        ground_cfg.get("friction_pipeline", {}) or {},
    )
    pipeline_defaults = _as_mapping(
        "ground.friction_pipeline.defaults",
        pipeline_cfg.get("defaults", {}) or {},
    )
    require_friction = bool(pipeline_cfg.get("enable", False))

    friction_values = [float(v) for v in ground_cfg.get("friction_values", [0.5]) if v is not None]
    if not friction_values:
        friction_values = [0.5]

    rows: list[WorldReportRow] = []

    if assignments:
        assignments = expand_world_assignments(assignments, world_count=world_count, world_cfg=world_cfg)
        for world_index, entry in enumerate(assignments):
            entry_mapping = _as_mapping(f"world.assignments[{world_index}]", entry)
            scene_json_name = _coerce_scene_json_name(entry_mapping.get("scene_json"))
            friction_cfg = entry_mapping.get("friction")
            if friction_cfg is None:
                if require_friction:
                    raise ValueError(
                        "ground.friction_pipeline.enable=true requires a friction block "
                        f"for world.assignments[{world_index}]"
                    )
                rows.append(
                    _legacy_row(
                        world_index=world_index,
                        scene_json_name=scene_json_name,
                        friction_value=friction_values[world_index % len(friction_values)],
                    )
                )
                continue
            rows.append(
                _estimated_row(
                    world_index=world_index,
                    scene_json_name=scene_json_name,
                    friction_cfg=_as_mapping(
                        f"world.assignments[{world_index}].friction",
                        friction_cfg,
                    ),
                    defaults=pipeline_defaults,
                )
            )
        return rows

    for world_index in range(world_count):
        scene_json_name = ""
        if world_index < len(explicit_scene_jsons):
            scene_json_name = _coerce_scene_json_name(explicit_scene_jsons[world_index])
        rows.append(
            _legacy_row(
                world_index=world_index,
                scene_json_name=scene_json_name,
                friction_value=friction_values[world_index % len(friction_values)],
            )
        )

    return rows


def load_curriculum_config(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Curriculum YAML must load to a mapping, got {type(loaded).__name__}")
    return loaded


def _numeric_stats(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": mean(ordered),
        "median": median(ordered),
        "max": ordered[-1],
    }


def summarize_world_report(rows: Sequence[WorldReportRow]) -> dict[str, Any]:
    road_counts = Counter((row.road_type or "unknown") for row in rows)
    precip_counts = Counter((row.precip_type or "unknown") for row in rows)
    source_counts = Counter(row.friction_source for row in rows)

    effective = [row.effective_friction for row in rows]
    applied_static = [row.applied_static_friction for row in rows]
    applied_dynamic = [row.applied_dynamic_friction for row in rows]
    water = [row.water_film_mm for row in rows if row.water_film_mm is not None]
    precip_intensity = [
        row.precip_intensity_mmph for row in rows if row.precip_intensity_mmph is not None
    ]

    return {
        "world_count": len(rows),
        "friction_source_counts": dict(sorted(source_counts.items())),
        "road_type_counts": dict(sorted(road_counts.items())),
        "precip_type_counts": dict(sorted(precip_counts.items())),
        "effective_friction_stats": _numeric_stats(effective),
        "applied_static_friction_stats": _numeric_stats(applied_static),
        "applied_dynamic_friction_stats": _numeric_stats(applied_dynamic),
        "water_film_mm_stats": _numeric_stats(water),
        "precip_intensity_mmph_stats": _numeric_stats(precip_intensity),
    }


def _markdown_count_rows(counter: Mapping[str, int]) -> str:
    total = sum(counter.values())
    if total <= 0:
        return "- none"
    lines = []
    for key, value in counter.items():
        pct = 100.0 * float(value) / float(total)
        lines.append(f"- `{key}`: {value} ({pct:.1f}%)")
    return "\n".join(lines)


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _render_markdown_table(rows: Sequence[WorldReportRow]) -> str:
    columns = [
        ("world", lambda row: str(row.world_index)),
        ("scene_json", lambda row: row.scene_json_name),
        ("source", lambda row: row.friction_source),
        ("road", lambda row: row.road_type or ""),
        ("precip", lambda row: row.precip_type or ""),
        ("precip_mmph", lambda row: _format_float(row.precip_intensity_mmph)),
        ("water_mm", lambda row: _format_float(row.water_film_mm)),
        ("v_ref_mps", lambda row: _format_float(row.reference_speed_mps)),
        ("slip_static", lambda row: _format_float(row.slip_static)),
        ("slip_dynamic", lambda row: _format_float(row.slip_dynamic)),
        ("mu_static", lambda row: _format_float(row.model_mu_static)),
        ("mu_dynamic", lambda row: _format_float(row.model_mu_dynamic)),
        ("mu_eff", lambda row: _format_float(row.effective_friction)),
        ("applied_static", lambda row: _format_float(row.applied_static_friction)),
        ("applied_dynamic", lambda row: _format_float(row.applied_dynamic_friction)),
    ]
    header = "| " + " | ".join(name for name, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(render(row) for _, render in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def render_markdown_report(
    *,
    config_path: str | Path,
    rows: Sequence[WorldReportRow],
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> str:
    output_root = Path(output_dir)
    effective_stats = summary.get("effective_friction_stats") or {}
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return "\n".join(
        [
            f"# Curriculum World Report: {Path(config_path).name}",
            "",
            f"- Generated: {generated_at}",
            f"- Config: `{Path(config_path).expanduser().resolve()}`",
            f"- Output dir: `{output_root.resolve()}`",
            f"- Worlds: {summary.get('world_count', len(rows))}",
            "",
            "## Friction Summary",
            "",
            f"- Effective friction min / mean / median / max: "
            f"{effective_stats.get('min', 0.0):.4f} / "
            f"{effective_stats.get('mean', 0.0):.4f} / "
            f"{effective_stats.get('median', 0.0):.4f} / "
            f"{effective_stats.get('max', 0.0):.4f}",
            f"- Friction source counts: {json.dumps(summary.get('friction_source_counts', {}), sort_keys=True)}",
            "",
            "## Road Distribution",
            "",
            _markdown_count_rows(summary.get("road_type_counts", {})),
            "",
            "## Weather Distribution",
            "",
            "### Precipitation Type",
            "",
            _markdown_count_rows(summary.get("precip_type_counts", {})),
            "",
            "### Water Film and Precipitation Intensity",
            "",
            f"- Water film stats: {json.dumps(summary.get('water_film_mm_stats', {}), sort_keys=True)}",
            f"- Precip intensity stats: {json.dumps(summary.get('precip_intensity_mmph_stats', {}), sort_keys=True)}",
            "",
            "## Per-World Inputs and Outputs",
            "",
            _render_markdown_table(rows),
            "",
            "## Generated Files",
            "",
            "- `world_report.md`",
            "- `world_report.csv`",
            "- `world_report.json`",
            "- `dashboard.svg`",
            "- `water_vs_friction.svg`",
        ]
    )


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int = 14,
    weight: str = "normal",
    anchor: str = "start",
    fill: str = "#111111",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans Mono, monospace" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{escape(text)}</text>"
    )


def _svg_line(x1: float, y1: float, x2: float, y2: float, *, stroke: str, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}" />'
    )


def _svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 0.0,
    rx: float = 0.0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.1f}" rx="{rx:.1f}" />'
    )


def _svg_circle(cx: float, cy: float, r: float, *, fill: str, stroke: str = "none", stroke_width: float = 0.0) -> str:
    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width:.1f}" />'
    )


def _svg_document(width: int, height: int, elements: Sequence[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            _svg_rect(0, 0, width, height, fill="#F7F7F5"),
            *elements,
            "</svg>",
        ]
    )


def _panel_bounds(x: float, y: float, width: float, height: float) -> tuple[float, float, float, float]:
    return x + 58.0, y + 42.0, width - 80.0, height - 76.0


def _scale_y(value: float, y_min: float, y_max: float, plot_y: float, plot_h: float) -> float:
    if y_max <= y_min:
        return plot_y + plot_h
    return plot_y + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h


def _axis_ticks(y_min: float, y_max: float, count: int = 4) -> list[float]:
    if count <= 0:
        return []
    if y_max <= y_min:
        y_max = y_min + 1.0
    step = (y_max - y_min) / float(count)
    return [y_min + step * idx for idx in range(count + 1)]


def _svg_frame(title: str, x: float, y: float, width: float, height: float) -> list[str]:
    return [
        _svg_rect(x, y, width, height, fill="#FFFFFF", stroke="#D6D3CE", stroke_width=1.0, rx=12.0),
        _svg_text(x + 18.0, y + 26.0, title, size=18, weight="bold"),
    ]


def _render_friction_panel(rows: Sequence[WorldReportRow], x: float, y: float, width: float, height: float) -> list[str]:
    elements = _svg_frame("Per-World Friction", x, y, width, height)
    if not rows:
        elements.append(_svg_text(x + width / 2.0, y + height / 2.0, "No data", anchor="middle"))
        return elements

    plot_x, plot_y, plot_w, plot_h = _panel_bounds(x, y, width, height)
    y_min = 0.0
    y_max = max(
        max(row.applied_static_friction for row in rows),
        max(row.effective_friction for row in rows),
    ) * 1.15
    elements.extend(
        [
            _svg_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#666666"),
            _svg_line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke="#666666"),
        ]
    )
    for tick in _axis_ticks(y_min, y_max):
        tick_y = _scale_y(tick, y_min, y_max, plot_y, plot_h)
        elements.append(_svg_line(plot_x, tick_y, plot_x + plot_w, tick_y, stroke="#ECEAE5"))
        elements.append(_svg_text(plot_x - 8.0, tick_y + 4.0, f"{tick:.2f}", size=11, anchor="end", fill="#555555"))

    slot_w = plot_w / float(max(1, len(rows)))
    bar_w = slot_w * 0.56
    for idx, row in enumerate(rows):
        cx = plot_x + slot_w * (idx + 0.5)
        color = ROAD_COLOR_MAP.get(row.road_type or "unknown", ROAD_COLOR_MAP["unknown"])
        bar_top = _scale_y(row.effective_friction, y_min, y_max, plot_y, plot_h)
        bar_h = plot_y + plot_h - bar_top
        elements.append(_svg_rect(cx - bar_w / 2.0, bar_top, bar_w, bar_h, fill=color, rx=4.0))

        static_y = _scale_y(row.applied_static_friction, y_min, y_max, plot_y, plot_h)
        dynamic_y = _scale_y(row.applied_dynamic_friction, y_min, y_max, plot_y, plot_h)
        elements.append(_svg_line(cx, static_y, cx, dynamic_y, stroke="#222222", width=2.0))
        elements.append(_svg_circle(cx, static_y, 3.5, fill="#222222"))
        elements.append(_svg_circle(cx, dynamic_y, 3.5, fill="#6C757D"))

        elements.append(_svg_text(cx, plot_y + plot_h + 18.0, f"W{row.world_index}", size=11, anchor="middle", fill="#555555"))
        elements.append(_svg_text(cx, bar_top - 8.0, f"{row.effective_friction:.2f}", size=11, anchor="middle"))

    legend_x = x + width - 140.0
    legend_y = y + 22.0
    road_labels = list(ROAD_TYPE_ORDER) + ["unknown"]
    legend_offset = 0.0
    for road in road_labels:
        if not any((row.road_type or "unknown") == road for row in rows):
            continue
        elements.append(_svg_rect(legend_x, legend_y + legend_offset, 16.0, 10.0, fill=ROAD_COLOR_MAP.get(road, ROAD_COLOR_MAP["unknown"])))
        elements.append(_svg_text(legend_x + 22.0, legend_y + legend_offset + 10.0, road, size=11))
        legend_offset += 16.0

    elements.append(_svg_text(x + 18.0, y + height - 12.0, "bars=mu_eff, line span=applied dynamic/static", size=11, fill="#666666"))
    return elements


def _render_histogram_panel(
    values: Sequence[float],
    *,
    title: str,
    x_label: str,
    color: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    elements = _svg_frame(title, x, y, width, height)
    if not values:
        elements.append(_svg_text(x + width / 2.0, y + height / 2.0, "No data", anchor="middle"))
        return elements

    plot_x, plot_y, plot_w, plot_h = _panel_bounds(x, y, width, height)
    v_min = min(values)
    v_max = max(values)
    if v_max <= v_min:
        v_max = v_min + 1.0
    bin_count = max(4, min(12, int(math.sqrt(len(values))) + 1))
    bin_width = (v_max - v_min) / float(bin_count)
    counts = [0 for _ in range(bin_count)]
    for value in values:
        idx = min(bin_count - 1, int((value - v_min) / bin_width))
        counts[idx] += 1
    max_count = max(counts) or 1
    bar_w = plot_w / float(bin_count)

    elements.extend(
        [
            _svg_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#666666"),
            _svg_line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke="#666666"),
        ]
    )
    for idx, count in enumerate(counts):
        bx = plot_x + idx * bar_w + 4.0
        bh = plot_h * (float(count) / float(max_count))
        by = plot_y + plot_h - bh
        elements.append(_svg_rect(bx, by, max(1.0, bar_w - 8.0), bh, fill=color, rx=3.0))
        elements.append(_svg_text(bx + (bar_w - 8.0) / 2.0, by - 6.0, str(count), size=10, anchor="middle"))

    elements.append(_svg_text(plot_x, plot_y + plot_h + 18.0, f"{v_min:.3f}", size=11, fill="#555555"))
    elements.append(_svg_text(plot_x + plot_w, plot_y + plot_h + 18.0, f"{max(values):.3f}", size=11, anchor="end", fill="#555555"))
    elements.append(_svg_text(plot_x + plot_w / 2.0, y + height - 12.0, x_label, size=12, anchor="middle", fill="#666666"))
    return elements


def _render_counter_panel(
    counter: Mapping[str, int],
    *,
    title: str,
    color: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    elements = _svg_frame(title, x, y, width, height)
    if not counter:
        elements.append(_svg_text(x + width / 2.0, y + height / 2.0, "No data", anchor="middle"))
        return elements

    plot_x, plot_y, plot_w, plot_h = _panel_bounds(x, y, width, height)
    labels = list(counter.keys())
    values = list(counter.values())
    max_count = max(values) or 1
    bar_w = plot_w / float(max(1, len(labels)))

    elements.extend(
        [
            _svg_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#666666"),
            _svg_line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke="#666666"),
        ]
    )
    for idx, (label, value) in enumerate(zip(labels, values)):
        bx = plot_x + idx * bar_w + bar_w * 0.18
        bh = plot_h * (float(value) / float(max_count))
        by = plot_y + plot_h - bh
        elements.append(_svg_rect(bx, by, bar_w * 0.64, bh, fill=color, rx=4.0))
        elements.append(_svg_text(bx + bar_w * 0.32, by - 6.0, str(value), size=11, anchor="middle"))
        elements.append(_svg_text(bx + bar_w * 0.32, plot_y + plot_h + 18.0, label, size=11, anchor="middle", fill="#555555"))
    return elements


def render_dashboard(
    *,
    rows: Sequence[WorldReportRow],
    summary: Mapping[str, Any],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    width = 1800
    height = 1120
    gap = 28.0
    margin = 28.0
    header_h = 52.0
    panel_w = (width - 2.0 * margin - 2.0 * gap) / 3.0
    panel_h = (height - margin - header_h - 2.0 * gap - margin) / 2.0

    elements: list[str] = [
        _svg_text(margin, 34.0, title, size=26, weight="bold"),
        _svg_text(margin, 54.0, "Generated from the same friction inputs used by the training pipeline.", size=13, fill="#555555"),
    ]

    top_y = margin + header_h
    bottom_y = top_y + panel_h + gap
    left_x = margin
    mid_x = margin + panel_w + gap
    right_x = margin + 2.0 * (panel_w + gap)

    water = [row.water_film_mm for row in rows if row.water_film_mm is not None]
    precip = [row.precip_intensity_mmph for row in rows if row.precip_intensity_mmph is not None]

    elements.extend(_render_friction_panel(rows, left_x, top_y, panel_w, panel_h))
    elements.extend(
        _render_histogram_panel(
            [row.effective_friction for row in rows],
            title="Effective Friction Distribution",
            x_label="effective friction",
            color="#4C78A8",
            x=mid_x,
            y=top_y,
            width=panel_w,
            height=panel_h,
        )
    )
    elements.extend(
        _render_counter_panel(
            summary.get("road_type_counts", {}),
            title="Road Type Distribution",
            color="#59A14F",
            x=right_x,
            y=top_y,
            width=panel_w,
            height=panel_h,
        )
    )
    elements.extend(
        _render_counter_panel(
            summary.get("precip_type_counts", {}),
            title="Precipitation Type Distribution",
            color="#76B7B2",
            x=left_x,
            y=bottom_y,
            width=panel_w,
            height=panel_h,
        )
    )
    elements.extend(
        _render_histogram_panel(
            water,
            title="Water Film Distribution",
            x_label="water film (mm)",
            color="#F28E2B",
            x=mid_x,
            y=bottom_y,
            width=panel_w,
            height=panel_h,
        )
    )
    elements.extend(
        _render_histogram_panel(
            precip,
            title="Precipitation Intensity Distribution",
            x_label="precip intensity (mm/h)",
            color="#E15759",
            x=right_x,
            y=bottom_y,
            width=panel_w,
            height=panel_h,
        )
    )

    output_path.write_text(_svg_document(width, height, elements), encoding="utf-8")


def render_water_vs_friction_plot(
    *,
    rows: Sequence[WorldReportRow],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    width = 1000
    height = 680
    margin = 48.0
    plot_x = 82.0
    plot_y = 74.0
    plot_w = width - plot_x - margin
    plot_h = height - plot_y - 90.0

    rows_with_weather = [row for row in rows if row.water_film_mm is not None]
    elements = [
        _svg_text(margin, 34.0, title, size=24, weight="bold"),
        _svg_rect(24.0, 48.0, width - 48.0, height - 72.0, fill="#FFFFFF", stroke="#D6D3CE", stroke_width=1.0, rx=12.0),
    ]

    if not rows_with_weather:
        elements.append(_svg_text(width / 2.0, height / 2.0, "No water-film data", anchor="middle", size=18))
        output_path.write_text(_svg_document(width, height, elements), encoding="utf-8")
        return

    x_min = min(row.water_film_mm for row in rows_with_weather)
    x_max = max(row.water_film_mm for row in rows_with_weather)
    y_min = 0.0
    y_max = max(row.effective_friction for row in rows_with_weather) * 1.15
    if x_max <= x_min:
        x_max = x_min + 1.0

    elements.extend(
        [
            _svg_line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#666666"),
            _svg_line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke="#666666"),
        ]
    )
    for tick in _axis_ticks(y_min, y_max):
        tick_y = _scale_y(tick, y_min, y_max, plot_y, plot_h)
        elements.append(_svg_line(plot_x, tick_y, plot_x + plot_w, tick_y, stroke="#ECEAE5"))
        elements.append(_svg_text(plot_x - 10.0, tick_y + 4.0, f"{tick:.2f}", size=11, anchor="end", fill="#555555"))

    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = x_min + (x_max - x_min) * ratio
        tick_x = plot_x + plot_w * ratio
        elements.append(_svg_line(tick_x, plot_y, tick_x, plot_y + plot_h, stroke="#F0EEEA"))
        elements.append(_svg_text(tick_x, plot_y + plot_h + 18.0, f"{value:.3f}", size=11, anchor="middle", fill="#555555"))

    for row in rows_with_weather:
        ratio_x = (row.water_film_mm - x_min) / (x_max - x_min)
        px = plot_x + plot_w * ratio_x
        py = _scale_y(row.effective_friction, y_min, y_max, plot_y, plot_h)
        precip = row.precip_intensity_mmph or 0.0
        radius = 8.0 + min(18.0, float(precip) * 0.7)
        color = ROAD_COLOR_MAP.get(row.road_type or "unknown", ROAD_COLOR_MAP["unknown"])
        elements.append(_svg_circle(px, py, radius, fill=color, stroke="#FFFFFF", stroke_width=1.5))
        elements.append(_svg_text(px, py + 4.0, f"W{row.world_index}", size=11, anchor="middle", fill="#FFFFFF", weight="bold"))

    legend_x = width - 210.0
    legend_y = 94.0
    for idx, road in enumerate(list(ROAD_TYPE_ORDER) + ["unknown"]):
        if not any((row.road_type or "unknown") == road for row in rows_with_weather):
            continue
        y0 = legend_y + 20.0 * idx
        elements.append(_svg_circle(legend_x, y0, 7.0, fill=ROAD_COLOR_MAP.get(road, ROAD_COLOR_MAP["unknown"])))
        elements.append(_svg_text(legend_x + 14.0, y0 + 4.0, road, size=11))

    elements.append(_svg_text(plot_x + plot_w / 2.0, height - 24.0, "water film (mm)", size=13, anchor="middle", fill="#555555"))
    elements.append(_svg_text(28.0, plot_y + plot_h / 2.0, "effective friction", size=13, anchor="middle", fill="#555555"))
    elements.append(_svg_text(plot_x, height - 48.0, "Bubble size scales with precip intensity (mm/h).", size=12, fill="#666666"))
    output_path.write_text(_svg_document(width, height, elements), encoding="utf-8")


def write_curriculum_report(
    *,
    config_path: str | Path,
    output_dir: str | Path | None = None,
    write_plots: bool = True,
) -> dict[str, Path]:
    config_path = Path(config_path).expanduser()
    cfg = load_curriculum_config(config_path)
    rows = build_world_report_rows(cfg)
    summary = summarize_world_report(rows)

    if output_dir is None:
        output_root = Path("out/world_reports") / config_path.stem
    else:
        output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    report_md = output_root / "world_report.md"
    report_csv = output_root / "world_report.csv"
    report_json = output_root / "world_report.json"
    dashboard_png = output_root / "dashboard.svg"
    relationship_png = output_root / "water_vs_friction.svg"

    markdown_text = render_markdown_report(
        config_path=config_path,
        rows=rows,
        summary=summary,
        output_dir=output_root,
    )
    report_md.write_text(markdown_text, encoding="utf-8")

    with report_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].to_mapping().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_mapping())

    report_payload = {
        "config_path": str(config_path.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "rows": [row.to_mapping() for row in rows],
    }
    report_json.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    if write_plots:
        title = f"Curriculum World Report: {config_path.stem}"
        render_dashboard(
            rows=rows,
            summary=summary,
            output_path=dashboard_png,
            title=title,
        )
        render_water_vs_friction_plot(
            rows=rows,
            output_path=relationship_png,
            title=f"{title} - Water vs Friction",
        )

    return {
        "markdown": report_md,
        "csv": report_csv,
        "json": report_json,
        "dashboard": dashboard_png,
        "relationship": relationship_png,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a world friction report from a curriculum YAML.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the curriculum stage YAML.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to out/world_reports/<config_stem>.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write the tabular report only.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    outputs = write_curriculum_report(
        config_path=args.config,
        output_dir=args.out_dir,
        write_plots=not bool(args.skip_plots),
    )
    print(f"[report] wrote {outputs['markdown']}")
    print(f"[report] wrote {outputs['csv']}")
    print(f"[report] wrote {outputs['json']}")
    if not bool(args.skip_plots):
        print(f"[report] wrote {outputs['dashboard']}")
        print(f"[report] wrote {outputs['relationship']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
