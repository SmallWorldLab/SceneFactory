from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.physx_teacher_rollout_visualizer import compute_world_bounds, estimate_vehicle_footprint, load_rollout_dir


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _infer_best_trial_dir(sysid_root: Path, best_result: Dict[str, Any]) -> Path:
    output_dir = best_result.get("best_trial_output_dir", "")
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return sysid_root


def _is_staged_summary(best_result: Dict[str, Any]) -> bool:
    return isinstance(best_result.get("stages"), list) and "teacher_rollout_dir" not in best_result


def _load_staged_history(sysid_root: Path) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for history_path in sorted((sysid_root / "stages").glob("*/search_history.jsonl")):
        history.extend(_load_jsonl(history_path))
    return history


def _ordered_rollout_names(names: Sequence[str]) -> List[str]:
    priority = [
        "straight_accel_brake",
        "step_steer_left",
        "step_steer_right",
        "constant_steer_left",
        "constant_steer_right",
        "sine_steer",
        "surface_transition_s",
    ]
    seen = set()
    ordered: List[str] = []
    for name in priority:
        if name in names and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in sorted(str(value) for value in names):
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _load_report_rollout(
    *,
    rollout_name: str,
    teacher_rollout_dir: Path,
    student_output_dir: Path,
    loss: Dict[str, Any] | None,
) -> Dict[str, Any]:
    teacher_meta, teacher_frames = load_rollout_dir(teacher_rollout_dir)
    student_meta_path = student_output_dir / "student_sysid_meta.json"
    student_frames_path = student_output_dir / "student_rollout_frames.jsonl"
    if not student_meta_path.exists():
        raise FileNotFoundError(f"Missing student sysid metadata: {student_meta_path}")
    if not student_frames_path.exists():
        raise FileNotFoundError(f"Missing student rollout frames: {student_frames_path}")
    student_meta = _load_json(student_meta_path)
    student_frames = _load_jsonl(student_frames_path)
    combined_frames = list(teacher_frames) + list(student_frames)
    return {
        "name": str(rollout_name),
        "teacher_rollout_dir": str(teacher_rollout_dir),
        "student_output_dir": str(student_output_dir),
        "teacher_meta": teacher_meta,
        "teacher_frames": teacher_frames,
        "student_meta": student_meta,
        "student_frames": student_frames,
        "bounds": compute_world_bounds(teacher_meta, combined_frames),
        "footprint": estimate_vehicle_footprint(combined_frames),
        "loss": {} if loss is None else dict(loss),
    }


def _load_bundle_report_rollouts(bundle_summary_path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bundle_summary = _load_json(bundle_summary_path)
    per_rollout = bundle_summary.get("per_rollout", {})
    teacher_rollout_dirs = bundle_summary.get("teacher_rollout_dirs", {})
    if not isinstance(per_rollout, dict) or not per_rollout:
        raise ValueError(f"Bundle summary does not contain any per-rollout entries: {bundle_summary_path}")
    rollouts: List[Dict[str, Any]] = []
    for rollout_name in _ordered_rollout_names(list(per_rollout.keys())):
        entry = per_rollout.get(rollout_name, {})
        teacher_rollout_dir = Path(entry.get("teacher_rollout_dir") or teacher_rollout_dirs.get(rollout_name, "")).expanduser().resolve()
        student_output_dir = Path(entry["output_dir"]).expanduser().resolve()
        rollouts.append(
            _load_report_rollout(
                rollout_name=str(rollout_name),
                teacher_rollout_dir=teacher_rollout_dir,
                student_output_dir=student_output_dir,
                loss=entry.get("loss"),
            )
        )
    return rollouts, bundle_summary


def _find_staged_bundle_summary_path(sysid_root: Path, best_result: Dict[str, Any]) -> Path | None:
    explicit_dir = best_result.get("final_bundle_report_dir", "")
    if explicit_dir:
        explicit_path = Path(explicit_dir).expanduser().resolve() / "trial_bundle_summary.json"
        if explicit_path.exists():
            return explicit_path
    for stage_summary in reversed(best_result.get("stages", [])):
        teacher_rollout_names = stage_summary.get("teacher_rollout_names", [])
        if len(teacher_rollout_names) <= 1:
            continue
        best_trial_output_dir = stage_summary.get("best_trial_output_dir", "")
        if not best_trial_output_dir:
            continue
        candidate = Path(best_trial_output_dir).expanduser().resolve() / "trial_bundle_summary.json"
        if candidate.exists():
            return candidate
    return None


def load_sysid_dir(sysid_dir: str | Path) -> Dict[str, Any]:
    sysid_root = Path(sysid_dir).expanduser().resolve()
    best_result_path = sysid_root / "best_result.json"
    if not best_result_path.exists():
        raise FileNotFoundError(f"Missing sysid summary: {best_result_path}")
    best_result = _load_json(best_result_path)

    if _is_staged_summary(best_result):
        history = _load_staged_history(sysid_root)
        representative_dir = sysid_root / "final_report"
        representative_best_result = _load_json(representative_dir / "best_result.json")
        stages = best_result.get("stages", [])
        final_stage = stages[-1] if stages else {}
        bundle_summary_path = _find_staged_bundle_summary_path(sysid_root, best_result)
        report_rollouts: List[Dict[str, Any]]
        bundle_summary: Dict[str, Any] | None = None
        if bundle_summary_path is not None:
            report_rollouts, bundle_summary = _load_bundle_report_rollouts(bundle_summary_path)
        else:
            teacher_rollout_dir = Path(best_result["representative_rollout_dir"]).expanduser().resolve()
            report_rollouts = [
                _load_report_rollout(
                    rollout_name=str(teacher_rollout_dir.name),
                    teacher_rollout_dir=teacher_rollout_dir,
                    student_output_dir=representative_dir,
                    loss=representative_best_result.get("best_loss", {}),
                )
            ]
        display_best_result = {
            "teacher_rollout_dir": str(report_rollouts[0]["teacher_rollout_dir"]),
            "best_trial_name": str(final_stage.get("best_trial_name") or representative_best_result.get("best_trial_name") or "single"),
            "best_trial_output_dir": str(final_stage.get("best_trial_output_dir") or representative_dir),
            "best_loss": (
                bundle_summary.get("aggregate_loss", {})
                if bundle_summary is not None
                else final_stage.get("best_loss", representative_best_result.get("best_loss", {}))
            ),
            "best_config": best_result.get("best_config", representative_best_result.get("best_config", {})),
            "num_trials": len(history),
            "optimizer": best_result.get("optimizer", representative_best_result.get("optimizer", "")),
            "search_mode": best_result.get("search_mode", "staged"),
            "stages": stages,
        }
    else:
        history_path = sysid_root / "search_history.jsonl"
        history = _load_jsonl(history_path) if history_path.exists() else []
        if not history:
            best_config_path = sysid_root / "best_config.json"
            loss_path = sysid_root / "loss.json"
            if best_config_path.exists() and loss_path.exists():
                history = [
                    {
                        "trial_name": "single",
                        "output_dir": str(sysid_root),
                        "num_frames": 0,
                        "loss": _load_json(loss_path),
                        "config": _load_json(best_config_path),
                    }
                ]
        best_trial_dir = _infer_best_trial_dir(sysid_root, best_result)
        teacher_rollout_dir = Path(best_result["teacher_rollout_dir"]).expanduser().resolve()
        report_rollouts = [
            _load_report_rollout(
                rollout_name=str(teacher_rollout_dir.name),
                teacher_rollout_dir=teacher_rollout_dir,
                student_output_dir=best_trial_dir,
                loss=best_result.get("best_loss", {}),
            )
        ]
        display_best_result = best_result

    primary_rollout = report_rollouts[0]
    payload = {
        "sysid_root": str(sysid_root),
        "best_result": display_best_result,
        "history": history,
        "student_meta": primary_rollout["student_meta"],
        "student_frames": primary_rollout["student_frames"],
        "teacher_meta": primary_rollout["teacher_meta"],
        "teacher_frames": primary_rollout["teacher_frames"],
        "bounds": primary_rollout["bounds"],
        "footprint": primary_rollout["footprint"],
        "report_rollouts": report_rollouts,
        "is_staged_summary": _is_staged_summary(best_result),
        "stage_summaries": best_result.get("stages", []),
    }
    return payload


def build_sysid_report_html(
    payload: Dict[str, Any],
    *,
    title: str,
    source_sysid_dir: str,
) -> str:
    display_dt_s = float(payload["teacher_meta"].get("dt_s", 1.0 / 60.0))
    client_payload = {
        "title": str(title),
        "source_sysid_dir": str(source_sysid_dir),
        **payload,
        "display_dt_s": display_dt_s,
    }
    data_json = json.dumps(client_payload, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    html_title = html.escape(str(title))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_title}</title>
  <style>
    :root {{
      --bg: #0d1014;
      --panel: rgba(22, 27, 34, 0.92);
      --panel-2: #1b222c;
      --text: #edf2f7;
      --muted: #9aa5b1;
      --accent: #4cc38a;
      --accent-2: #5aa9ff;
      --accent-3: #f2c94c;
      --danger: #ff6b6b;
      --border: #2a3441;
      --shadow: 0 14px 32px rgba(0, 0, 0, 0.24);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(90, 169, 255, 0.08), transparent 30%),
        radial-gradient(circle at top right, rgba(76, 195, 138, 0.10), transparent 25%),
        linear-gradient(180deg, #0a0d11 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .app {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 16px;
    }}
    .title {{
      margin: 0;
      font-size: 30px;
      line-height: 1.08;
      font-weight: 650;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .card, .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .card {{
      padding: 16px;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .value {{
      margin-top: 8px;
      font-size: 24px;
      font-weight: 650;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto 1fr auto auto auto;
      gap: 12px;
      align-items: center;
      padding: 14px 16px;
      margin-bottom: 16px;
    }}
    .controls button,
    .controls select {{
      appearance: none;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
    }}
    .controls button {{
      cursor: pointer;
      min-width: 88px;
    }}
    .controls input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(760px, 1fr) 380px;
      gap: 16px;
    }}
    .main {{
      display: grid;
      gap: 16px;
    }}
    .panel {{
      padding: 16px;
    }}
    .panel h2 {{
      margin: 0 0 10px;
      font-size: 18px;
      font-weight: 620;
    }}
    canvas {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: linear-gradient(180deg, #0b0f14 0%, #11161d 100%);
      border: 1px solid rgba(255, 255, 255, 0.04);
    }}
    .world canvas {{ aspect-ratio: 16 / 10; }}
    .chart canvas {{ aspect-ratio: 16 / 4.5; }}
    .sidebar {{
      display: grid;
      gap: 16px;
      align-content: start;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .stat {{
      padding: 12px;
      border-radius: 12px;
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(255,255,255,0.04);
    }}
    .stat .name {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .stat .num {{
      font-size: 18px;
      font-weight: 620;
    }}
    .mono {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 12px;
      white-space: pre-wrap;
      line-height: 1.5;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-weight: 500;
    }}
    .chip {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.08);
      background: rgba(255,255,255,0.04);
      color: var(--muted);
      font-size: 12px;
    }}
    @media (max-width: 1100px) {{
      .summary {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .layout {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      .summary {{
        grid-template-columns: 1fr;
      }}
      .controls {{
        grid-template-columns: 1fr;
      }}
      .stats-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div>
        <h1 class="title">{html_title}</h1>
        <div class="subtitle" id="subtitle"></div>
      </div>
      <div class="chip" id="bestChip"></div>
    </div>
    <div class="summary">
      <div class="card"><div class="label">Best Total Loss</div><div class="value" id="bestLoss"></div></div>
      <div class="card"><div class="label">Trials</div><div class="value" id="trialCount"></div></div>
      <div class="card"><div class="label">Frames</div><div class="value" id="frameCount"></div></div>
      <div class="card"><div class="label">Replay Maneuver</div><div class="value" id="teacherName" style="font-size:18px"></div></div>
    </div>
    <div class="panel controls">
      <button id="playButton" type="button">Play</button>
      <input id="frameSlider" type="range" min="0" max="0" value="0" step="1">
      <select id="rolloutSelect"></select>
      <select id="speedSelect">
        <option value="0.5">0.5x</option>
        <option value="1" selected>1.0x</option>
        <option value="2">2.0x</option>
        <option value="4">4.0x</option>
      </select>
      <div class="chip" id="frameLabel"></div>
    </div>
    <div class="layout">
      <div class="main">
        <div class="panel world">
          <h2>Trajectory Match</h2>
          <canvas id="worldCanvas" width="1100" height="680"></canvas>
        </div>
        <div class="panel chart">
          <h2>Search Progress</h2>
          <canvas id="searchCanvas" width="1100" height="320"></canvas>
        </div>
        <div class="panel chart">
          <h2>Telemetry Match</h2>
          <canvas id="telemetryCanvas" width="1100" height="340"></canvas>
        </div>
      </div>
      <div class="sidebar">
        <div class="panel">
          <h2>Current Frame</h2>
          <div class="stats-grid" id="currentStats"></div>
        </div>
        <div class="panel">
          <h2>Loss Breakdown</h2>
          <table id="lossTable"></table>
        </div>
        <div class="panel">
          <h2>Top Trials</h2>
          <table id="topTrialsTable"></table>
        </div>
        <div class="panel">
          <h2>Best Config</h2>
          <div class="mono" id="configDump"></div>
        </div>
      </div>
    </div>
  </div>
  <script>
  const DATA = {data_json};

  function clamp(value, lo, hi) {{
    return Math.max(lo, Math.min(hi, value));
  }}

  function worldToCanvas(bounds, x, y, width, height, padding) {{
    const xSpan = Math.max(1e-6, bounds.x_max - bounds.x_min);
    const ySpan = Math.max(1e-6, bounds.y_max - bounds.y_min);
    const px = padding + ((x - bounds.x_min) / xSpan) * (width - 2 * padding);
    const py = height - padding - ((y - bounds.y_min) / ySpan) * (height - 2 * padding);
    return [px, py];
  }}

  function cumulativeBest(history) {{
    let best = Number.POSITIVE_INFINITY;
    return history.map((entry) => {{
      const loss = Number(entry.loss.total_loss || 0);
      best = Math.min(best, loss);
      return best;
    }});
  }}

  function formatNumber(value) {{
    const num = Number(value || 0);
    if (!Number.isFinite(num)) return 'n/a';
    if (Math.abs(num) >= 1000 || (Math.abs(num) > 0 && Math.abs(num) < 1e-3)) {{
      return num.toExponential(3);
    }}
    return num.toFixed(6);
  }}

  const history = Array.isArray(DATA.history) ? DATA.history.slice() : [];
  const historyWithIndex = history.map((entry, index) => ({{ ...entry, index }}));
  const topTrials = historyWithIndex.slice().sort((a, b) => Number(a.loss.total_loss) - Number(b.loss.total_loss)).slice(0, 8);
  const stageNames = Array.from(new Set(history.map((entry) => String(entry.stage_name || 'single'))));
  const stagePalette = ['#f2c94c', '#5aa9ff', '#4cc38a', '#ff8a65', '#c084fc', '#78dce8'];
  const stageColor = (stageName) => stagePalette[Math.max(0, stageNames.indexOf(String(stageName || 'single'))) % stagePalette.length];
  const reportRollouts = Array.isArray(DATA.report_rollouts) && DATA.report_rollouts.length
    ? DATA.report_rollouts.slice()
    : [{{
        name: 'replay',
        loss: DATA.best_result?.best_loss || {{}},
        teacher_meta: DATA.teacher_meta || {{}},
        teacher_frames: DATA.teacher_frames || [],
        student_meta: DATA.student_meta || {{}},
        student_frames: DATA.student_frames || [],
        bounds: DATA.bounds || {{}},
        footprint: DATA.footprint || {{}},
      }}];
  const bestTrace = cumulativeBest(history);
  const state = {{
    rolloutIndex: 0,
    frameIndex: 0,
    playing: false,
    playbackRate: 1.0,
    lastTickMs: 0,
  }};

  const subtitle = document.getElementById('subtitle');
  const bestChip = document.getElementById('bestChip');
  const bestLoss = document.getElementById('bestLoss');
  const trialCount = document.getElementById('trialCount');
  const frameCountEl = document.getElementById('frameCount');
  const teacherName = document.getElementById('teacherName');
  const frameLabel = document.getElementById('frameLabel');
  const frameSlider = document.getElementById('frameSlider');
  const playButton = document.getElementById('playButton');
  const rolloutSelect = document.getElementById('rolloutSelect');
  const speedSelect = document.getElementById('speedSelect');
  const worldCanvas = document.getElementById('worldCanvas');
  const searchCanvas = document.getElementById('searchCanvas');
  const telemetryCanvas = document.getElementById('telemetryCanvas');
  const currentStats = document.getElementById('currentStats');
  const lossTable = document.getElementById('lossTable');
  const topTrialsTable = document.getElementById('topTrialsTable');
  const configDump = document.getElementById('configDump');

  const best = DATA.best_result || {{}};
  subtitle.textContent = DATA.source_sysid_dir;
  bestChip.textContent = DATA.is_staged_summary
    ? `Best Final Stage: ${{best.best_trial_name || 'single'}}`
    : `Best: ${{best.best_trial_name || 'single'}}`;
  bestLoss.textContent = formatNumber(best.best_loss?.total_loss || 0);
  trialCount.textContent = String(history.length || 1);
  configDump.textContent = JSON.stringify(best.best_config || {{}}, null, 2);
  rolloutSelect.innerHTML = reportRollouts.map((rollout, index) => `<option value="${{index}}">${{rollout.name}}</option>`).join('');

  function activeRollout() {{
    return reportRollouts[Math.max(0, Math.min(state.rolloutIndex, reportRollouts.length - 1))];
  }}

  function activeTeacherFrames() {{
    return activeRollout().teacher_frames || [];
  }}

  function activeStudentFrames() {{
    return activeRollout().student_frames || [];
  }}

  function activeFrameCount() {{
    return Math.min(activeTeacherFrames().length, activeStudentFrames().length);
  }}

  function syncFrameSlider() {{
    const frameCount = activeFrameCount();
    frameSlider.max = String(Math.max(0, frameCount - 1));
    if (state.frameIndex >= frameCount) {{
      state.frameIndex = Math.max(0, frameCount - 1);
    }}
    frameSlider.value = String(state.frameIndex);
    frameCountEl.textContent = String(frameCount);
  }}

  syncFrameSlider();

  function drawWorld() {{
    const ctx = worldCanvas.getContext('2d');
    const width = worldCanvas.width;
    const height = worldCanvas.height;
    const padding = 46;
    const rollout = activeRollout();
    const teacherFrames = activeTeacherFrames();
    const studentFrames = activeStudentFrames();
    const frameCount = activeFrameCount();
    const bounds = rollout.bounds || DATA.bounds;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    for (const patch of (rollout.teacher_meta?.surface_patches || [])) {{
      const rgb = patch.color_srgb || [0.3, 0.3, 0.3];
      const x0 = patch.x_center_m - patch.length_m * 0.5;
      const x1 = patch.x_center_m + patch.length_m * 0.5;
      const y0 = patch.y_center_m - patch.width_m * 0.5;
      const y1 = patch.y_center_m + patch.width_m * 0.5;
      const p0 = worldToCanvas(bounds, x0, y0, width, height, padding);
      const p1 = worldToCanvas(bounds, x1, y1, width, height, padding);
      ctx.fillStyle = `rgba(${{Math.round(rgb[0] * 255)}}, ${{Math.round(rgb[1] * 255)}}, ${{Math.round(rgb[2] * 255)}}, 0.82)`;
      ctx.fillRect(p0[0], p1[1], p1[0] - p0[0], p0[1] - p1[1]);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.strokeRect(p0[0], p1[1], p1[0] - p0[0], p0[1] - p1[1]);
    }}

    function drawTrajectory(frames, color, widthPx) {{
      if (!frames.length) return;
      ctx.beginPath();
      frames.forEach((frame, index) => {{
        const pos = frame.vehicle.position_m;
        const p = worldToCanvas(bounds, pos[0], pos[1], width, height, padding);
        if (index === 0) ctx.moveTo(p[0], p[1]);
        else ctx.lineTo(p[0], p[1]);
      }});
      ctx.strokeStyle = color;
      ctx.lineWidth = widthPx;
      ctx.stroke();
    }}

    drawTrajectory(teacherFrames.slice(0, frameCount), 'rgba(90, 169, 255, 0.95)', 3);
    drawTrajectory(studentFrames.slice(0, frameCount), 'rgba(76, 195, 138, 0.95)', 3);

    const teacher = teacherFrames[state.frameIndex];
    const student = studentFrames[state.frameIndex];
    if (teacher) {{
      const p = worldToCanvas(bounds, teacher.vehicle.position_m[0], teacher.vehicle.position_m[1], width, height, padding);
      ctx.fillStyle = '#5aa9ff';
      ctx.beginPath();
      ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
      ctx.fill();
    }}
    if (student) {{
      const p = worldToCanvas(bounds, student.vehicle.position_m[0], student.vehicle.position_m[1], width, height, padding);
      ctx.fillStyle = '#4cc38a';
      ctx.beginPath();
      ctx.arc(p[0], p[1], 6, 0, Math.PI * 2);
      ctx.fill();
    }}

    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '12px IBM Plex Sans, sans-serif';
    ctx.fillText('Teacher', padding, 22);
    ctx.fillStyle = '#5aa9ff';
    ctx.fillRect(padding + 52, 12, 18, 4);
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.fillText('Student', padding + 82, 22);
    ctx.fillStyle = '#4cc38a';
    ctx.fillRect(padding + 140, 12, 18, 4);
  }}

  function drawSearch() {{
    const ctx = searchCanvas.getContext('2d');
    const width = searchCanvas.width;
    const height = searchCanvas.height;
    const padding = 46;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);
    if (!history.length) return;

    const actual = history.map((entry) => Number(entry.loss.total_loss || 0));
    const yMin = Math.min(...actual, ...bestTrace);
    const yMax = Math.max(...actual, ...bestTrace);
    const ySpan = Math.max(1e-9, yMax - yMin);
    const xSpan = Math.max(1, history.length - 1);
    const xFor = (index) => padding + (index / xSpan) * (width - 2 * padding);
    const yFor = (value) => height - padding - ((value - yMin) / ySpan) * (height - 2 * padding);

    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();

    ctx.beginPath();
    actual.forEach((value, index) => {{
      const x = xFor(index);
      const y = yFor(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }});
    ctx.strokeStyle = 'rgba(242, 201, 76, 0.6)';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    bestTrace.forEach((value, index) => {{
      const x = xFor(index);
      const y = yFor(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }});
    ctx.strokeStyle = '#4cc38a';
    ctx.lineWidth = 3;
    ctx.stroke();

    let previousStage = null;
    history.forEach((entry, index) => {{
      const stageName = String(entry.stage_name || 'single');
      if (index > 0 && stageName !== previousStage) {{
        const x = xFor(index) - 0.5 * (width - 2 * padding) / xSpan;
        ctx.strokeStyle = 'rgba(255,255,255,0.18)';
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.moveTo(x, padding);
        ctx.lineTo(x, height - padding);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(255,255,255,0.65)';
        ctx.fillText(stageName, clamp(x + 8, padding + 4, width - padding - 120), padding + 14);
      }}
      previousStage = stageName;
    }});

    actual.forEach((value, index) => {{
      const stageName = String(history[index].stage_name || 'single');
      ctx.fillStyle = index === (best.best_trial_name ? history.findIndex((entry) => entry.trial_name === best.best_trial_name) : -1)
        ? '#4cc38a' : stageColor(stageName);
      ctx.beginPath();
      ctx.arc(xFor(index), yFor(value), 4, 0, Math.PI * 2);
      ctx.fill();
    }});

    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '12px IBM Plex Sans, sans-serif';
    ctx.fillText(`trial`, width - padding - 24, height - 12);
    ctx.save();
    ctx.translate(14, padding + 10);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('loss', 0, 0);
    ctx.restore();
  }}

  function drawTelemetry() {{
    const ctx = telemetryCanvas.getContext('2d');
    const width = telemetryCanvas.width;
    const height = telemetryCanvas.height;
    const padding = 46;
    const teacherFrames = activeTeacherFrames();
    const studentFrames = activeStudentFrames();
    const frameCount = activeFrameCount();
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);
    if (!frameCount) return;

    const halfHeight = (height - 3 * padding) / 2;
    const xSpan = Math.max(1, frameCount - 1);
    const xFor = (index) => padding + (index / xSpan) * (width - 2 * padding);

    const teacherSpeed = teacherFrames.slice(0, frameCount).map((frame) => Math.hypot(frame.vehicle.linear_velocity_mps[0], frame.vehicle.linear_velocity_mps[1]));
    const studentSpeed = studentFrames.slice(0, frameCount).map((frame) => Math.hypot(frame.vehicle.linear_velocity_mps[0], frame.vehicle.linear_velocity_mps[1]));
    const teacherYaw = teacherFrames.slice(0, frameCount).map((frame) => Number(frame.vehicle.yaw_rad || 0));
    const studentYaw = studentFrames.slice(0, frameCount).map((frame) => Number(frame.vehicle.yaw_rad || 0));

    function drawSeries(title, yOffset, valuesA, valuesB, colorA, colorB) {{
      const yMin = Math.min(...valuesA, ...valuesB);
      const yMax = Math.max(...valuesA, ...valuesB);
      const ySpan = Math.max(1e-6, yMax - yMin);
      const yFor = (value) => yOffset + halfHeight - ((value - yMin) / ySpan) * halfHeight;

      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.beginPath();
      ctx.moveTo(padding, yOffset + halfHeight);
      ctx.lineTo(width - padding, yOffset + halfHeight);
      ctx.moveTo(padding, yOffset);
      ctx.lineTo(padding, yOffset + halfHeight);
      ctx.stroke();

      [ [valuesA, colorA], [valuesB, colorB] ].forEach(([values, color]) => {{
        ctx.beginPath();
        values.forEach((value, index) => {{
          const x = xFor(index);
          const y = yFor(value);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }});

      ctx.fillStyle = 'rgba(255,255,255,0.7)';
      ctx.font = '12px IBM Plex Sans, sans-serif';
      ctx.fillText(title, padding, yOffset - 10);
    }}

    drawSeries('speed (m/s)', padding, teacherSpeed, studentSpeed, '#5aa9ff', '#4cc38a');
    drawSeries('yaw (rad)', padding * 2 + halfHeight, teacherYaw, studentYaw, '#5aa9ff', '#4cc38a');

    const x = xFor(state.frameIndex);
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, padding - 8);
    ctx.lineTo(x, height - padding + 8);
    ctx.stroke();
  }}

  function updateTables() {{
    const rollout = activeRollout();
    const teacherFrames = activeTeacherFrames();
    const studentFrames = activeStudentFrames();
    const teacher = teacherFrames[state.frameIndex];
    const student = studentFrames[state.frameIndex];
    if (!teacher || !student) return;
    const tPos = teacher.vehicle.position_m;
    const sPos = student.vehicle.position_m;
    const dx = sPos[0] - tPos[0];
    const dy = sPos[1] - tPos[1];
    const teacherSpeed = Math.hypot(teacher.vehicle.linear_velocity_mps[0], teacher.vehicle.linear_velocity_mps[1]);
    const studentSpeed = Math.hypot(student.vehicle.linear_velocity_mps[0], student.vehicle.linear_velocity_mps[1]);

    const stats = [
      ['rollout', String(rollout.name || 'replay')],
      ['sim time', `${{teacher.sim_time_s.toFixed(3)}} s`],
      ['pos error', `${{Math.hypot(dx, dy).toFixed(5)}} m`],
      ['teacher speed', `${{teacherSpeed.toFixed(5)}} m/s`],
      ['student speed', `${{studentSpeed.toFixed(5)}} m/s`],
      ['steering cmd', formatNumber(teacher.command.steering)],
      ['accel cmd', formatNumber(teacher.command.accelerator)],
      ['brake cmd', formatNumber(teacher.command.brake)],
      ['yaw error', formatNumber(student.vehicle.yaw_rad - teacher.vehicle.yaw_rad)],
    ];
    currentStats.innerHTML = stats.map(([name, num]) => `<div class="stat"><div class="name">${{name}}</div><div class="num">${{num}}</div></div>`).join('');

    const loss = rollout.loss || best.best_loss || {{}};
    const rows = [
      ['total_loss', loss.total_loss],
      ['position_xy_mse', loss.position_xy_mse],
      ['yaw_mse', loss.yaw_mse],
      ['speed_mse', loss.speed_mse],
      ['yaw_rate_mse', loss.yaw_rate_mse],
      ['wheel_speed_mse', loss.wheel_speed_mse],
      ['steer_angle_mse', loss.steer_angle_mse],
      ['suspension_mse', loss.suspension_mse],
    ];
    lossTable.innerHTML = '<tr><th>Metric</th><th>Value</th></tr>' + rows.map(([name, value]) => `<tr><td>${{name}}</td><td>${{formatNumber(value)}}</td></tr>`).join('');

    topTrialsTable.innerHTML = '<tr><th>Trial</th><th>Stage</th><th>Loss</th></tr>' + topTrials.map((entry) => `<tr><td>${{entry.trial_name}}</td><td>${{String(entry.stage_name || 'single')}}</td><td>${{formatNumber(entry.loss.total_loss)}}</td></tr>`).join('');
  }}

  function render() {{
    const rollout = activeRollout();
    const teacherMeta = rollout.teacher_meta || {{}};
    const frameCount = activeFrameCount();
    syncFrameSlider();
    teacherName.textContent = String(rollout.name || (teacherMeta.command_program_source || '').split('/').slice(-1)[0] || 'teacher');
    frameLabel.textContent = frameCount ? `frame ${{state.frameIndex + 1}} / ${{frameCount}}` : 'no frames';
    drawWorld();
    drawSearch();
    drawTelemetry();
    updateTables();
  }}

  function stepPlayback(timestampMs) {{
    const frameCount = activeFrameCount();
    if (!state.playing || frameCount <= 1) {{
      state.lastTickMs = timestampMs;
      return;
    }}
    if (!state.lastTickMs) {{
      state.lastTickMs = timestampMs;
    }}
    const dtMs = timestampMs - state.lastTickMs;
    const frameDurationMs = 1000 * DATA.display_dt_s / state.playbackRate;
    if (dtMs >= frameDurationMs) {{
      const advance = Math.max(1, Math.floor(dtMs / frameDurationMs));
      state.frameIndex = (state.frameIndex + advance) % frameCount;
      frameSlider.value = String(state.frameIndex);
      state.lastTickMs = timestampMs;
      render();
    }}
  }}

  function animate(timestampMs) {{
    stepPlayback(timestampMs);
    requestAnimationFrame(animate);
  }}

  playButton.addEventListener('click', () => {{
    state.playing = !state.playing;
    playButton.textContent = state.playing ? 'Pause' : 'Play';
    state.lastTickMs = 0;
  }});
  frameSlider.addEventListener('input', (event) => {{
    state.frameIndex = clamp(Number(event.target.value), 0, Math.max(0, activeFrameCount() - 1));
    render();
  }});
  rolloutSelect.addEventListener('change', (event) => {{
    state.rolloutIndex = clamp(Number(event.target.value), 0, Math.max(0, reportRollouts.length - 1));
    state.frameIndex = 0;
    state.lastTickMs = 0;
    syncFrameSlider();
    render();
  }});
  speedSelect.addEventListener('change', (event) => {{
    state.playbackRate = Number(event.target.value) || 1.0;
    state.lastTickMs = 0;
  }});

  render();
  requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


def write_sysid_report(
    sysid_dir: str | Path,
    *,
    output_html: str | Path | None = None,
    title: str = "Student Vehicle SysId Report",
) -> Path:
    sysid_root = Path(sysid_dir).expanduser().resolve()
    payload = load_sysid_dir(sysid_root)
    html_text = build_sysid_report_html(payload, title=title, source_sysid_dir=str(sysid_root))
    output_path = (
        Path(output_html).expanduser().resolve()
        if output_html is not None
        else sysid_root / "sysid_report.html"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an HTML report for a student vehicle sysid run.")
    parser.add_argument("--sysid-dir", type=str, required=True)
    parser.add_argument("--output-html", type=str, default="")
    parser.add_argument("--title", type=str, default="Student Vehicle SysId Report")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_path = write_sysid_report(
        args.sysid_dir,
        output_html=args.output_html if str(args.output_html) else None,
        title=args.title,
    )
    print(f"[student_vehicle_sysid_visualizer] wrote report to {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
