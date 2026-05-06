from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def load_rollout_dir(rollout_dir: str | Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rollout_root = Path(rollout_dir).expanduser().resolve()
    meta_path = rollout_root / "rollout_meta.json"
    frames_path = rollout_root / "rollout_frames.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing rollout metadata: {meta_path}")
    if not frames_path.exists():
        raise FileNotFoundError(f"Missing rollout frames: {frames_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames = [
        json.loads(line)
        for line in frames_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not frames:
        raise ValueError(f"Rollout has no frames: {frames_path}")
    return meta, frames


def downsample_frames(frames: Sequence[Dict[str, Any]], stride: int) -> List[Dict[str, Any]]:
    stride_value = max(1, int(stride))
    return list(frames[::stride_value])


def compute_world_bounds(
    meta: Dict[str, Any],
    frames: Sequence[Dict[str, Any]],
    *,
    margin_m: float = 1.0,
) -> Dict[str, float]:
    x_values: List[float] = []
    y_values: List[float] = []

    for patch in meta.get("surface_patches", []):
        center_x = float(patch["x_center_m"])
        center_y = float(patch["y_center_m"])
        half_length = 0.5 * float(patch["length_m"])
        half_width = 0.5 * float(patch["width_m"])
        x_values.extend([center_x - half_length, center_x + half_length])
        y_values.extend([center_y - half_width, center_y + half_width])

    for frame in frames:
        vehicle = frame.get("vehicle", {})
        position = vehicle.get("position_m", [0.0, 0.0, 0.0])
        x_values.append(float(position[0]))
        y_values.append(float(position[1]))
        for wheel in frame.get("wheels", []):
            hit = wheel.get("ground_hit_position_m", None)
            if hit is None:
                continue
            x_values.append(float(hit[0]))
            y_values.append(float(hit[1]))

    if not x_values:
        x_values = [-10.0, 10.0]
    if not y_values:
        y_values = [-10.0, 10.0]

    margin_value = float(margin_m)
    x_min = min(x_values) - margin_value
    x_max = max(x_values) + margin_value
    y_min = min(y_values) - margin_value
    y_max = max(y_values) + margin_value

    if math.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
    }


def estimate_vehicle_footprint(
    frames: Sequence[Dict[str, Any]],
    *,
    default_length_m: float = 2.8,
    default_width_m: float = 1.8,
) -> Dict[str, float]:
    length_m = float(default_length_m)
    width_m = float(default_width_m)

    for frame in frames:
        hit_points = [
            wheel["ground_hit_position_m"]
            for wheel in frame.get("wheels", [])
            if wheel.get("ground_hit_position_m") is not None
        ]
        if len(hit_points) < 2:
            continue

        xs = [float(hit[0]) for hit in hit_points]
        ys = [float(hit[1]) for hit in hit_points]
        length_m = max(length_m, max(xs) - min(xs) + 0.6)
        width_m = max(width_m, max(ys) - min(ys) + 0.4)
        break

    return {
        "length_m": float(length_m),
        "width_m": float(width_m),
    }


def build_replay_html(
    meta: Dict[str, Any],
    frames: Sequence[Dict[str, Any]],
    *,
    title: str,
    source_rollout_dir: str,
    frame_stride: int,
) -> str:
    display_dt_s = float(meta.get("dt_s", 1.0 / 60.0)) * max(1, int(frame_stride))
    payload = {
        "title": str(title),
        "source_rollout_dir": str(source_rollout_dir),
        "meta": meta,
        "frames": list(frames),
        "bounds": compute_world_bounds(meta, frames),
        "footprint": estimate_vehicle_footprint(frames),
        "display_dt_s": float(display_dt_s),
    }
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")

    html_title = html.escape(str(title))
    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #0f1115;
      --panel: #171a20;
      --panel-2: #1f242c;
      --text: #edf2f7;
      --muted: #98a2b3;
      --accent: #3ddc97;
      --accent-2: #58a6ff;
      --border: #2d3642;
      --danger: #ff6b6b;
      --warning: #f2c94c;
      --shadow: 0 12px 24px rgba(0, 0, 0, 0.22);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(88, 166, 255, 0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(61, 220, 151, 0.09), transparent 24%),
        linear-gradient(180deg, #0b0d11 0%, var(--bg) 100%);
    }
    .app {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 16px;
    }
    .title {
      margin: 0;
      font-size: 28px;
      line-height: 1.1;
      font-weight: 650;
      letter-spacing: -0.02em;
    }
    .subtitle {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .controls {
      display: grid;
      grid-template-columns: auto 1fr auto auto;
      gap: 12px;
      align-items: center;
      background: rgba(23, 26, 32, 0.9);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 16px;
      margin-bottom: 16px;
      box-shadow: var(--shadow);
    }
    .controls button,
    .controls select {
      appearance: none;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
    }
    .controls button {
      cursor: pointer;
      min-width: 88px;
    }
    .controls input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(720px, 1fr) 360px;
      gap: 16px;
    }
    .panel {
      background: rgba(23, 26, 32, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      box-shadow: var(--shadow);
    }
    canvas {
      display: block;
      width: 100%;
      border-radius: 12px;
      background: linear-gradient(180deg, #0b0e13 0%, #10141c 100%);
      border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .world-panel canvas {
      aspect-ratio: 16 / 10;
    }
    .timeline-panel {
      margin-top: 16px;
    }
    .timeline-panel canvas {
      aspect-ratio: 16 / 4;
    }
    .sidebar {
      display: grid;
      gap: 16px;
      align-content: start;
    }
    .stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .stat {
      padding: 12px;
      border-radius: 12px;
      background: var(--panel-2);
      border: 1px solid rgba(255, 255, 255, 0.04);
    }
    .stat-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .stat-value {
      font-size: 18px;
      font-weight: 650;
    }
    .meter-stack {
      display: grid;
      gap: 10px;
    }
    .meter-row {
      display: grid;
      grid-template-columns: 92px 1fr 56px;
      gap: 10px;
      align-items: center;
      font-size: 14px;
    }
    .meter-track {
      position: relative;
      height: 10px;
      border-radius: 999px;
      background: #0f1319;
      border: 1px solid rgba(255,255,255,0.05);
      overflow: hidden;
    }
    .meter-fill {
      position: absolute;
      top: 0;
      bottom: 0;
      left: 0;
      border-radius: 999px;
    }
    .meter-fill.accel { background: linear-gradient(90deg, #1ecb74, #54f3a8); }
    .meter-fill.brake { background: linear-gradient(90deg, #ff7a7a, #ffb199); }
    .meter-center {
      position: absolute;
      left: 50%;
      top: -3px;
      bottom: -3px;
      width: 1px;
      background: rgba(255,255,255,0.15);
    }
    .meter-pointer {
      position: absolute;
      top: -2px;
      width: 10px;
      height: 12px;
      margin-left: -5px;
      border-radius: 6px;
      background: var(--accent-2);
      box-shadow: 0 0 0 1px rgba(255,255,255,0.16);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 500;
    }
    td:last-child, th:last-child {
      text-align: right;
    }
    .surface-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .surface-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.12);
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
    }
    @media (max-width: 1100px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .controls {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div>
        <h1 class="title">__TITLE__</h1>
        <div class="subtitle" id="subtitle"></div>
      </div>
    </div>

    <div class="controls">
      <button id="playPauseButton" type="button">Play</button>
      <input id="frameSlider" type="range" min="0" max="0" value="0" step="1">
      <select id="speedSelect">
        <option value="0.25">0.25x</option>
        <option value="0.5">0.5x</option>
        <option value="1" selected>1.0x</option>
        <option value="2">2.0x</option>
        <option value="4">4.0x</option>
      </select>
      <div id="frameLabel">Frame 0 / 0</div>
    </div>

    <div class="layout">
      <div>
        <div class="panel world-panel">
          <canvas id="worldCanvas" width="1120" height="700"></canvas>
        </div>
        <div class="panel timeline-panel">
          <canvas id="timelineCanvas" width="1120" height="240"></canvas>
        </div>
      </div>

      <div class="sidebar">
        <div class="panel">
          <div class="stats">
            <div class="stat">
              <div class="stat-label">Time</div>
              <div class="stat-value" id="timeValue">0.00 s</div>
            </div>
            <div class="stat">
              <div class="stat-label">Speed</div>
              <div class="stat-value" id="speedValue">0.00 m/s</div>
            </div>
            <div class="stat">
              <div class="stat-label">Position</div>
              <div class="stat-value" id="positionValue">0.00, 0.00</div>
            </div>
            <div class="stat">
              <div class="stat-label">Yaw</div>
              <div class="stat-value" id="yawValue">0.0 deg</div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="stat-label">Command</div>
          <div class="meter-stack">
            <div class="meter-row">
              <div>Accelerator</div>
              <div class="meter-track"><div id="accelFill" class="meter-fill accel"></div></div>
              <div id="accelValue">0.00</div>
            </div>
            <div class="meter-row">
              <div>Brake</div>
              <div class="meter-track"><div id="brakeFill" class="meter-fill brake"></div></div>
              <div id="brakeValue">0.00</div>
            </div>
            <div class="meter-row">
              <div>Steering</div>
              <div class="meter-track">
                <div class="meter-center"></div>
                <div id="steerPointer" class="meter-pointer"></div>
              </div>
              <div id="steerValue">0.00</div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="stat-label">Surface Legend</div>
          <div id="surfaceLegend" class="legend"></div>
        </div>

        <div class="panel">
          <div class="stat-label">Wheel Contacts</div>
          <table>
            <thead>
              <tr>
                <th>Wheel</th>
                <th>Surface</th>
                <th>Slip</th>
              </tr>
            </thead>
            <tbody id="wheelTableBody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script id="replay-data" type="application/json">__DATA__</script>
  <script>
    const replayData = JSON.parse(document.getElementById("replay-data").textContent);
    const meta = replayData.meta;
    const frames = replayData.frames;
    const bounds = replayData.bounds;
    const footprint = replayData.footprint;
    const displayDtS = replayData.display_dt_s;

    const frameSlider = document.getElementById("frameSlider");
    const playPauseButton = document.getElementById("playPauseButton");
    const speedSelect = document.getElementById("speedSelect");
    const frameLabel = document.getElementById("frameLabel");
    const subtitle = document.getElementById("subtitle");

    const worldCanvas = document.getElementById("worldCanvas");
    const worldCtx = worldCanvas.getContext("2d");
    const timelineCanvas = document.getElementById("timelineCanvas");
    const timelineCtx = timelineCanvas.getContext("2d");

    const surfaceByName = new Map((meta.surface_patches || []).map((patch) => [patch.name, patch]));
    const surfaceColorCss = new Map(
      (meta.surface_patches || []).map((patch) => [patch.name, srgbToCss(patch.color_srgb || [0.6, 0.6, 0.6])])
    );

    const state = {
      frameIdx: 0,
      playing: false,
      speed: 1.0,
      lastTimestampMs: null,
      accumulatorMs: 0.0,
    };

    frameSlider.max = String(Math.max(0, frames.length - 1));
    subtitle.textContent = `${replayData.source_rollout_dir} | ${frames.length} frames | dt=${displayDtS.toFixed(4)} s`;

    function srgbToCss(rgb) {
      const [r, g, b] = rgb || [0.6, 0.6, 0.6];
      return `rgb(${Math.round(255 * r)}, ${Math.round(255 * g)}, ${Math.round(255 * b)})`;
    }

    function clamp(value, lo, hi) {
      return Math.max(lo, Math.min(hi, value));
    }

    function formatSigned(value, digits = 2) {
      const number = Number(value || 0);
      return `${number >= 0 ? "" : "-"}${Math.abs(number).toFixed(digits)}`;
    }

    function metersPerPixelTransform(canvas) {
      const pad = 56;
      const dataWidth = Math.max(1e-6, bounds.x_max - bounds.x_min);
      const dataHeight = Math.max(1e-6, bounds.y_max - bounds.y_min);
      const usableWidth = canvas.width - 2 * pad;
      const usableHeight = canvas.height - 2 * pad;
      const scale = Math.min(usableWidth / dataWidth, usableHeight / dataHeight);
      const viewWidth = dataWidth * scale;
      const viewHeight = dataHeight * scale;
      const xOffset = 0.5 * (canvas.width - viewWidth);
      const yOffset = 0.5 * (canvas.height - viewHeight);
      return { pad, scale, xOffset, yOffset };
    }

    function worldToCanvas(canvas, x, y) {
      const transform = metersPerPixelTransform(canvas);
      const screenX = transform.xOffset + (x - bounds.x_min) * transform.scale;
      const screenY = canvas.height - transform.yOffset - (y - bounds.y_min) * transform.scale;
      return [screenX, screenY];
    }

    function frameSpeedMps(frame) {
      const velocity = (((frame || {}).vehicle || {}).linear_velocity_mps) || [0, 0, 0];
      return Math.hypot(Number(velocity[0] || 0), Number(velocity[1] || 0), Number(velocity[2] || 0));
    }

    function drawWorld(frame) {
      worldCtx.clearRect(0, 0, worldCanvas.width, worldCanvas.height);
      worldCtx.fillStyle = "#0d1218";
      worldCtx.fillRect(0, 0, worldCanvas.width, worldCanvas.height);

      for (const patch of meta.surface_patches || []) {
        const x0 = patch.x_center_m - 0.5 * patch.length_m;
        const x1 = patch.x_center_m + 0.5 * patch.length_m;
        const y0 = patch.y_center_m - 0.5 * patch.width_m;
        const y1 = patch.y_center_m + 0.5 * patch.width_m;
        const [sx0, sy0] = worldToCanvas(worldCanvas, x0, y1);
        const [sx1, sy1] = worldToCanvas(worldCanvas, x1, y0);
        const left = Math.min(sx0, sx1);
        const top = Math.min(sy0, sy1);
        const width = Math.abs(sx1 - sx0);
        const height = Math.abs(sy1 - sy0);
        worldCtx.fillStyle = srgbToCss(patch.color_srgb || [0.5, 0.5, 0.5]);
        worldCtx.fillRect(left, top, width, height);
        worldCtx.strokeStyle = "rgba(255,255,255,0.16)";
        worldCtx.lineWidth = 1;
        worldCtx.strokeRect(left, top, width, height);
        worldCtx.fillStyle = "rgba(255,255,255,0.82)";
        worldCtx.font = "14px IBM Plex Sans, sans-serif";
        worldCtx.fillText(patch.name, left + 10, top + 22);
      }

      worldCtx.lineWidth = 3;
      worldCtx.strokeStyle = "rgba(88,166,255,0.35)";
      worldCtx.beginPath();
      frames.forEach((item, idx) => {
        const position = item.vehicle.position_m;
        const [sx, sy] = worldToCanvas(worldCanvas, position[0], position[1]);
        if (idx === 0) {
          worldCtx.moveTo(sx, sy);
        } else {
          worldCtx.lineTo(sx, sy);
        }
      });
      worldCtx.stroke();

      worldCtx.lineWidth = 4;
      worldCtx.strokeStyle = "rgba(61,220,151,0.92)";
      worldCtx.beginPath();
      for (let idx = 0; idx <= state.frameIdx; idx += 1) {
        const position = frames[idx].vehicle.position_m;
        const [sx, sy] = worldToCanvas(worldCanvas, position[0], position[1]);
        if (idx === 0) {
          worldCtx.moveTo(sx, sy);
        } else {
          worldCtx.lineTo(sx, sy);
        }
      }
      worldCtx.stroke();

      for (const wheel of frame.wheels || []) {
        if (!wheel.ground_hit_position_m) {
          continue;
        }
        const [sx, sy] = worldToCanvas(worldCanvas, wheel.ground_hit_position_m[0], wheel.ground_hit_position_m[1]);
        worldCtx.beginPath();
        worldCtx.arc(sx, sy, 6, 0, 2 * Math.PI);
        worldCtx.fillStyle = surfaceColorCss.get(wheel.surface_name) || "#d6dbe4";
        worldCtx.fill();
        worldCtx.strokeStyle = "rgba(10, 12, 16, 0.95)";
        worldCtx.lineWidth = 2;
        worldCtx.stroke();
      }

      const position = frame.vehicle.position_m;
      const yaw = Number(frame.vehicle.yaw_rad || 0);
      const halfLength = 0.5 * Number(footprint.length_m || 2.8);
      const halfWidth = 0.5 * Number(footprint.width_m || 1.8);
      const corners = [
        [halfLength, halfWidth],
        [halfLength, -halfWidth],
        [-halfLength, -halfWidth],
        [-halfLength, halfWidth],
      ].map(([lx, ly]) => {
        const x = position[0] + Math.cos(yaw) * lx - Math.sin(yaw) * ly;
        const y = position[1] + Math.sin(yaw) * lx + Math.cos(yaw) * ly;
        return worldToCanvas(worldCanvas, x, y);
      });

      worldCtx.beginPath();
      worldCtx.moveTo(corners[0][0], corners[0][1]);
      for (let idx = 1; idx < corners.length; idx += 1) {
        worldCtx.lineTo(corners[idx][0], corners[idx][1]);
      }
      worldCtx.closePath();
      worldCtx.fillStyle = "rgba(255, 255, 255, 0.12)";
      worldCtx.fill();
      worldCtx.strokeStyle = "#ffffff";
      worldCtx.lineWidth = 2;
      worldCtx.stroke();

      const nose = worldToCanvas(
        worldCanvas,
        position[0] + Math.cos(yaw) * (halfLength + 0.7),
        position[1] + Math.sin(yaw) * (halfLength + 0.7)
      );
      const center = worldToCanvas(worldCanvas, position[0], position[1]);
      worldCtx.beginPath();
      worldCtx.moveTo(center[0], center[1]);
      worldCtx.lineTo(nose[0], nose[1]);
      worldCtx.strokeStyle = "#fffbcc";
      worldCtx.lineWidth = 3;
      worldCtx.stroke();
    }

    function drawTimeline() {
      timelineCtx.clearRect(0, 0, timelineCanvas.width, timelineCanvas.height);
      timelineCtx.fillStyle = "#0d1218";
      timelineCtx.fillRect(0, 0, timelineCanvas.width, timelineCanvas.height);

      const padLeft = 64;
      const padRight = 20;
      const padTop = 24;
      const padBottom = 30;
      const width = timelineCanvas.width - padLeft - padRight;
      const height = timelineCanvas.height - padTop - padBottom;
      const maxSpeed = Math.max(0.5, ...frames.map(frameSpeedMps));

      timelineCtx.strokeStyle = "rgba(255,255,255,0.08)";
      timelineCtx.lineWidth = 1;
      for (let idx = 0; idx <= 4; idx += 1) {
        const y = padTop + (height * idx) / 4;
        timelineCtx.beginPath();
        timelineCtx.moveTo(padLeft, y);
        timelineCtx.lineTo(timelineCanvas.width - padRight, y);
        timelineCtx.stroke();
      }

      timelineCtx.strokeStyle = "rgba(61,220,151,0.95)";
      timelineCtx.lineWidth = 2;
      timelineCtx.beginPath();
      frames.forEach((frame, idx) => {
        const x = padLeft + (width * idx) / Math.max(1, frames.length - 1);
        const y = padTop + height - (height * frameSpeedMps(frame)) / maxSpeed;
        if (idx === 0) {
          timelineCtx.moveTo(x, y);
        } else {
          timelineCtx.lineTo(x, y);
        }
      });
      timelineCtx.stroke();

      timelineCtx.fillStyle = "rgba(255,255,255,0.85)";
      timelineCtx.font = "13px IBM Plex Sans, sans-serif";
      timelineCtx.fillText("Speed (m/s)", 12, 18);
      timelineCtx.fillText(maxSpeed.toFixed(2), 10, padTop + 6);
      timelineCtx.fillText("0.00", 16, padTop + height + 4);

      const cursorX = padLeft + (width * state.frameIdx) / Math.max(1, frames.length - 1);
      timelineCtx.strokeStyle = "rgba(88,166,255,0.95)";
      timelineCtx.lineWidth = 2;
      timelineCtx.beginPath();
      timelineCtx.moveTo(cursorX, padTop);
      timelineCtx.lineTo(cursorX, padTop + height);
      timelineCtx.stroke();
    }

    function updateSidebar(frame) {
      const position = frame.vehicle.position_m || [0, 0, 0];
      const speedMps = frameSpeedMps(frame);
      const yawDeg = (Number(frame.vehicle.yaw_rad || 0) * 180.0) / Math.PI;
      document.getElementById("timeValue").textContent = `${Number(frame.sim_time_s || 0).toFixed(2)} s`;
      document.getElementById("speedValue").textContent = `${speedMps.toFixed(3)} m/s`;
      document.getElementById("positionValue").textContent = `${position[0].toFixed(2)}, ${position[1].toFixed(2)}`;
      document.getElementById("yawValue").textContent = `${yawDeg.toFixed(1)} deg`;

      const command = frame.command || {};
      const accel = clamp(Number(command.accelerator || 0), 0, 1);
      const brake = clamp(Number(command.brake || 0), 0, 1);
      const steer = clamp(Number(command.steering || 0), -1, 1);
      document.getElementById("accelFill").style.width = `${100 * accel}%`;
      document.getElementById("brakeFill").style.width = `${100 * brake}%`;
      document.getElementById("steerPointer").style.left = `${50 + 50 * steer}%`;
      document.getElementById("accelValue").textContent = accel.toFixed(2);
      document.getElementById("brakeValue").textContent = brake.toFixed(2);
      document.getElementById("steerValue").textContent = formatSigned(steer, 2);

      const wheelRows = (frame.wheels || []).map((wheel) => {
        const surfaceName = wheel.surface_name || "none";
        const color = surfaceColorCss.get(surfaceName) || "#d7dce4";
        const slip = `${Number(wheel.tire_longitudinal_slip || 0).toFixed(3)} / ${Number(wheel.tire_lateral_slip || 0).toFixed(3)}`;
        return `
          <tr>
            <td>${wheel.label}</td>
            <td><span class="surface-chip"><span class="surface-dot" style="background:${color}"></span>${surfaceName}</span></td>
            <td>${slip}</td>
          </tr>
        `;
      }).join("");
      document.getElementById("wheelTableBody").innerHTML = wheelRows;
    }

    function updateFrameLabel() {
      frameLabel.textContent = `Frame ${state.frameIdx + 1} / ${frames.length}`;
      frameSlider.value = String(state.frameIdx);
    }

    function render() {
      const frame = frames[state.frameIdx];
      drawWorld(frame);
      drawTimeline();
      updateSidebar(frame);
      updateFrameLabel();
    }

    function animate(timestampMs) {
      if (!state.playing) {
        state.lastTimestampMs = null;
        state.accumulatorMs = 0.0;
        return;
      }

      if (state.lastTimestampMs === null) {
        state.lastTimestampMs = timestampMs;
      }
      const elapsedMs = timestampMs - state.lastTimestampMs;
      state.lastTimestampMs = timestampMs;
      state.accumulatorMs += elapsedMs * state.speed;

      const stepMs = Math.max(1.0, 1000.0 * displayDtS);
      while (state.accumulatorMs >= stepMs && state.frameIdx < frames.length - 1) {
        state.frameIdx += 1;
        state.accumulatorMs -= stepMs;
      }

      if (state.frameIdx >= frames.length - 1) {
        state.playing = false;
        playPauseButton.textContent = "Play";
      }

      render();
      window.requestAnimationFrame(animate);
    }

    playPauseButton.addEventListener("click", () => {
      state.playing = !state.playing;
      playPauseButton.textContent = state.playing ? "Pause" : "Play";
      if (state.playing) {
        window.requestAnimationFrame(animate);
      }
    });

    frameSlider.addEventListener("input", () => {
      state.frameIdx = Number(frameSlider.value);
      state.playing = false;
      playPauseButton.textContent = "Play";
      render();
    });

    speedSelect.addEventListener("change", () => {
      state.speed = Number(speedSelect.value || 1.0);
    });

    document.addEventListener("keydown", (event) => {
      if (event.code === "Space") {
        event.preventDefault();
        playPauseButton.click();
      }
      if (event.code === "ArrowRight") {
        state.frameIdx = Math.min(frames.length - 1, state.frameIdx + 1);
        state.playing = false;
        playPauseButton.textContent = "Play";
        render();
      }
      if (event.code === "ArrowLeft") {
        state.frameIdx = Math.max(0, state.frameIdx - 1);
        state.playing = false;
        playPauseButton.textContent = "Play";
        render();
      }
    });

    document.getElementById("surfaceLegend").innerHTML = (meta.surface_patches || []).map((patch) => {
      return `<span class="surface-chip"><span class="surface-dot" style="background:${srgbToCss(patch.color_srgb)}"></span>${patch.name}</span>`;
    }).join("");

    render();
  </script>
</body>
</html>
"""
    return template.replace("__TITLE__", html_title).replace("__DATA__", data_json)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML replay for a recorded PhysX teacher rollout."
    )
    parser.add_argument("--rollout-dir", type=str, required=True, help="Directory containing rollout_meta.json and rollout_frames.jsonl")
    parser.add_argument("--output-html", type=str, default="", help="Output HTML path. Defaults to <rollout-dir>/replay.html")
    parser.add_argument("--title", type=str, default="", help="Optional page title override.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Optional frame downsampling factor for large rollouts.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rollout_dir = Path(args.rollout_dir).expanduser().resolve()
    meta, frames = load_rollout_dir(rollout_dir)
    frame_stride = max(1, int(args.frame_stride))
    frames_for_display = downsample_frames(frames, frame_stride)
    title = args.title or f"PhysX Teacher Replay: {rollout_dir.name}"
    output_html = (
        Path(args.output_html).expanduser().resolve()
        if args.output_html
        else rollout_dir / "replay.html"
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    html_text = build_replay_html(
        meta,
        frames_for_display,
        title=title,
        source_rollout_dir=str(rollout_dir),
        frame_stride=frame_stride,
    )
    output_html.write_text(html_text, encoding="utf-8")
    print(f"[physx_teacher_rollout_visualizer] wrote replay HTML to {output_html}")


if __name__ == "__main__":
    main()
