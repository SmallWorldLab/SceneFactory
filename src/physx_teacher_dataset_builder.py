from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Sequence

from src.physx_teacher_command_program_generator import build_program_from_preset, write_command_program
from src.physx_teacher_patch_track import build_default_surface_patches
from src.physx_teacher_rollout_visualizer import build_replay_html, load_rollout_dir


@dataclass(frozen=True)
class ManeuverSpec:
    name: str
    preset: str
    params: Dict[str, Any]
    description: str

    def to_manifest_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "preset": str(self.preset),
            "params": dict(self.params),
            "description": str(self.description),
        }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_comprehensive_fwd_suite() -> List[ManeuverSpec]:
    """Build a comprehensive sysid dataset covering the full operating envelope.

    Tiers:
      1. Longitudinal – throttle sweep (0.1 to 1.0), brake sweep, ramp-throttle
      2. Lateral – step-steer and constant-steer at multiple throttles × steer values
      3. Combined – trail-brake, chirp-steer at multiple speeds
      4. Frequency response – sine-steer at multiple frequencies and amplitudes
      5. Surface transitions – existing S-transition maneuver
    """
    specs: List[ManeuverSpec] = []

    # ── Tier 1: Longitudinal ──────────────────────────────────────────────
    # Straight accel-brake at many throttle levels
    for throttle in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        t_str = f"{throttle:.0%}".replace("%", "pct")
        specs.append(ManeuverSpec(
            name=f"straight_accel_t{t_str}",
            preset="straight-accel-brake",
            params={"idle_s": 1.0, "accel_s": 5.0, "coast_s": 1.0, "brake_s": 2.0,
                    "throttle": throttle, "brake": 0.65},
            description=f"Straight accel-coast-brake at throttle={throttle:.2f}.",
        ))

    # Brake sweep at moderate speed
    for brake_val in [0.20, 0.40, 0.60, 0.80, 1.00]:
        b_str = f"{brake_val:.0%}".replace("%", "pct")
        specs.append(ManeuverSpec(
            name=f"straight_brake_b{b_str}",
            preset="straight-accel-brake",
            params={"idle_s": 1.0, "accel_s": 4.0, "coast_s": 0.5, "brake_s": 3.0,
                    "throttle": 0.70, "brake": brake_val},
            description=f"Accel then brake at brake={brake_val:.2f}.",
        ))

    # Ramp throttle 0→1 and 1→0
    specs.append(ManeuverSpec(
        name="ramp_throttle_up",
        preset="ramp-throttle",
        params={"idle_s": 1.0, "ramp_s": 8.0, "sample_dt_s": 0.2,
                "throttle_start": 0.0, "throttle_end": 1.0, "steer": 0.0,
                "brake_s": 2.0, "brake": 0.65},
        description="Linear throttle ramp 0→1.",
    ))
    specs.append(ManeuverSpec(
        name="ramp_throttle_down",
        preset="ramp-throttle",
        params={"idle_s": 1.0, "ramp_s": 8.0, "sample_dt_s": 0.2,
                "throttle_start": 1.0, "throttle_end": 0.0, "steer": 0.0,
                "brake_s": 2.0, "brake": 0.65},
        description="Linear throttle ramp 1→0 (coast-down).",
    ))

    # ── Tier 2: Lateral ───────────────────────────────────────────────────
    # Step-steer at multiple throttles × steer magnitudes
    for throttle in [0.30, 0.50, 0.70, 0.90]:
        for steer in [0.10, 0.25, 0.50, 0.75, 1.00]:
            t_str = f"{throttle:.0%}".replace("%", "pct")
            s_str = f"{int(steer * 100)}pct"
            for sign, label in [(1.0, "left"), (-1.0, "right")]:
                specs.append(ManeuverSpec(
                    name=f"step_steer_{label}_t{t_str}_s{s_str}",
                    preset="step-steer",
                    params={"idle_s": 1.0, "entry_s": 3.0, "step_hold_s": 3.0,
                            "recenter_s": 1.0, "brake_s": 1.5,
                            "throttle": throttle, "steer": sign * steer, "brake": 0.65},
                    description=f"Step steer {label} steer={sign * steer:.2f} at throttle={throttle:.2f}.",
                ))

    # Constant-steer (steady-state cornering) at multiple throttles × steer
    for throttle in [0.30, 0.50, 0.70]:
        for steer in [0.15, 0.30, 0.50, 0.80]:
            t_str = f"{throttle:.0%}".replace("%", "pct")
            s_str = f"{int(steer * 100)}pct"
            for sign, label in [(1.0, "left"), (-1.0, "right")]:
                specs.append(ManeuverSpec(
                    name=f"const_steer_{label}_t{t_str}_s{s_str}",
                    preset="constant-steer",
                    params={"idle_s": 1.0, "hold_s": 5.0, "brake_s": 1.5,
                            "throttle": throttle, "steer": sign * steer, "brake": 0.65},
                    description=f"Constant steer {label} steer={sign * steer:.2f} at throttle={throttle:.2f}.",
                ))

    # ── Tier 3: Combined (trail-brake) ────────────────────────────────────
    for throttle in [0.60, 0.80, 1.00]:
        for steer in [0.20, 0.40, 0.70]:
            t_str = f"{throttle:.0%}".replace("%", "pct")
            s_str = f"{int(steer * 100)}pct"
            for sign, label in [(1.0, "left"), (-1.0, "right")]:
                specs.append(ManeuverSpec(
                    name=f"trail_brake_{label}_t{t_str}_s{s_str}",
                    preset="trail-brake",
                    params={"idle_s": 1.0, "accel_s": 4.0, "trail_s": 3.0,
                            "sample_dt_s": 0.2, "throttle": throttle,
                            "steer": sign * steer, "brake_peak": 0.80, "coast_s": 1.0},
                    description=f"Trail-brake {label} steer={sign * steer:.2f} from throttle={throttle:.2f}.",
                ))

    # ── Tier 4: Frequency response ────────────────────────────────────────
    # Sine-steer at multiple frequencies, amplitudes, throttles
    for throttle in [0.35, 0.55, 0.80]:
        for amplitude in [0.15, 0.30, 0.50]:
            for freq_hz in [0.25, 0.50, 1.00, 2.00]:
                t_str = f"{throttle:.0%}".replace("%", "pct")
                a_str = f"{int(amplitude * 100)}pct"
                f_str = f"{freq_hz:.2f}hz".replace(".", "p")
                specs.append(ManeuverSpec(
                    name=f"sine_steer_t{t_str}_a{a_str}_f{f_str}",
                    preset="sine-steer",
                    params={"idle_s": 1.0, "run_s": 6.0, "sample_dt_s": 0.05,
                            "throttle": throttle, "amplitude": amplitude,
                            "frequency_hz": freq_hz, "phase_deg": 0.0,
                            "brake_s": 1.5, "brake": 0.65},
                    description=f"Sine steer amp={amplitude:.2f} freq={freq_hz:.2f}Hz at throttle={throttle:.2f}.",
                ))

    # Chirp-steer at multiple throttles
    for throttle in [0.40, 0.65, 0.90]:
        t_str = f"{throttle:.0%}".replace("%", "pct")
        specs.append(ManeuverSpec(
            name=f"chirp_steer_t{t_str}",
            preset="chirp-steer",
            params={"idle_s": 1.0, "run_s": 10.0, "sample_dt_s": 0.05,
                    "throttle": throttle, "amplitude": 0.30,
                    "freq_start_hz": 0.1, "freq_end_hz": 2.5,
                    "brake_s": 1.5, "brake": 0.65},
            description=f"Chirp steer 0.1–2.5 Hz at throttle={throttle:.2f}.",
        ))

    # ── Tier 5: Surface transition ────────────────────────────────────────
    specs.append(ManeuverSpec(
        name="surface_transition_s",
        preset="surface-transition-s",
        params={"launch_s": 1.0, "dry_s": 3.5, "wet_s": 2.0, "gravel_s": 2.0,
                "brake_s": 1.5, "launch_throttle": 0.20, "cruise_throttle": 0.55,
                "wet_steer": 0.24, "gravel_steer": -0.24, "brake": 0.60},
        description="Cross dry, wet, and gravel patches with an S-turn and brake.",
    ))

    return specs


def build_dataset_suite(suite_name: str) -> List[ManeuverSpec]:
    suites: Dict[str, List[ManeuverSpec]] = {
        "smoke": [
            ManeuverSpec(
                name="straight_accel_brake_smoke",
                preset="straight-accel-brake",
                params={"idle_s": 1.0, "accel_s": 2.0, "coast_s": 0.5, "brake_s": 1.0, "throttle": 0.45, "brake": 0.55},
                description="Minimal smoke maneuver with accel, coast, and brake.",
            )
        ],
        "sysid-comprehensive-fwd": _build_comprehensive_fwd_suite(),
        "sysid-basic-fwd": [
            ManeuverSpec(
                name="straight_accel_brake",
                preset="straight-accel-brake",
                params={"idle_s": 1.0, "accel_s": 4.0, "coast_s": 1.0, "brake_s": 2.0, "throttle": 0.60, "brake": 0.65},
                description="Straight launch, coast-down, and service brake.",
            ),
            ManeuverSpec(
                name="step_steer_left",
                preset="step-steer",
                params={"idle_s": 1.0, "entry_s": 2.0, "step_hold_s": 2.0, "recenter_s": 1.0, "throttle": 0.45, "steer": 0.25, "brake_s": 1.5, "brake": 0.65},
                description="Step steer left at constant throttle, then recenter and brake.",
            ),
            ManeuverSpec(
                name="step_steer_right",
                preset="step-steer",
                params={"idle_s": 1.0, "entry_s": 2.0, "step_hold_s": 2.0, "recenter_s": 1.0, "throttle": 0.45, "steer": -0.25, "brake_s": 1.5, "brake": 0.65},
                description="Step steer right at constant throttle, then recenter and brake.",
            ),
            ManeuverSpec(
                name="constant_steer_left",
                preset="constant-steer",
                params={"idle_s": 1.0, "hold_s": 4.0, "throttle": 0.35, "steer": 0.18, "brake_s": 1.5, "brake": 0.65},
                description="Constant-radius left turn at fixed throttle.",
            ),
            ManeuverSpec(
                name="constant_steer_right",
                preset="constant-steer",
                params={"idle_s": 1.0, "hold_s": 4.0, "throttle": 0.35, "steer": -0.18, "brake_s": 1.5, "brake": 0.65},
                description="Constant-radius right turn at fixed throttle.",
            ),
            ManeuverSpec(
                name="sine_steer",
                preset="sine-steer",
                params={"idle_s": 1.0, "run_s": 6.0, "sample_dt_s": 0.1, "throttle": 0.42, "amplitude": 0.20, "frequency_hz": 0.50, "phase_deg": 0.0, "brake_s": 1.5, "brake": 0.65},
                description="Constant-throttle sine steering for frequency response and lateral fit.",
            ),
            ManeuverSpec(
                name="surface_transition_s",
                preset="surface-transition-s",
                params={"launch_s": 1.0, "dry_s": 3.5, "wet_s": 2.0, "gravel_s": 2.0, "brake_s": 1.5, "launch_throttle": 0.20, "cruise_throttle": 0.55, "wet_steer": 0.24, "gravel_steer": -0.24, "brake": 0.60},
                description="Cross dry, wet, and gravel patches with an S-turn and brake.",
            ),
        ],
    }
    if str(suite_name) not in suites:
        raise KeyError(f"Unknown dataset suite: {suite_name}")
    return list(suites[str(suite_name)])


def _rollout_is_complete(rollout_dir: str | Path) -> bool:
    rollout_root = Path(rollout_dir).expanduser().resolve()
    return (rollout_root / "rollout_meta.json").exists() and (rollout_root / "rollout_frames.jsonl").exists()


def _write_replay_for_rollout(rollout_dir: str | Path, *, frame_stride: int = 1) -> Path:
    rollout_root = Path(rollout_dir).expanduser().resolve()
    meta, frames = load_rollout_dir(rollout_root)
    stride_value = max(1, int(frame_stride))
    replay_frames = list(frames[::stride_value])
    html_text = build_replay_html(
        meta,
        replay_frames,
        title=f"PhysX Teacher Replay: {rollout_root.name}",
        source_rollout_dir=str(rollout_root),
        frame_stride=stride_value,
    )
    replay_path = rollout_root / "replay.html"
    replay_path.write_text(html_text, encoding="utf-8")
    return replay_path


def build_teacher_record_command(
    *,
    record_python: str,
    program_path: str | Path,
    rollout_dir: str | Path,
    headless: bool,
    dt: float,
    track_width_m: float,
    patch_length_m: float,
    spawn_height_m: float,
    warmup_steps: int,
    settle_steps: int,
    max_steps: int,
) -> List[str]:
    cmd = [
        str(record_python),
        "-m",
        "src.physx_teacher_patch_track",
        "--command-program",
        str(Path(program_path).expanduser().resolve()),
        "--output-dir",
        str(Path(rollout_dir).expanduser().resolve()),
        "--dt",
        str(float(dt)),
        "--track-width-m",
        str(float(track_width_m)),
        "--patch-length-m",
        str(float(patch_length_m)),
        "--spawn-height-m",
        str(float(spawn_height_m)),
        "--warmup-steps",
        str(int(warmup_steps)),
        "--settle-steps",
        str(int(settle_steps)),
    ]
    if bool(headless):
        cmd.append("--headless")
    if int(max_steps) > 0:
        cmd.extend(["--max-steps", str(int(max_steps))])
    return cmd


def generate_programs(dataset_dir: str | Path, suite_specs: Sequence[ManeuverSpec]) -> List[Dict[str, Any]]:
    dataset_root = Path(dataset_dir).expanduser().resolve()
    programs_dir = dataset_root / "programs"
    entries: List[Dict[str, Any]] = []
    for spec in suite_specs:
        program = build_program_from_preset(spec.preset, **spec.params)
        program_path = write_command_program(programs_dir / f"{spec.name}.json", program)
        entries.append(
            {
                "name": spec.name,
                "preset": spec.preset,
                "description": spec.description,
                "params": dict(spec.params),
                "path": str(program_path),
            }
        )
    return entries


def build_dataset_manifest(
    *,
    dataset_dir: str | Path,
    dataset_name: str,
    suite_name: str,
    program_entries: Sequence[Dict[str, Any]],
    headless: bool,
    dt_s: float,
    track_width_m: float,
    patch_length_m: float,
    spawn_height_m: float,
    warmup_steps: int,
    settle_steps: int,
    max_steps: int,
    record_python: str,
) -> Dict[str, Any]:
    dataset_root = Path(dataset_dir).expanduser().resolve()
    surface_patches = build_default_surface_patches(
        patch_length_m=float(patch_length_m),
        track_width_m=float(track_width_m),
    )
    rollouts: List[Dict[str, Any]] = []
    for program_entry in program_entries:
        rollout_root = dataset_root / "rollouts" / str(program_entry["name"])
        rollouts.append(
            {
                "name": str(program_entry["name"]),
                "preset": str(program_entry["preset"]),
                "description": str(program_entry["description"]),
                "program_path": str(program_entry["path"]),
                "rollout_dir": str(rollout_root),
                "rollout_meta_path": str(rollout_root / "rollout_meta.json"),
                "rollout_frames_path": str(rollout_root / "rollout_frames.jsonl"),
                "replay_html_path": str(rollout_root / "replay.html"),
                "status": "pending",
                "params": dict(program_entry["params"]),
            }
        )

    return {
        "dataset_name": str(dataset_name),
        "suite_name": str(suite_name),
        "dataset_dir": str(dataset_root),
        "created_utc": _utc_now_iso(),
        "teacher_recording": {
            "record_python": str(record_python),
            "headless": bool(headless),
            "dt_s": float(dt_s),
            "track_width_m": float(track_width_m),
            "patch_length_m": float(patch_length_m),
            "spawn_height_m": float(spawn_height_m),
            "warmup_steps": int(warmup_steps),
            "settle_steps": int(settle_steps),
            "max_steps": int(max_steps),
        },
        "surface_patches": [patch.to_dict() for patch in surface_patches],
        "programs": list(program_entries),
        "rollouts": rollouts,
    }


def _write_manifest(manifest_path: str | Path, manifest: Dict[str, Any]) -> None:
    output_path = Path(manifest_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fixed teacher dataset by creating maneuver JSONs and recording PhysX rollouts."
    )
    parser.add_argument("--dataset-dir", type=str, required=True, help="Dataset root containing programs/, rollouts/, and manifest.json.")
    parser.add_argument("--dataset-name", type=str, default="", help="Optional manifest name override. Defaults to the dataset directory name.")
    parser.add_argument("--suite", type=str, default="sysid-comprehensive-fwd", choices=["smoke", "sysid-basic-fwd", "sysid-comprehensive-fwd"])
    parser.add_argument("--record-python", type=str, default=sys.executable, help="Python executable used to launch the teacher recorder.")
    parser.add_argument("--generate-only", action="store_true", help="Only write maneuver JSONs and manifest; do not record rollouts.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip rollouts that already have both rollout files.")
    parser.add_argument("--skip-replays", action="store_true", help="Do not generate replay.html files after recording.")
    parser.add_argument("--replay-frame-stride", type=int, default=1, help="Frame downsampling factor for generated replay HTML files.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--track-width-m", type=float, default=10.0)
    parser.add_argument("--patch-length-m", type=float, default=12.0)
    parser.add_argument("--spawn-height-m", type=float, default=1.2)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=0, help="Optional per-rollout step cap. 0 uses each program's full duration.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(args.dataset_name) if args.dataset_name else dataset_dir.name

    suite_specs = build_dataset_suite(args.suite)
    program_entries = generate_programs(dataset_dir, suite_specs)
    manifest = build_dataset_manifest(
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        suite_name=args.suite,
        program_entries=program_entries,
        headless=bool(args.headless),
        dt_s=float(args.dt),
        track_width_m=float(args.track_width_m),
        patch_length_m=float(args.patch_length_m),
        spawn_height_m=float(args.spawn_height_m),
        warmup_steps=int(args.warmup_steps),
        settle_steps=int(args.settle_steps),
        max_steps=int(args.max_steps),
        record_python=str(args.record_python),
    )
    manifest_path = dataset_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)

    if args.generate_only:
        print(f"[physx_teacher_dataset_builder] wrote programs and manifest to {dataset_dir}")
        return

    for index, rollout_entry in enumerate(manifest["rollouts"], start=1):
        rollout_dir = Path(rollout_entry["rollout_dir"])
        rollout_dir.mkdir(parents=True, exist_ok=True)
        rollout_name = str(rollout_entry["name"])

        if bool(args.skip_existing) and _rollout_is_complete(rollout_dir):
            rollout_entry["status"] = "existing"
            rollout_entry["completed_utc"] = _utc_now_iso()
            if not bool(args.skip_replays):
                replay_path = _write_replay_for_rollout(rollout_dir, frame_stride=int(args.replay_frame_stride))
                rollout_entry["replay_html_path"] = str(replay_path)
            _write_manifest(manifest_path, manifest)
            print(f"[physx_teacher_dataset_builder] [{index}/{len(manifest['rollouts'])}] skipped existing {rollout_name}")
            continue

        record_cmd = build_teacher_record_command(
            record_python=str(args.record_python),
            program_path=rollout_entry["program_path"],
            rollout_dir=rollout_dir,
            headless=bool(args.headless),
            dt=float(args.dt),
            track_width_m=float(args.track_width_m),
            patch_length_m=float(args.patch_length_m),
            spawn_height_m=float(args.spawn_height_m),
            warmup_steps=int(args.warmup_steps),
            settle_steps=int(args.settle_steps),
            max_steps=int(args.max_steps),
        )

        print(f"[physx_teacher_dataset_builder] [{index}/{len(manifest['rollouts'])}] recording {rollout_name}")
        result = subprocess.run(record_cmd, check=False)
        if result.returncode != 0:
            rollout_entry["status"] = "failed"
            rollout_entry["completed_utc"] = _utc_now_iso()
            _write_manifest(manifest_path, manifest)
            raise SystemExit(result.returncode)

        if not _rollout_is_complete(rollout_dir):
            rollout_entry["status"] = "failed"
            rollout_entry["completed_utc"] = _utc_now_iso()
            _write_manifest(manifest_path, manifest)
            raise RuntimeError(f"Rollout did not produce expected files: {rollout_dir}")

        rollout_entry["status"] = "recorded"
        rollout_entry["completed_utc"] = _utc_now_iso()
        if not bool(args.skip_replays):
            replay_path = _write_replay_for_rollout(rollout_dir, frame_stride=int(args.replay_frame_stride))
            rollout_entry["replay_html_path"] = str(replay_path)
        _write_manifest(manifest_path, manifest)

    print(f"[physx_teacher_dataset_builder] dataset ready at {dataset_dir}")


if __name__ == "__main__":
    main()
