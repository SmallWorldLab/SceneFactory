from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict

from src.physx_teacher_patch_track import CommandProgram, CommandSegment, VehicleCommand


def write_command_program(path: str | Path, program: CommandProgram) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"segments": program.to_dict_list()}
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path


def build_straight_accel_brake_program(
    *,
    idle_s: float = 1.0,
    accel_s: float = 4.0,
    coast_s: float = 1.0,
    brake_s: float = 2.0,
    throttle: float = 0.60,
    brake: float = 0.65,
) -> CommandProgram:
    return CommandProgram(
        [
            CommandSegment(float(idle_s), VehicleCommand(0.0, 0.0, 0.0), "idle"),
            CommandSegment(float(accel_s), VehicleCommand(throttle, 0.0, 0.0), "straight_accel"),
            CommandSegment(float(coast_s), VehicleCommand(0.0, 0.0, 0.0), "coast"),
            CommandSegment(float(brake_s), VehicleCommand(0.0, 0.0, brake), "service_brake"),
        ]
    )


def build_step_steer_program(
    *,
    idle_s: float = 1.0,
    entry_s: float = 2.0,
    step_hold_s: float = 2.0,
    recenter_s: float = 1.0,
    brake_s: float = 1.5,
    throttle: float = 0.45,
    steer: float = 0.25,
    brake: float = 0.65,
) -> CommandProgram:
    return CommandProgram(
        [
            CommandSegment(float(idle_s), VehicleCommand(0.0, 0.0, 0.0), "idle"),
            CommandSegment(float(entry_s), VehicleCommand(throttle, 0.0, 0.0), "entry"),
            CommandSegment(float(step_hold_s), VehicleCommand(throttle, steer, 0.0), "step_steer"),
            CommandSegment(float(recenter_s), VehicleCommand(throttle, 0.0, 0.0), "recenter"),
            CommandSegment(float(brake_s), VehicleCommand(0.0, 0.0, brake), "service_brake"),
        ]
    )


def build_constant_steer_program(
    *,
    idle_s: float = 1.0,
    hold_s: float = 4.0,
    brake_s: float = 1.5,
    throttle: float = 0.35,
    steer: float = 0.18,
    brake: float = 0.65,
) -> CommandProgram:
    return CommandProgram(
        [
            CommandSegment(float(idle_s), VehicleCommand(0.0, 0.0, 0.0), "idle"),
            CommandSegment(float(hold_s), VehicleCommand(throttle, steer, 0.0), "constant_steer"),
            CommandSegment(float(brake_s), VehicleCommand(0.0, 0.0, brake), "service_brake"),
        ]
    )


def build_sine_steer_program(
    *,
    idle_s: float = 1.0,
    run_s: float = 6.0,
    sample_dt_s: float = 0.1,
    throttle: float = 0.42,
    amplitude: float = 0.20,
    frequency_hz: float = 0.50,
    phase_deg: float = 0.0,
    brake_s: float = 1.5,
    brake: float = 0.65,
) -> CommandProgram:
    run_duration_s = float(run_s)
    sample_dt = float(sample_dt_s)
    if sample_dt <= 0.0:
        raise ValueError(f"sample_dt_s must be positive, got {sample_dt_s}")

    phase_rad = math.radians(float(phase_deg))
    segments = [CommandSegment(float(idle_s), VehicleCommand(0.0, 0.0, 0.0), "idle")]

    sample_count = int(math.ceil(run_duration_s / sample_dt))
    for sample_idx in range(sample_count):
        t0_s = float(sample_idx * sample_dt)
        duration_s = min(sample_dt, run_duration_s - t0_s)
        if duration_s <= 0.0:
            continue
        steer = float(amplitude) * math.sin(2.0 * math.pi * float(frequency_hz) * t0_s + phase_rad)
        segments.append(
            CommandSegment(
                float(duration_s),
                VehicleCommand(throttle, steer, 0.0),
                f"sine_steer_{sample_idx:04d}",
            )
        )

    segments.append(CommandSegment(float(brake_s), VehicleCommand(0.0, 0.0, brake), "service_brake"))
    return CommandProgram(segments)


def build_surface_transition_s_program(
    *,
    launch_s: float = 1.0,
    dry_s: float = 3.5,
    wet_s: float = 2.0,
    gravel_s: float = 2.0,
    brake_s: float = 1.5,
    launch_throttle: float = 0.20,
    cruise_throttle: float = 0.55,
    wet_steer: float = 0.24,
    gravel_steer: float = -0.24,
    brake: float = 0.60,
) -> CommandProgram:
    return CommandProgram(
        [
            CommandSegment(float(launch_s), VehicleCommand(launch_throttle, 0.0, 0.0), "launch"),
            CommandSegment(float(dry_s), VehicleCommand(0.65, 0.0, 0.0), "cross_dry"),
            CommandSegment(float(wet_s), VehicleCommand(cruise_throttle, wet_steer, 0.0), "wet_patch_left"),
            CommandSegment(float(gravel_s), VehicleCommand(cruise_throttle, gravel_steer, 0.0), "gravel_patch_right"),
            CommandSegment(float(brake_s), VehicleCommand(0.0, 0.0, brake), "brake"),
        ]
    )


def build_ramp_throttle_program(
    *,
    idle_s: float = 1.0,
    ramp_s: float = 6.0,
    sample_dt_s: float = 0.2,
    throttle_start: float = 0.0,
    throttle_end: float = 1.0,
    steer: float = 0.0,
    brake_s: float = 2.0,
    brake: float = 0.65,
) -> CommandProgram:
    """Linear throttle ramp from throttle_start to throttle_end."""
    segments = [CommandSegment(float(idle_s), VehicleCommand(0.0, 0.0, 0.0), "idle")]
    n_samples = max(1, int(math.ceil(float(ramp_s) / float(sample_dt_s))))
    for i in range(n_samples):
        frac = i / max(1, n_samples - 1)
        throttle = float(throttle_start) + frac * (float(throttle_end) - float(throttle_start))
        dur = min(float(sample_dt_s), float(ramp_s) - i * float(sample_dt_s))
        if dur <= 0:
            break
        segments.append(CommandSegment(dur, VehicleCommand(throttle, float(steer), 0.0), f"ramp_{i:04d}"))
    segments.append(CommandSegment(float(brake_s), VehicleCommand(0.0, 0.0, float(brake)), "service_brake"))
    return CommandProgram(segments)


def build_trail_brake_program(
    *,
    idle_s: float = 1.0,
    accel_s: float = 4.0,
    trail_s: float = 3.0,
    sample_dt_s: float = 0.2,
    throttle: float = 0.80,
    steer: float = 0.30,
    brake_peak: float = 0.80,
    coast_s: float = 1.0,
) -> CommandProgram:
    """Accelerate, then simultaneously apply steering while ramping brake and reducing throttle."""
    segments = [
        CommandSegment(float(idle_s), VehicleCommand(0.0, 0.0, 0.0), "idle"),
        CommandSegment(float(accel_s), VehicleCommand(float(throttle), 0.0, 0.0), "straight_accel"),
    ]
    n_samples = max(1, int(math.ceil(float(trail_s) / float(sample_dt_s))))
    for i in range(n_samples):
        frac = i / max(1, n_samples - 1)
        t = float(throttle) * (1.0 - frac)
        b = float(brake_peak) * frac
        s = float(steer) * min(1.0, frac * 2.0)  # steer ramps in first half
        dur = min(float(sample_dt_s), float(trail_s) - i * float(sample_dt_s))
        if dur <= 0:
            break
        segments.append(CommandSegment(dur, VehicleCommand(t, s, b), f"trail_{i:04d}"))
    segments.append(CommandSegment(float(coast_s), VehicleCommand(0.0, 0.0, 0.0), "coast"))
    return CommandProgram(segments)


def build_chirp_steer_program(
    *,
    idle_s: float = 1.0,
    run_s: float = 8.0,
    sample_dt_s: float = 0.05,
    throttle: float = 0.50,
    amplitude: float = 0.30,
    freq_start_hz: float = 0.2,
    freq_end_hz: float = 2.0,
    brake_s: float = 1.5,
    brake: float = 0.65,
) -> CommandProgram:
    """Chirp (frequency sweep) steering for broad frequency-response identification."""
    segments = [CommandSegment(float(idle_s), VehicleCommand(0.0, 0.0, 0.0), "idle")]
    n_samples = max(1, int(math.ceil(float(run_s) / float(sample_dt_s))))
    for i in range(n_samples):
        t_s = i * float(sample_dt_s)
        dur = min(float(sample_dt_s), float(run_s) - t_s)
        if dur <= 0:
            break
        frac = t_s / float(run_s)
        freq = float(freq_start_hz) + frac * (float(freq_end_hz) - float(freq_start_hz))
        steer_val = float(amplitude) * math.sin(2.0 * math.pi * freq * t_s)
        segments.append(CommandSegment(dur, VehicleCommand(float(throttle), steer_val, 0.0), f"chirp_{i:04d}"))
    segments.append(CommandSegment(float(brake_s), VehicleCommand(0.0, 0.0, float(brake)), "service_brake"))
    return CommandProgram(segments)


def _add_shared_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-json", type=str, required=True, help="Where to write the generated command program JSON.")


def _add_common_brake_args(parser: argparse.ArgumentParser, *, default_brake_s: float = 1.5, default_brake: float = 0.65) -> None:
    parser.add_argument("--brake-s", type=float, default=default_brake_s)
    parser.add_argument("--brake", type=float, default=default_brake)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate teacher command-program JSON files for PhysX vehicle rollout collection."
    )
    subparsers = parser.add_subparsers(dest="preset", required=True)

    straight = subparsers.add_parser("straight-accel-brake", help="Idle, accelerate, coast, then brake.")
    _add_shared_output_args(straight)
    straight.add_argument("--idle-s", type=float, default=1.0)
    straight.add_argument("--accel-s", type=float, default=4.0)
    straight.add_argument("--coast-s", type=float, default=1.0)
    straight.add_argument("--throttle", type=float, default=0.60)
    _add_common_brake_args(straight, default_brake_s=2.0)

    step = subparsers.add_parser("step-steer", help="Accelerate, apply a steering step, then recenter and brake.")
    _add_shared_output_args(step)
    step.add_argument("--idle-s", type=float, default=1.0)
    step.add_argument("--entry-s", type=float, default=2.0)
    step.add_argument("--step-hold-s", type=float, default=2.0)
    step.add_argument("--recenter-s", type=float, default=1.0)
    step.add_argument("--throttle", type=float, default=0.45)
    step.add_argument("--steer", type=float, default=0.25)
    _add_common_brake_args(step)

    constant = subparsers.add_parser("constant-steer", help="Hold constant throttle and steering, then brake.")
    _add_shared_output_args(constant)
    constant.add_argument("--idle-s", type=float, default=1.0)
    constant.add_argument("--hold-s", type=float, default=4.0)
    constant.add_argument("--throttle", type=float, default=0.35)
    constant.add_argument("--steer", type=float, default=0.18)
    _add_common_brake_args(constant)

    sine = subparsers.add_parser("sine-steer", help="Discretized sine-wave steering at constant throttle, then brake.")
    _add_shared_output_args(sine)
    sine.add_argument("--idle-s", type=float, default=1.0)
    sine.add_argument("--run-s", type=float, default=6.0)
    sine.add_argument("--sample-dt-s", type=float, default=0.1)
    sine.add_argument("--throttle", type=float, default=0.42)
    sine.add_argument("--amplitude", type=float, default=0.20)
    sine.add_argument("--frequency-hz", type=float, default=0.50)
    sine.add_argument("--phase-deg", type=float, default=0.0)
    _add_common_brake_args(sine)

    surface = subparsers.add_parser("surface-transition-s", help="Cross dry, wet, and gravel patches with an S-turn and brake.")
    _add_shared_output_args(surface)
    surface.add_argument("--launch-s", type=float, default=1.0)
    surface.add_argument("--dry-s", type=float, default=3.5)
    surface.add_argument("--wet-s", type=float, default=2.0)
    surface.add_argument("--gravel-s", type=float, default=2.0)
    surface.add_argument("--launch-throttle", type=float, default=0.20)
    surface.add_argument("--cruise-throttle", type=float, default=0.55)
    surface.add_argument("--wet-steer", type=float, default=0.24)
    surface.add_argument("--gravel-steer", type=float, default=-0.24)
    _add_common_brake_args(surface)

    return parser


def build_program_from_preset(preset: str, **kwargs: Any) -> CommandProgram:
    builders: Dict[str, Callable[..., CommandProgram]] = {
        "straight-accel-brake": build_straight_accel_brake_program,
        "step-steer": build_step_steer_program,
        "constant-steer": build_constant_steer_program,
        "sine-steer": build_sine_steer_program,
        "surface-transition-s": build_surface_transition_s_program,
        "ramp-throttle": build_ramp_throttle_program,
        "trail-brake": build_trail_brake_program,
        "chirp-steer": build_chirp_steer_program,
    }
    if str(preset) not in builders:
        raise KeyError(f"Unknown command-program preset: {preset}")
    return builders[str(preset)](**kwargs)


def _program_from_args(args: argparse.Namespace) -> CommandProgram:
    builders: Dict[str, Callable[[argparse.Namespace], CommandProgram]] = {
        "straight-accel-brake": lambda ns: build_straight_accel_brake_program(
            idle_s=ns.idle_s,
            accel_s=ns.accel_s,
            coast_s=ns.coast_s,
            brake_s=ns.brake_s,
            throttle=ns.throttle,
            brake=ns.brake,
        ),
        "step-steer": lambda ns: build_step_steer_program(
            idle_s=ns.idle_s,
            entry_s=ns.entry_s,
            step_hold_s=ns.step_hold_s,
            recenter_s=ns.recenter_s,
            brake_s=ns.brake_s,
            throttle=ns.throttle,
            steer=ns.steer,
            brake=ns.brake,
        ),
        "constant-steer": lambda ns: build_constant_steer_program(
            idle_s=ns.idle_s,
            hold_s=ns.hold_s,
            brake_s=ns.brake_s,
            throttle=ns.throttle,
            steer=ns.steer,
            brake=ns.brake,
        ),
        "sine-steer": lambda ns: build_sine_steer_program(
            idle_s=ns.idle_s,
            run_s=ns.run_s,
            sample_dt_s=ns.sample_dt_s,
            throttle=ns.throttle,
            amplitude=ns.amplitude,
            frequency_hz=ns.frequency_hz,
            phase_deg=ns.phase_deg,
            brake_s=ns.brake_s,
            brake=ns.brake,
        ),
        "surface-transition-s": lambda ns: build_surface_transition_s_program(
            launch_s=ns.launch_s,
            dry_s=ns.dry_s,
            wet_s=ns.wet_s,
            gravel_s=ns.gravel_s,
            brake_s=ns.brake_s,
            launch_throttle=ns.launch_throttle,
            cruise_throttle=ns.cruise_throttle,
            wet_steer=ns.wet_steer,
            gravel_steer=ns.gravel_steer,
            brake=ns.brake,
        ),
    }
    return builders[str(args.preset)](args)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    program = _program_from_args(args)
    output_path = write_command_program(args.output_json, program)
    print(f"[physx_teacher_command_program_generator] wrote {args.preset} program to {output_path}")


if __name__ == "__main__":
    main()
