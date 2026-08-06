"""
Physics validation test for the SceneFactory PhysX vehicle.

Runs 3 controlled experiments in parallel across 14 worlds (1 agent each):
  Phase 1 (envs  0– 3): Longitudinal — throttle sweep [0.25, 0.50, 0.75, 1.00], steer=0
  Phase 2 (envs  4– 8): Lateral      — fixed throttle=0.5, steer sweep [-1, -0.5, 0, 0.5, 1]
  Phase 3 (envs  9–13): Friction      — full throttle then full brake, μ = [0.02, 0.20, 0.40, 0.65, 0.95]

Launch via:
    bash run_physics_validation.sh
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


# ── Phase parameters ──────────────────────────────────────────────────────────
SETTLE_STEPS  = 48    # zero-action warmup: vehicle drops from spawn height and settles
RAMP_STEPS    = 30    # throttle ramp-up steps at start of drive phases (suppresses suspension impulse)
PHASE1_STEPS  = 400   # throttle sweep drive steps (~6.7 s at 60 Hz)
PHASE2_STEPS  = 300   # steer sweep drive steps   (~5.0 s)
PHASE3_DRIVE  = 400   # friction drive steps       (~6.7 s)
PHASE3_BRAKE  = 400   # friction brake steps       (~6.7 s) — extended to ensure full stop
TOTAL_STEPS  = SETTLE_STEPS + max(PHASE1_STEPS, PHASE2_STEPS, PHASE3_DRIVE + PHASE3_BRAKE)

# Env slices
_P1 = list(range(0, 4))    # Phase 1 env indices
_P2 = list(range(4, 9))    # Phase 2 env indices
_P3 = list(range(9, 14))   # Phase 3 env indices

# Action parameters
THROTTLE_LEVELS = [0.25, 0.50, 0.75, 1.00]
STEER_LEVELS    = [-1.00, -0.50, 0.00, 0.50, 1.00]
# Must match friction_ruler_mu_values in configs/scene_factory/physics_validation.yaml
# (last 5 entries: envs 9–13)
MU_VALUES       = [0.02, 0.20, 0.40, 0.65, 0.95]

# Pass/fail thresholds
P1_MIN_FULL_THROTTLE_SPEED = 5.0   # m/s — minimum acceptable peak speed at throttle=1.0
_P1_MAX_Z_RANGE_M = 0.15           # m   — maximum acceptable vertical oscillation (grounded criterion)
P2_SYMMETRY_TOL            = 0.20  # fraction — left/right radius must match within 20 %
P2_STRAIGHT_YAWRATE_MAX    = 0.05  # rad/s — steer=0 yaw rate ceiling
P3_MIN_FRICTION_EFFECT     = 0.20  # fraction — (v_hi_mu - v_lo_mu) / v_hi_mu >= 0.20
_P3_STOP_SPEED_THRESHOLD   = 0.15  # m/s — speed below which the vehicle is considered stopped


# ── Helpers ───────────────────────────────────────────────────────────────────

def _planar_speed(vel_b: torch.Tensor) -> torch.Tensor:
    """Planar speed (m/s) from body-frame velocity [..., 3]."""
    return torch.norm(vel_b[..., :2], dim=-1)


def _yaw_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
    """Yaw angle (rad) from IsaacLab quaternion convention [w, x, y, z] [..., 4]."""
    w, x, y, z = quat_w[..., 0], quat_w[..., 1], quat_w[..., 2], quat_w[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _build_action_dict(env, throttle: torch.Tensor, steer: torch.Tensor, brake: torch.Tensor) -> dict:
    """Build action dict from per-agent-per-env components [num_agents, num_envs]."""
    action_dict: dict[str, torch.Tensor] = {}
    for agent_idx, agent_id in enumerate(env.cfg.possible_agents):
        action_dict[agent_id] = torch.stack(
            [throttle[agent_idx], steer[agent_idx], brake[agent_idx]], dim=-1
        ).to(device=env.device, dtype=torch.float32)
    return action_dict


# ── Main test ─────────────────────────────────────────────────────────────────

def run_physics_validation(env, run_dir: Path) -> None:
    """
    Run the physics validation test.  The env must be configured with:
      - num_envs = 14
      - num_agents_per_env = 1
      - dynamics_mode = "physx"
      - friction_ruler_mode = True
      - friction_ruler_mu_values covering 14 envs (first 9 at μ=1.0, last 5 at MU_VALUES)
    """
    device  = env.device
    N       = env.num_envs
    dt      = float(env.cfg.sim.dt) * float(env.cfg.decimation)  # wall-clock per env step (s)

    assert N == 14, f"physics_validation requires exactly 14 envs, got {N}"

    # ── Storage ───────────────────────────────────────────────────────────────
    p1_speed: list[list[float]] = [[] for _ in _P1]   # [local_env][step]
    p1_z:     list[list[float]] = [[] for _ in _P1]

    p2_speed: list[list[float]] = [[] for _ in _P2]
    p2_yaw:   list[list[float]] = [[] for _ in _P2]

    p3_speed: list[list[float]] = [[] for _ in _P3]
    # Stopping distance: cumulative XY displacement after brake engages, locked
    # once speed drops below _P3_STOP_SPEED_THRESHOLD (or at end of brake window).
    p3_brake_start_pos:   list[tuple[float, float]] = [(0.0, 0.0)] * len(_P3)
    p3_brake_last_pos:    list[tuple[float, float]] = [(0.0, 0.0)] * len(_P3)
    p3_stop_dist_locked:  list[float] = [float("nan")] * len(_P3)
    p3_brake_entry_speed: list[float] = [0.0] * len(_P3)   # speed at the start of brake phase

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"[PhysicsValidation] Resetting {N} worlds (1 agent each)...", flush=True)
    env.reset()

    print(f"[PhysicsValidation] Running {TOTAL_STEPS} steps "
          f"(settle={SETTLE_STEPS}, test≤{TOTAL_STEPS - SETTLE_STEPS})...", flush=True)

    for step in range(TOTAL_STEPS):
        t = step - SETTLE_STEPS  # drive-phase step index (negative during settle)

        throttle = torch.zeros(1, N, device=device)
        steer    = torch.zeros(1, N, device=device)
        brake    = torch.zeros(1, N, device=device)

        if t >= 0:
            # Throttle ramp factor: linearly ramp from 0→1 over the first RAMP_STEPS.
            # This spreads the initial drive impulse, suppressing suspension resonance.
            ramp = min(1.0, (t + 1) / max(1, RAMP_STEPS)) if t < RAMP_STEPS else 1.0

            # Phase 1: throttle sweep (envs 0–3)
            if t < PHASE1_STEPS:
                for li, gi in enumerate(_P1):
                    throttle[0, gi] = THROTTLE_LEVELS[li] * ramp

            # Phase 2: steer sweep (envs 4–8)
            if t < PHASE2_STEPS:
                for li, gi in enumerate(_P2):
                    throttle[0, gi] = 0.5 * ramp
                    steer[0, gi]    = STEER_LEVELS[li] * ramp

            # Phase 3: friction — drive then brake (envs 9–13)
            if t < PHASE3_DRIVE:
                for gi in _P3:
                    throttle[0, gi] = 1.0 * ramp
            elif t < PHASE3_DRIVE + PHASE3_BRAKE:
                for gi in _P3:
                    brake[0, gi] = 1.0

        env.step(_build_action_dict(env, throttle, steer, brake))

        if t < 0:
            continue  # still settling

        # Read state from PhysX (authoritative source)
        vehicle = env._vehicles[0]
        vel_b   = vehicle.data.root_lin_vel_b.detach().cpu()   # [N, 3]
        pos_w   = vehicle.data.root_pos_w.detach().cpu()        # [N, 3]
        quat_w  = vehicle.data.root_quat_w.detach().cpu()       # [N, 4]
        speeds  = _planar_speed(vel_b)                          # [N]
        yaws    = _yaw_from_quat(quat_w)                        # [N]

        # Phase 1 record
        if t < PHASE1_STEPS:
            for li, gi in enumerate(_P1):
                p1_speed[li].append(float(speeds[gi]))
                p1_z[li].append(float(pos_w[gi, 2]))

        # Phase 2 record
        if t < PHASE2_STEPS:
            for li, gi in enumerate(_P2):
                p2_speed[li].append(float(speeds[gi]))
                p2_yaw[li].append(float(yaws[gi]))

        # Phase 3 record
        if t < PHASE3_DRIVE:
            for li, gi in enumerate(_P3):
                p3_speed[li].append(float(speeds[gi]))
            if t == PHASE3_DRIVE - 1:
                # Capture brake start state: position and speed at the last drive step
                # (state reflects drive phase end, before brake command takes effect).
                for li, gi in enumerate(_P3):
                    xy = (float(pos_w[gi, 0]), float(pos_w[gi, 1]))
                    p3_brake_start_pos[li]   = xy
                    p3_brake_last_pos[li]    = xy
                    p3_brake_entry_speed[li] = float(speeds[gi])
        elif PHASE3_DRIVE <= t < PHASE3_DRIVE + PHASE3_BRAKE:
            for li, gi in enumerate(_P3):
                # Update stop_dist only while vehicle is still moving
                if math.isnan(p3_stop_dist_locked[li]):
                    cur_xy = (float(pos_w[gi, 0]), float(pos_w[gi, 1]))
                    cum_dist = math.hypot(
                        cur_xy[0] - p3_brake_start_pos[li][0],
                        cur_xy[1] - p3_brake_start_pos[li][1],
                    )
                    if float(speeds[gi]) < _P3_STOP_SPEED_THRESHOLD:
                        # Vehicle has stopped — lock this distance
                        p3_stop_dist_locked[li] = cum_dist
                    else:
                        p3_brake_last_pos[li] = cur_xy

        if (step + 1) % 100 == 0:
            p1s = " ".join(f"{p1_speed[i][-1]:.2f}" for i in range(4)) if p1_speed[0] else "N/A"
            p3s = " ".join(f"{p3_speed[i][-1]:.2f}" if p3_speed[i] else "?" for i in range(5))
            print(f"  step {step+1:4d}/{TOTAL_STEPS}  "
                  f"phase1=[{p1s}]  phase3=[{p3s}]", flush=True)

    # ── Analyse ───────────────────────────────────────────────────────────────
    report = _analyse(p1_speed, p1_z, p2_speed, p2_yaw, p3_speed,
                      p3_brake_start_pos, p3_brake_last_pos,
                      p3_stop_dist_locked, p3_brake_entry_speed, dt)
    _print_report(report)

    out_path = run_dir / "physics_validation_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    # The JSON is for tooling. Write the same human-readable report the printer
    # produces to a .txt as well -- otherwise the only readable form is buried in
    # the Isaac Sim console log.
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_report(report)
    txt_path = run_dir / "physics_validation_report.txt"
    txt_path.write_text(buf.getvalue())
    print(f"[PhysicsValidation] Report → {out_path}", flush=True)
    print(f"[PhysicsValidation] Readable report → {txt_path}", flush=True)


# ── Analysis ──────────────────────────────────────────────────────────────────

def _turning_radius(speed_list: list[float], yaw_list: list[float], dt: float,
                    tail: int = 100) -> float:
    """Compute mean turning radius from the last `tail` steps."""
    spd = speed_list[-tail:] if len(speed_list) >= tail else speed_list
    yaw = yaw_list[-tail:]   if len(yaw_list)   >= tail else yaw_list
    if len(yaw) < 2:
        return math.inf
    dyaw = torch.tensor(yaw, dtype=torch.float64)
    dyaw = torch.diff(dyaw)
    # Unwrap ±π discontinuities
    dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
    yaw_rate = float(dyaw.abs().mean()) / dt  # rad/s
    mean_spd = sum(spd) / len(spd) if spd else 0.0
    if yaw_rate < 1e-4:
        return math.inf
    return mean_spd / yaw_rate


def _analyse(
    p1_speed, p1_z,
    p2_speed, p2_yaw,
    p3_speed, p3_brake_start_pos, p3_brake_last_pos,
    p3_stop_dist_locked, p3_brake_entry_speed,
    dt: float,
) -> dict[str, Any]:

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    p1_peak   = [max(s) if s else 0.0 for s in p1_speed]
    # Two different questions, previously conflated into one number:
    #   launch  = did the car bounce while the drive impulse came on?
    #   steady  = is it riding on the ground once it settled?
    # Only the second says the physics is wrong. Measuring max-min over the WHOLE
    # phase let a single launch excursion decide 'grounded', which is why the
    # failures were non-monotonic in throttle (bad at 0.50 and 1.00, fine at 0.25
    # and 0.75) -- a signature of transients, not of contact.
    p1_z_range = [max(z) - min(z) if z else 0.0 for z in p1_z]          # full phase
    _tail = lambda z: z[int(len(z) * 0.6):] if len(z) >= 5 else z
    p1_z_range_steady = [max(_tail(z)) - min(_tail(z)) if z else 0.0 for z in p1_z]
    p1_mono   = all(p1_peak[i] < p1_peak[i + 1] for i in range(len(p1_peak) - 1))
    p1_grounded = all(zr < _P1_MAX_Z_RANGE_M for zr in p1_z_range_steady)
    p1_pass   = p1_peak[-1] >= P1_MIN_FULL_THROTTLE_SPEED and p1_mono and p1_grounded

    # Speed curve sampled every 20 steps (compact for JSON)
    p1_curves = {
        f"throttle_{t:.2f}": s[::20] for t, s in zip(THROTTLE_LEVELS, p1_speed)
    }

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    p2_radii = [_turning_radius(p2_speed[i], p2_yaw[i], dt) for i in range(len(_P2))]

    r_left  = p2_radii[0]   # steer = -1.0
    r_right = p2_radii[4]   # steer = +1.0
    if not math.isinf(r_left) and not math.isinf(r_right) and r_left > 0 and r_right > 0:
        symm_err = abs(r_left - r_right) / max(r_left, r_right)
        p2_symmetry = bool(symm_err <= P2_SYMMETRY_TOL)
    else:
        symm_err = float("inf")
        p2_symmetry = False

    # Straight: env 2 (steer=0)
    yaw_straight = p2_yaw[2]
    if len(yaw_straight) >= 2:
        dyaw0 = torch.tensor(yaw_straight, dtype=torch.float64)
        dyaw0 = (torch.diff(dyaw0) + math.pi) % (2 * math.pi) - math.pi
        yaw_rate_0 = float(dyaw0.abs().mean()) / dt
    else:
        yaw_rate_0 = 0.0
    p2_straight = yaw_rate_0 < P2_STRAIGHT_YAWRATE_MAX

    p2_pass = p2_symmetry and p2_straight

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    p3_peak     = [max(s) if s else 0.0 for s in p3_speed]
    # Use locked stop_dist (vehicle stopped) or fallback to final cumulative dist
    p3_stop_dist = [
        p3_stop_dist_locked[i]
        if not math.isnan(p3_stop_dist_locked[i])
        else math.hypot(
            p3_brake_last_pos[i][0] - p3_brake_start_pos[i][0],
            p3_brake_last_pos[i][1] - p3_brake_start_pos[i][1],
        )
        for i in range(len(_P3))
    ]
    # Effective deceleration = v_entry² / (2 * stop_dist), where v_entry is the
    # vehicle speed at the moment braking starts.
    # Higher μ → more traction → harder braking → higher effective decel.
    # This metric is used for the monotonicity check because raw stop_dist is
    # NOT monotonically decreasing with μ when terminal drive speeds differ:
    # low-μ vehicles reach lower terminal speeds but have much worse braking,
    # while high-μ vehicles reach higher terminal speeds and brake much harder.
    # Effective deceleration captures the braking force regardless of entry speed.
    p3_eff_decel = [
        v * v / (2.0 * d) if d > 1e-3 else 0.0
        for v, d in zip(p3_brake_entry_speed, p3_stop_dist)
    ]
    p3_mono_peak = all(p3_peak[i] < p3_peak[i + 1] for i in range(len(p3_peak) - 1))
    # Higher μ → higher effective deceleration (monotonically increasing)
    p3_mono_stop = all(p3_eff_decel[i] <= p3_eff_decel[i + 1] for i in range(len(p3_eff_decel) - 1))
    hi, lo = p3_peak[-1], p3_peak[0]
    p3_effect = (hi - lo) / hi if hi > 1e-3 else 0.0
    p3_pass = p3_mono_peak and p3_mono_stop and float(p3_effect) >= P3_MIN_FRICTION_EFFECT

    overall = p1_pass and p2_pass and p3_pass

    return {
        "phase1_longitudinal": {
            "throttle_levels": THROTTLE_LEVELS,
            "peak_speed_mps": p1_peak,
            "z_range_m": p1_z_range,
            "z_range_steady_m": p1_z_range_steady,
            "grounded": p1_grounded,
            "monotonic": p1_mono,
            "full_throttle_peak_mps": p1_peak[-1],
            "threshold_mps": P1_MIN_FULL_THROTTLE_SPEED,
            "pass": p1_pass,
            "speed_curves_sampled_20": p1_curves,
        },
        "phase2_lateral": {
            "steer_levels": STEER_LEVELS,
            "turning_radii_m": [r if not math.isinf(r) else -1.0 for r in p2_radii],
            "symmetry_error_fraction": symm_err if not math.isinf(symm_err) else -1.0,
            "symmetry_ok": p2_symmetry,
            "straight_yaw_rate_rad_s": yaw_rate_0,
            "straight_ok": p2_straight,
            "pass": p2_pass,
        },
        "phase3_friction": {
            "mu_values": MU_VALUES,
            "peak_speed_mps": p3_peak,
            "stopping_distance_m": p3_stop_dist,
            "effective_decel_mps2": p3_eff_decel,
            "monotonic_peak": p3_mono_peak,
            "monotonic_eff_decel": p3_mono_stop,
            "friction_effect_fraction": float(p3_effect),
            "threshold_effect": P3_MIN_FRICTION_EFFECT,
            "pass": p3_pass,
        },
        "overall_pass": overall,
        "dt_s": dt,
    }


# ── Report printer ─────────────────────────────────────────────────────────────

def _print_report(r: dict) -> None:
    p1 = r["phase1_longitudinal"]
    p2 = r["phase2_lateral"]
    p3 = r["phase3_friction"]

    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  PHYSICS VALIDATION REPORT")
    print(SEP)

    print("\nPhase 1 — Longitudinal (steer=0, μ=1.0)")
    for t, spd, zr, zs in zip(p1["throttle_levels"], p1["peak_speed_mps"],
                              p1["z_range_m"], p1.get("z_range_steady_m", p1["z_range_m"])):
        tag = " ← full throttle" if t == 1.0 else ""
        z_flag = " ← BAD (not grounded)" if zs >= _P1_MAX_Z_RANGE_M else ""
        launch = "  (launch transient)" if zr >= _P1_MAX_Z_RANGE_M > zs else ""
        print(f"  throttle={t:.2f}  peak={spd:.2f} m/s  z_steady={zs:.3f} m  "
              f"z_full={zr:.3f} m{tag}{z_flag}{launch}")
    print(f"  grounded(z<{_P1_MAX_Z_RANGE_M:.2f}m): {p1['grounded']}  monotonic: {p1['monotonic']}  "
          f"→ {'PASS' if p1['pass'] else 'FAIL'}")

    print("\nPhase 2 — Lateral (throttle=0.5, μ=1.0)")
    for s, rad in zip(p2["steer_levels"], p2["turning_radii_m"]):
        r_str = f"{rad:.1f} m" if rad > 0 else "∞ (straight)"
        print(f"  steer={s:+.1f}  R={r_str}")
    print(f"  symmetry_err={p2['symmetry_error_fraction']:.2f}  "
          f"straight_yaw_rate={p2['straight_yaw_rate_rad_s']:.3f} rad/s  "
          f"→ {'PASS' if p2['pass'] else 'FAIL'}")

    print("\nPhase 3 — Friction (throttle=1.0 → brake=1.0)")
    for mu, spd, stop, decel in zip(
        p3["mu_values"], p3["peak_speed_mps"],
        p3["stopping_distance_m"], p3["effective_decel_mps2"]
    ):
        print(f"  μ={mu:.2f}  peak={spd:.2f} m/s  stop_dist={stop:.1f} m  eff_decel={decel:.2f} m/s²")
    print(f"  mono_peak={p3['monotonic_peak']}  mono_eff_decel={p3['monotonic_eff_decel']}  "
          f"friction_effect={p3['friction_effect_fraction']:.1%}  "
          f"→ {'PASS' if p3['pass'] else 'FAIL'}")

    print(f"\n{SEP}")
    overall = "PASS ✓" if r["overall_pass"] else "FAIL ✗"
    print(f"  OVERALL: {overall}")
    print(f"{SEP}\n")
