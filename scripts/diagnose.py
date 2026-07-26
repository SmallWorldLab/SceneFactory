#!/usr/bin/env python3
"""End-to-end diagnostic: is this install actually working?

Runs a staged set of checks and prints a coverage table mapping each check to
the defect it verifies. Stage 1 needs no GPU; stage 2 launches Isaac Sim.

    # stage 1 only -- fast, no GPU, safe on a busy machine
    PYTHONPATH=. python scripts/diagnose.py

    # stages 1 and 2 -- boots Isaac Sim on the named device
    PYTHONPATH=. python scripts/diagnose.py --gpu --device cuda:0

Exit code 0 if every check that ran passed, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_MARK = {PASS: "[PASS]", FAIL: "[FAIL]", SKIP: "[skip]"}

# check name -> what it proves
COVERAGE = {
    "install": "requirements, IsaacLab layout, assets, libstdc++ (ibEY defects 1, 2, 3)",
    "friction-tests": "Eq. (12) coefficient A is dimensionless; mu no longer cliffs at 0.8 mm",
    "friction-paper": "corrected model vs the cited paper's published curves",
    "scene-cycling": "world_count may exceed the number of available scene JSONs",
    "config-resolution": "demo config resolves to real scene files before Isaac Sim boots",
    "ground-params": "ground cuboid size and contact_offset are configurable",
    "traction-probe": "every agent drives on a CUBOID ground, not just agent 0 (wheel tunnelling)",
    "physics-validation": "longitudinal, lateral and friction response of the PhysX vehicle",
}

results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"{_MARK[status]} {name}" + (f" - {detail}" if detail else ""), flush=True)


LOG_DIR = REPO_ROOT / "artifacts" / "diagnose" / "logs"


def _save_log(name: str, text: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{name}.log"
    path.write_text(text)
    return path


def _echo_report(text: str, start_marker: str) -> bool:
    """Re-print a child's own report block on our stdout.

    The child's output is captured so failures can be classified, which means it
    never reaches the terminal. Rather than making the reader open a file, echo
    the report block back out.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if start_marker in line:
            block = lines[max(0, i - 1):]
            print()
            for b in block:
                print("    " + b)
            print()
            return True
    return False


def _run(cmd: list[str], timeout: float, env: dict | None = None) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
            timeout=timeout, env=env or {**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


# --------------------------------------------------------------------------- #
# stage 1 -- no GPU
# --------------------------------------------------------------------------- #

def check_install() -> None:
    rc, out = _run([sys.executable, "scripts/check_install.py"], timeout=180)
    n_fail = 0
    for line in out.splitlines():
        if line.startswith("[FAIL]"):
            n_fail += 1
    if rc == 0:
        record("install", PASS, "all check_install.py checks green")
    else:
        scene_only = all("scene" in l.lower() or "config demo" in l.lower()
                         for l in out.splitlines() if l.startswith("[FAIL]"))
        if scene_only and n_fail:
            record("install", FAIL, f"{n_fail} failures, all scene-data related - run README section 5")
        else:
            record("install", FAIL, f"{n_fail} failing checks; run scripts/check_install.py")


def check_friction_tests() -> None:
    # Disable pytest's third-party plugin autoload. Isaac Sim's dependency tree
    # can register plugins whose own imports fail (e.g. a Flask plugin pulling
    # 'blinker'), which aborts collection and looks like our tests failing.
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    rc, out = _run([sys.executable, "-m", "pytest",
                    "src/trfc/tests/test_friction_water_film.py", "-q", "-p", "no:cacheprovider"],
                   timeout=300, env=env)
    tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
    record("friction-tests", PASS if rc == 0 else FAIL, tail[0][:90])


def check_friction_paper() -> None:
    out_dir = REPO_ROOT / "artifacts" / "diagnose" / "trfc"
    rc, out = _run([sys.executable, "scripts/trfc_paper_validation.py", "--out", str(out_dir)],
                   timeout=300)
    # Criterion 1 cannot pass simultaneously with Fig. 6 -- the source paper is
    # internally inconsistent -- so a nonzero exit is expected. What must hold is
    # that the mu-cliff criteria pass.
    ok = ("[PASS] 3." in out) and ("[PASS] 4." in out)
    detail = "criteria 3 and 4 pass (criterion 1 cannot; see docs/friction_model.md)"
    record("friction-paper", PASS if ok else FAIL, detail if ok else "mu-cliff criteria did not pass")


def check_scene_cycling() -> None:
    """world_count may now exceed the number of scene JSONs."""
    code = (
        "import sys; sys.path.insert(0, '.');\n"
        "from src.trfc.world_pipeline import prepare_stage_world_specs as P;\n"
        "import tempfile, json, pathlib;\n"
        "d = pathlib.Path(tempfile.mkdtemp());\n"
        "[ (d / f'scene_{i:06d}.json').write_text(json.dumps("
        "  {'meta': {}, 'road': {'polylines': []}, 'agents': []})) for i in range(3) ];\n"
        "s = P({'io': {'scene_json_dir': str(d)}, 'world': {'world_count': 7}});\n"
        "assert len(s) == 7, len(s);\n"
        "names = [x.scene_json_name for x in s];\n"
        "assert len(set(names)) == 3, names;\n"
        "print('cycled', len(s), 'worlds over', len(set(names)), 'scenes')\n"
    )
    rc, out = _run([sys.executable, "-c", code], timeout=120)
    record("scene-cycling", PASS if rc == 0 else FAIL,
           out.strip().splitlines()[-1][:90] if out.strip() else "")


def check_config_resolution() -> None:
    code = (
        "import sys, yaml; sys.path.insert(0, '.');\n"
        "from src.trfc.world_pipeline import prepare_stage_world_specs as P;\n"
        "c = yaml.safe_load(open('configs/scene_factory/demo_weather_scenes.yaml'));\n"
        "s = P(c); print('resolved', len(s), 'worlds')\n"
    )
    rc, out = _run([sys.executable, "-c", code], timeout=120)
    if rc == 0:
        record("config-resolution", PASS, out.strip().splitlines()[-1][:90])
    else:
        record("config-resolution", FAIL, "demo scene pool does not resolve - see README section 5")


def check_ground_params() -> None:
    """The ground cuboid must accept size and contact_offset."""
    code = (
        "import ast, pathlib;\n"
        "src = pathlib.Path('src/student_vehicle_goal_env.py').read_text();\n"
        "t = ast.parse(src);\n"
        "fns = {n.name: n for n in ast.walk(t) if isinstance(n, ast.FunctionDef)};\n"
        "for f in ('_spawn_ground', '_spawn_local_ground_plane'):\n"
        "    args = [a.arg for a in fns[f].args.args];\n"
        "    assert 'size_m' in args and 'contact_offset' in args, (f, args);\n"
        "assert 'rest_offset=0.0' in src;\n"
        "env = pathlib.Path('src/student_vehicle_multiagent_goal_env.py').read_text();\n"
        "assert 'ground_cuboid_size_m' in env and 'ground_contact_offset_m' in env;\n"
        "print('ground params present and wired')\n"
    )
    rc, out = _run([sys.executable, "-c", code], timeout=120)
    record("ground-params", PASS if rc == 0 else FAIL,
           out.strip().splitlines()[-1][:90] if out.strip() else "not wired")


# --------------------------------------------------------------------------- #
# stage 2 -- needs a GPU
# --------------------------------------------------------------------------- #

def _gpu_index(device: str) -> str:
    return device.split(":")[1] if ":" in device else "0"


def check_traction_probe(args: argparse.Namespace) -> None:
    """Do agents after agent 0 actually drive?

    This is the check that covers the contact_offset fix. The historical defect
    was that agent 0 drove and agents 1..k sat with their wheels spinning inside
    the ground cuboid.
    """
    out_dir = REPO_ROOT / "artifacts" / "diagnose" / "traction_probe"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT),
           "CUDA_VISIBLE_DEVICES": _gpu_index(args.device),
           "TP_DRIVE_STEPS": str(args.drive_steps)}
    cmd = [
        sys.executable, "-u", "src/train_student_vehicle_goal_multiagent_rsl_rl.py",
        "--config", str(args.config), "--headless",
        "--test_mode", "traction_probe",
        "--num_envs", str(args.probe_envs),
        "--num_agents_per_env", str(args.probe_agents),
        "--log_dir", str(out_dir),
        "--experiment_name", "diagnose", "--run_name", "traction_probe",
    ]
    if args.ground_mode != "config":
        spacing = args.env_spacing_m or (args.ground_cuboid_size_m * 1.3)
        cmd += ["--ground_mode", args.ground_mode,
                "--ground_cuboid_size_m", str(args.ground_cuboid_size_m),
                "--env_spacing", str(spacing)]
    rc, out = _run(cmd, timeout=args.gpu_timeout, env=env)
    log = _save_log("traction_probe", out)
    if rc != 0:
        tail = [l for l in out.strip().splitlines() if l.strip()][-1:] or [""]
        record("traction-probe", FAIL, f"rc={rc}: {tail[0][:120]} (full log: {log})")
        return

    report = sorted(out_dir.rglob("traction_probe.json"))
    if not report:
        record("traction-probe", FAIL, "ran but produced no traction_probe.json")
        return
    try:
        data = json.loads(report[-1].read_text())
    except Exception as exc:  # noqa: BLE001
        record("traction-probe", FAIL, f"unreadable report: {exc}")
        return

    # Assert, do not merely report. The historical defect is agent-index
    # dependent: agent 0 drives and agents 1..k sit with wheels spinning, so a
    # fleet mean can look healthy while most agents are stuck.
    per_agent = data.get("per_agent_mean_speed") or []
    stalled = [i for i, v in enumerate(per_agent) if float(v) < args.min_speed_mps]
    idle0 = data.get("idle_fraction_agent0")
    idle_rest = data.get("idle_fraction_agents_1plus")
    if not per_agent:
        record("traction-probe", FAIL, "report has no per_agent_mean_speed")
    elif stalled:
        record("traction-probe", FAIL,
               f"agents {stalled} below {args.min_speed_mps} m/s "
               f"(idle agent0={idle0}, agents1+={idle_rest}) - {data.get('verdict', '')}")
    else:
        speeds = ", ".join(f"{float(v):.2f}" for v in per_agent)
        spread = max(per_agent) - min(per_agent)
        note = ""
        mean_v = sum(per_agent) / max(1, len(per_agent))
        trend = str(data.get("speed_trend", ""))
        if trend == "rising":
            note = " -- speed still RISING at the end; raise --drive-steps"
        elif trend == "falling":
            note = " -- speed FALLING at the end; check collisions / agents leaving the slab"
        # Relative, not absolute: 0.26 m/s spread on a 0.9 m/s mean is 28%.
        if mean_v > 0 and (spread / mean_v) > 0.15:
            note += f" -- per-agent spread {spread / mean_v:.0%} of mean, check index dependence"
        z_min = data.get("min_z_settled_m")
        if isinstance(z_min, (int, float)) and z_min < -0.5:
            note += f" -- an agent settled {z_min:.2f} m BELOW the slab (penetration)"
        if data.get("slabs_overlap"):
            note += " -- slabs OVERLAP (spacing < cuboid)"
        straight = data.get("path_straightness")
        traj = f", straightness {straight:.2f}" if isinstance(straight, (int, float)) else ""
        _echo_report(out, "TRACTION PROBE")
        mode = str(data.get("ground_mode", "")).lower()
        if mode != "cuboid":
            note += (f" -- ran on ground_mode={mode or 'unknown'}, which does NOT "
                     "exercise the contact_offset fix; use --ground-mode cuboid")
        record("traction-probe", PASS,
               f"all {len(per_agent)} agents moving ({speeds} m/s{traj}){note}")


def check_physics_validation(args: argparse.Namespace) -> None:
    out_dir = REPO_ROOT / "artifacts" / "diagnose" / "physics_validation"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT),
           "CUDA_VISIBLE_DEVICES": _gpu_index(args.device)}
    cmd = [
        sys.executable, "-u", "src/train_student_vehicle_goal_multiagent_rsl_rl.py",
        "--config", str(args.config), "--headless",
        "--test_mode", "physics_validation",
        "--num_envs", "14", "--num_agents_per_env", "1",
        "--log_dir", str(out_dir),
        "--experiment_name", "diagnose", "--run_name", "physics_validation",
    ]
    rc, out = _run(cmd, timeout=args.gpu_timeout, env=env)
    log = _save_log("physics_validation", out)
    report = sorted(out_dir.rglob("physics_validation_report.json"))

    if rc != 0 or not report:
        # Surface the traceback, not just the last line -- the last line of an
        # Isaac Sim shutdown is rarely the actual error.
        tb = [l for l in out.splitlines() if "Error" in l or "error:" in l][-1:] or \
             ([l for l in out.strip().splitlines() if l.strip()][-1:] or [""])
        record("physics-validation", FAIL, f"rc={rc}: {tb[0][:130]} (full log: {log})")
        return

    _echo_report(out, "PHYSICS VALIDATION REPORT")

    # A report existing is NOT a pass. Read the verdict the report itself
    # reached, per phase, and fail if any phase failed.
    try:
        data = json.loads(report[-1].read_text())
    except Exception as exc:  # noqa: BLE001
        record("physics-validation", FAIL, f"unreadable report: {exc}")
        return
    phases = {
        "longitudinal": data.get("phase1_longitudinal", {}).get("pass"),
        "lateral": data.get("phase2_lateral", {}).get("pass"),
        "friction": data.get("phase3_friction", {}).get("pass"),
    }
    failed = [k for k, v in phases.items() if v is False]
    overall = data.get("overall_pass")
    if overall is True and not failed:
        record("physics-validation", PASS, "all three phases pass (longitudinal, lateral, friction)")
    else:
        p1 = data.get("phase1_longitudinal", {})
        p3 = data.get("phase3_friction", {})
        bits = []
        if phases["longitudinal"] is False:
            bits.append(
                f"longitudinal: full-throttle peak {p1.get('full_throttle_peak_mps', float('nan')):.2f} m/s "
                f"vs {p1.get('threshold_mps', float('nan')):.1f} expected"
                + ("" if p1.get("grounded", True) else ", and not grounded")
            )
        if phases["lateral"] is False:
            bits.append("lateral: turning symmetry or straight-line test failed")
        if phases["friction"] is False:
            bits.append(
                f"friction: mu changes top speed by only "
                f"{100 * p3.get('friction_effect_fraction', 0.0):.1f}% "
                f"vs {100 * p3.get('threshold_effect', 0.0):.0f}% expected"
            )
        detail = "; ".join(bits) or f"phases failed: {', '.join(failed) or 'unknown'}"
        if overall is None:
            detail += " (report has no overall_pass field)"
        record("physics-validation", FAIL, detail)


# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", action="store_true", help="also run stage 2 (boots Isaac Sim)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "configs/scene_factory/demo_weather_physx_train.yaml")
    p.add_argument("--probe-envs", type=int, default=4)
    p.add_argument("--probe-agents", type=int, default=4,
                   help="must be >1: the tunnelling defect never appears with a single agent")
    p.add_argument("--gpu-timeout", type=float, default=1800.0)
    p.add_argument("--min-speed-mps", type=float, default=0.5,
                   help="per-agent mean speed below which an agent counts as stalled")
    p.add_argument("--ground-mode", choices=("cuboid", "plane", "config"), default="cuboid",
                   help="ground for the traction probe. Default 'cuboid' because the "
                        "contact_offset fix only applies there; 'plane' cannot exercise it. "
                        "'config' leaves the config's own setting alone.")
    p.add_argument("--ground-cuboid-size-m", type=float, default=1000.0)
    p.add_argument("--env-spacing-m", type=float, default=None,
                   help="world spacing. Defaults to 1.3x the cuboid size so slabs do NOT "
                        "overlap; with spacing < cuboid each vehicle rests on several "
                        "worlds' slabs, which changes contact behaviour.")
    p.add_argument("--drive-steps", type=int, default=600,
                   help="traction-probe drive steps. Too few and a healthy vehicle is "
                        "still accelerating when the probe ends, which reads as a low speed.")
    p.add_argument("--skip-physics-validation", action="store_true")
    args = p.parse_args()

    print(f"SceneFactory diagnostics - repo at {REPO_ROOT}\n")
    print("Stage 1: static and CPU checks")
    t0 = time.time()
    check_install()
    check_friction_tests()
    check_friction_paper()
    check_scene_cycling()
    check_config_resolution()
    check_ground_params()

    if args.gpu:
        if args.probe_agents < 2:
            print("\nWARNING: --probe-agents must be > 1 to exercise the tunnelling defect.\n")
        print(f"\nStage 2: GPU checks on {args.device} (boots Isaac Sim, several minutes each)")
        check_traction_probe(args)
        if not args.skip_physics_validation:
            check_physics_validation(args)
        else:
            record("physics-validation", SKIP, "--skip-physics-validation")
    else:
        for n in ("traction-probe", "physics-validation"):
            record(n, SKIP, "pass --gpu to run")

    print("\n" + "=" * 78)
    print("COVERAGE - what each check verifies")
    print("=" * 78)
    width = max(len(n) for n, _, _ in results)
    for name, status, _ in results:
        print(f"  {_MARK[status]} {name:<{width}}  {COVERAGE.get(name, '')}")

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"\n{len(results)} checks - {n_fail} failed, {n_skip} skipped "
          f"({time.time() - t0:.0f}s)")
    if n_fail:
        print("\nNOT healthy. Fix the [FAIL] lines above.")
    elif n_skip:
        print("\nEverything that ran passed. Re-run with --gpu for full coverage.")
    else:
        print("\nAll checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
