#!/usr/bin/env python3
"""Pre-flight check for a SceneFactory install.

Verifies everything that can be verified WITHOUT starting Isaac Sim, so a broken
install reports in about a second instead of after a two-minute simulator boot
followed by a traceback.

Usage:
    PYTHONPATH=. python scripts/check_install.py
    PYTHONPATH=. python scripts/check_install.py --config configs/scene_factory/demo_weather_physx_train.yaml

Exit code 0 if every required check passes, 1 otherwise.

Deliberately does NOT import isaacsim/isaaclab or allocate a CUDA context: this
must be runnable on a login node or a busy machine.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util as iu
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

OK, WARN, FAIL = "ok", "warn", "fail"
_MARK = {OK: "[ ok ]", WARN: "[warn]", FAIL: "[FAIL]"}

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"{_MARK[status]} {name}" + (f" — {detail}" if detail else ""))


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def check_python() -> None:
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 11):
        record(OK, "Python 3.11", got)
    else:
        record(FAIL, "Python 3.11", f"found {got}; Isaac Sim 5.1.0 wheels are cp311")


def check_packages() -> None:
    required = {
        "isaacsim": "5.1.0.0",
        "isaaclab": "0.54.3",
        "rsl-rl-lib": "3.1.2",
        "torch": None,
        "numpy": None,
        "pyyaml": None,
        "stable-baselines3": None,
        "gymnasium": None,
    }
    for dist, expected in required.items():
        got = _version(dist)
        if got is None:
            hint = {
                "isaacsim": "README step 2",
                "isaaclab": "README step 3",
                "stable-baselines3": "pip install -r requirements.txt",
            }.get(dist, "pip install -r requirements.txt")
            record(FAIL, f"package {dist}", f"not installed — see {hint}")
        elif expected and got != expected:
            record(WARN, f"package {dist}", f"found {got}, expected {expected}")
        else:
            record(OK, f"package {dist}", got)


def check_isaaclab_sibling() -> None:
    """src/isaaclab_bootstrap.py resolves IsaacLab as a SIBLING of the repo."""
    sibling = REPO_ROOT.parent / "IsaacLab" / "source"
    inside = REPO_ROOT / "IsaacLab" / "source"
    if sibling.is_dir():
        record(OK, "IsaacLab source tree", str(sibling))
    elif inside.is_dir():
        record(
            FAIL,
            "IsaacLab source tree",
            f"found at {inside}, but it must be a SIBLING of the repo: "
            f"{sibling}. Move it up one level.",
        )
    else:
        record(
            FAIL,
            "IsaacLab source tree",
            f"not found at {sibling} — clone IsaacLab next to this repo, not inside it",
        )


def check_assets() -> None:
    assets = [
        "artifacts/student_vehicle_assets/vehicle_student/student_fwd_vehicle.usd",
        "artifacts/low_poly_car_proxy.usd",
        "artifacts/student_vehicle_sysid/comprehensive_fwd_v1_cem_v4/best_config.json",
    ]
    for rel in assets:
        p = REPO_ROOT / rel
        if p.is_file() and p.stat().st_size > 0:
            record(OK, f"asset {rel}", f"{p.stat().st_size} B")
        else:
            record(FAIL, f"asset {rel}", "missing or empty")


def check_scene_data(config: Path | None) -> None:
    scene_dir = REPO_ROOT / "data" / "processed" / "waymo_scenes_json"
    scenes = sorted(scene_dir.glob("*.json")) if scene_dir.is_dir() else []
    if scenes:
        record(OK, "Waymo scene JSONs", f"{len(scenes)} in {scene_dir}")
    else:
        record(
            FAIL,
            "Waymo scene JSONs",
            f"none in {scene_dir}. Training and eval CANNOT run without these. "
            "They are not redistributable — see README 'Preparing scene data'.",
        )

    if config is None:
        return
    try:
        import yaml
    except ImportError:
        return
    if not config.is_file():
        record(FAIL, f"config {config.name}", "file not found")
        return
    try:
        cfg = yaml.safe_load(config.read_text())
        scene_cfg_rel = cfg.get("scene_factory", {}).get("config_path")
        if not scene_cfg_rel:
            record(WARN, f"config {config.name}", "no scene_factory.config_path")
            return
        scene_cfg_path = REPO_ROOT / scene_cfg_rel
        if not scene_cfg_path.is_file():
            record(FAIL, f"config {config.name}", f"scene pool missing: {scene_cfg_rel}")
            return
        scene_cfg = yaml.safe_load(scene_cfg_path.read_text())
        wanted = {
            str(a.get("scene_json"))
            for a in (scene_cfg.get("world", {}).get("assignments") or [])
            if a.get("scene_json")
        }
        missing = sorted(n for n in wanted if not (scene_dir / n).is_file())
        if missing:
            record(
                FAIL,
                f"config {config.name}",
                f"{len(missing)}/{len(wanted)} referenced scenes missing "
                f"(first: {missing[0]})",
            )
        else:
            record(OK, f"config {config.name}", f"all {len(wanted)} referenced scenes present")
    except Exception as exc:  # noqa: BLE001 - report, don't crash the checker
        record(FAIL, f"config {config.name}", f"could not parse: {exc}")


def check_friction_model() -> None:
    if iu.find_spec("src.trfc.friction_api") is None:
        record(FAIL, "friction model", "src.trfc not importable — run with PYTHONPATH=.")
        return
    from src.trfc.friction_api import compute_mu_all_modified

    mu_dry = compute_mu_all_modified(v_ref=100 / 3.6, slip=0.15, h_w_mm=0.0, road_type="AC").mu
    mu_wet = compute_mu_all_modified(v_ref=100 / 3.6, slip=0.15, h_w_mm=1.0, road_type="AC").mu
    if mu_dry > mu_wet > 0.0:
        record(OK, "friction model", f"mu(dry)={mu_dry:.3f} > mu(1mm)={mu_wet:.3f} > 0")
    else:
        record(FAIL, "friction model", f"mu(dry)={mu_dry:.3f}, mu(1mm)={mu_wet:.3f}")


# Isaac Sim's libcarb.so requires this symbol version.
_REQUIRED_GLIBCXX = b"GLIBCXX_3.4.30"


def _provides_required_glibcxx(path: Path) -> bool:
    try:
        return _REQUIRED_GLIBCXX in path.read_bytes()
    except OSError:
        return False


def check_libstdcxx() -> None:
    """Isaac Sim binds the FIRST libstdc++.so.6 on the loader path.

    A stale directory on LD_LIBRARY_PATH — very commonly some other Anaconda
    install — shadows the environment's own copy and Isaac Sim dies with
    'Unable to bootstrap inner kit kernel: ... GLIBCXX_3.4.30 not found',
    long after this checker would otherwise have said everything is fine.
    """
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    search: list[Path] = [Path(d) for d in ld_path.split(":") if d.strip()]

    conda_prefix = os.environ.get("CONDA_PREFIX")
    env_lib = Path(conda_prefix) / "lib" if conda_prefix else None
    if env_lib:
        search.append(env_lib)
    search.append(Path("/usr/lib/x86_64-linux-gnu"))

    winner: Path | None = None
    for d in search:
        candidate = d / "libstdc++.so.6"
        if candidate.exists():
            winner = candidate
            break

    if winner is None:
        record(WARN, "libstdc++", "no libstdc++.so.6 found on the loader path")
        return

    resolved = winner.resolve()
    if _provides_required_glibcxx(resolved):
        record(OK, "libstdc++", f"{resolved.name} provides GLIBCXX_3.4.30 ({winner.parent})")
        return

    hint = (
        f"{winner} (-> {resolved.name}) does NOT provide GLIBCXX_3.4.30, and it "
        "shadows every later entry. Isaac Sim will fail with 'Unable to bootstrap "
        "inner kit kernel'."
    )
    if env_lib and winner.parent != env_lib:
        hint += (
            f" Fix: drop that directory from LD_LIBRARY_PATH, or put your env first: "
            f'export LD_LIBRARY_PATH="{env_lib}:$LD_LIBRARY_PATH"'
        )
    else:
        hint += " Fix: conda install -c conda-forge 'libstdcxx-ng>=12'"
    record(FAIL, "libstdc++", hint)


def check_foreign_ld_library_path() -> None:
    """Flag loader-path entries belonging to a different user's home."""
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld_path.strip():
        return
    home = Path.home()
    foreign = [
        d
        for d in ld_path.split(":")
        if d.strip().startswith("/home/") and not Path(d).is_relative_to(home)
    ]
    if foreign:
        record(
            WARN,
            "LD_LIBRARY_PATH",
            "contains another user's directories, which take precedence over your "
            f"environment: {', '.join(foreign)}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/scene_factory/demo_weather_physx_train.yaml",
        help="training config to validate scene references against",
    )
    args = parser.parse_args()

    print(f"SceneFactory install check — repo at {REPO_ROOT}\n")
    check_python()
    check_packages()
    check_isaaclab_sibling()
    check_libstdcxx()
    check_foreign_ld_library_path()
    check_assets()
    check_friction_model()
    check_scene_data(args.config)

    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)
    print(f"\n{len(results)} checks — {n_fail} failed, {n_warn} warnings")
    if n_fail:
        print("\nInstall is NOT ready. Fix the [FAIL] lines above, then re-run.")
    else:
        print("\nInstall looks good. Next: bash run_demo_train.sh")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
