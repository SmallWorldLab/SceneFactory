"""A config asking for a non-builtin OD mode must declare the plugin providing it.

Without the declaration the run aborts at reset. That is better than the old
silent fallback to lane spawns, but it is better still to catch it here, before
anyone waits for Isaac Sim to boot.
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
BUILTIN_OD_MODES = {"lane", "multilane_merge"}


def _configs():
    return sorted(REPO.glob("configs/**/*.yaml")) + sorted(REPO.glob("research/**/*.yaml"))


def _load(path: Path):
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}          # non-config yaml (saved run params etc.)


@pytest.mark.parametrize("path", _configs(), ids=lambda p: p.name)
def test_non_builtin_od_mode_declares_its_plugin(path: Path):
    cfg = _load(path)
    if not isinstance(cfg, dict):
        pytest.skip("not a mapping")
    mode = None
    for section in cfg.values():
        if isinstance(section, dict) and "random_od_mode" in section:
            mode = str(section["random_od_mode"]).strip()
    if mode is None or mode in BUILTIN_OD_MODES:
        pytest.skip("builtin or unset OD mode")

    plugins = (cfg.get("scene_factory") or {}).get("plugins") or []
    assert plugins, (
        f"{path.relative_to(REPO)} sets random_od_mode={mode!r}, which is not a "
        f"builtin ({sorted(BUILTIN_OD_MODES)}), but declares no "
        f"scene_factory.plugins. The mode would never register and the run "
        f"would abort at reset."
    )
