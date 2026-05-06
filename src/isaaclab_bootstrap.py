from __future__ import annotations

from pathlib import Path
import sys


def isaaclab_source_root() -> Path:
    """Return the sibling IsaacLab source root used in this workspace."""
    return Path(__file__).resolve().parents[2] / "IsaacLab" / "source"


def ensure_isaaclab_source_paths() -> Path:
    """Add local Isaac Lab source trees to ``sys.path`` if needed."""
    source_root = isaaclab_source_root()
    for package_name in ("isaaclab", "isaaclab_rl", "isaaclab_assets"):
        package_root = source_root / package_name
        if package_root.is_dir():
            package_root_str = str(package_root)
            if package_root_str not in sys.path:
                sys.path.insert(0, package_root_str)
    return source_root
