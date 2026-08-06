#!/usr/bin/env python3
"""Migrate scene JSONs from the ``workzone`` block to the generic ``zones`` block.

Why
---
The backbone reads only two things out of the old ``workzone`` block: the point
obstacles and the keep-out boxes.  Both are generic simulator concepts -- any
research line placing obstacles or closing a region needs them -- but the block
name meant SceneFactory's own scene format declared that it knew what a workzone
was.  It also meant genuinely general code (``compute_multilane_capacity``,
``sample_multilane_merge_start_goal_pairs``) had to reach into a key called
"workzone" to find a closed lane.

This rewrites::

    "workzone": {                        "zones": {
      "cones":            [...]   ->       "obstacles":     [...],
      "forbidden_boxes":  [...]   ->       "keepout_boxes": [...],
      "speed_limit_mps":  x       ->       "speed_limit_mps": x
    }                                    }

and moves the three workzone-only provenance keys -- which only ``dev/`` ever
reads -- into an extension block, establishing the pattern a second research
line can follow::

    "extensions": {"workzone": {"length_m": ..., "center_x": ..., "cone_cfg": {...}}}

Migrated files are stamped ``"sf_version": "2.0"``, which also makes the
migration idempotent.  (SF_FORMAT 1.0 was specified but never actually emitted;
2.0 is the first version written to disk.)

Usage
-----
    python scripts/migrate_scene_schema.py --root data/processed --dry-run
    python scripts/migrate_scene_schema.py --root data/processed --backup
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

SF_VERSION = "2.0"

# old workzone-block key -> new zones-block key
_ZONE_KEYS = {
    "cones": "obstacles",
    "forbidden_boxes": "keepout_boxes",
    "speed_limit_mps": "speed_limit_mps",
}
# old workzone-block key -> key under extensions.workzone
_EXTENSION_KEYS = {
    "workzone_length_m": "length_m",
    "workzone_center_x": "center_x",
    "cone_cfg": "cone_cfg",
}


def migrate_scene(scene: dict) -> tuple[dict, bool]:
    """Return ``(scene, changed)``.  Safe to call repeatedly."""
    if scene.get("sf_version") == SF_VERSION and "workzone" not in scene:
        return scene, False
    if "workzone" not in scene:
        # Plain Waymo scene with no workzone block: just stamp the version.
        changed = scene.get("sf_version") != SF_VERSION
        scene["sf_version"] = SF_VERSION
        return scene, changed

    wz = scene.pop("workzone") or {}
    if not isinstance(wz, dict):
        scene["sf_version"] = SF_VERSION
        return scene, True

    zones: dict = dict(scene.get("zones") or {})
    extension: dict = dict((scene.get("extensions") or {}).get("workzone") or {})

    for old, new in _ZONE_KEYS.items():
        if old in wz:
            zones[new] = wz.pop(old)
    for old, new in _EXTENSION_KEYS.items():
        if old in wz:
            extension[new] = wz.pop(old)
    # Anything unrecognised is workzone-specific by definition; keep it in the
    # extension rather than silently dropping it.
    for leftover, value in wz.items():
        extension[leftover] = value

    if zones:
        scene["zones"] = zones
    if extension:
        scene.setdefault("extensions", {})["workzone"] = extension
    scene["sf_version"] = SF_VERSION
    return scene, True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/processed",
                    help="directory tree to walk (default: data/processed)")
    ap.add_argument("--glob", default="scene_*.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change, write nothing")
    ap.add_argument("--backup", action="store_true",
                    help="write <file>.pre_sf2.bak beside each migrated file")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    paths = sorted(root.rglob(args.glob))
    changed = skipped = failed = 0
    per_dir: dict[str, int] = {}

    for path in paths:
        try:
            scene = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FAIL  {path}: {exc}", file=sys.stderr)
            failed += 1
            continue
        migrated, did = migrate_scene(scene)
        if not did:
            skipped += 1
            continue
        changed += 1
        per_dir[str(path.parent)] = per_dir.get(str(path.parent), 0) + 1
        if args.dry_run:
            continue
        if args.backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".pre_sf2.bak"))
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(migrated))
        tmp.replace(path)   # atomic: a crash cannot leave a half-written scene

    for d, n in sorted(per_dir.items()):
        print(f"{n:6d}  {d}")
    verb = "would migrate" if args.dry_run else "migrated"
    print(f"\n{verb} {changed}, already current {skipped}, failed {failed}, "
          f"of {len(paths)} files under {root}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
