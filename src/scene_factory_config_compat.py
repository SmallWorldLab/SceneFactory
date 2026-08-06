"""Detect config keys that were renamed out of the backbone's vocabulary.

Why this exists
---------------
The backbone owns a point-obstacle registry, keep-out regions and a zone speed
limit.  All three were originally named after the first research line that used
them (workzone cones), so the config fields were called things like
``reward_workzone_cone_penalty_enable``.  They have been renamed to the general
capability.

A rename of a *config key* fails silently and dangerously: the loader looks up
the new name, does not find it in an old YAML, and quietly falls back to the
default.  A run configured to enable an obstacle penalty would simply stop
applying it, with nothing in the log to say so.  This module makes that loud.

It does not rewrite the config.  Renaming the key is the user's decision, and a
silent auto-upgrade would hide the fact that a saved experiment config no longer
matches the code that produced its results.
"""

from __future__ import annotations

from typing import Any, Mapping

# old key -> new key.  Keep in sync with the Tier 1 rename.
RENAMED_CONFIG_KEYS: dict[str, str] = {
    "reward_workzone_cone_penalty_enable": "reward_obstacle_penalty_enable",
    "reward_workzone_cone_penalty_alpha": "reward_obstacle_penalty_alpha",
    "reward_workzone_cone_safe_dist_m": "reward_obstacle_safe_dist_m",
    "reward_workzone_cone_collision_dist_m": "reward_obstacle_collision_dist_m",
    "reward_workzone_cone_collision_penalty": "reward_obstacle_collision_penalty",
    "reward_workzone_box_entry_penalty": "reward_keepout_entry_penalty",
    "reward_workzone_speed_penalty_enable": "reward_zone_speed_penalty_enable",
    "reward_workzone_speed_penalty_beta": "reward_zone_speed_penalty_beta",
    "cone_randomize_num": "obstacle_randomize_num",
}


def find_renamed_keys(cfg: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Return ``(section, old_key, new_key)`` for every stale key in *cfg*.

    Walks one level of sections, which matches the flat ``section: {key: value}``
    shape of the SceneFactory config files.
    """
    found: list[tuple[str, str, str]] = []
    for section, payload in (cfg or {}).items():
        if not isinstance(payload, Mapping):
            continue
        for old, new in RENAMED_CONFIG_KEYS.items():
            if old in payload:
                found.append((str(section), old, new))
    return found


def warn_on_renamed_keys(cfg: Mapping[str, Any], *, config_path: str = "") -> list[tuple[str, str, str]]:
    """Print a loud warning for each stale key.  Returns what was found."""
    found = find_renamed_keys(cfg)
    if not found:
        return found
    where = f" in {config_path}" if config_path else ""
    print(
        f"\n[WARN][SceneFactory] {len(found)} config key(s){where} use names that no "
        "longer exist. They are being IGNORED and the defaults apply instead:",
        flush=True,
    )
    for section, old, new in found:
        print(f"    {section}.{old}  ->  {section}.{new}", flush=True)
    print(
        "  These were renamed when obstacles, keep-out regions and zone speed "
        "limits became first-class backbone concepts rather than workzone ones.\n",
        flush=True,
    )
    return found
