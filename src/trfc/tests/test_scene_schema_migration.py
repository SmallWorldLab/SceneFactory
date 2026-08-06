"""The workzone -> zones scene migration must be lossless and idempotent."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "migrate_scene_schema",
    Path(__file__).resolve().parents[3] / "scripts" / "migrate_scene_schema.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
migrate_scene, SF_VERSION = _mod.migrate_scene, _mod.SF_VERSION


def _waymo_style():
    return {
        "meta": {"id": "s1"},
        "road": {"polylines": []},
        "agents": {},
        "workzone": {
            "cones": [{"x": 1.0, "y": 2.0, "z": 0.0}],
            "forbidden_boxes": [{"cx": 1.0, "cy": 2.0, "host_polyline_id": 42}],
        },
    }


def _synthetic_style():
    s = _waymo_style()
    s["workzone"].update({
        "speed_limit_mps": 11.0,
        "workzone_length_m": 21.5,
        "workzone_center_x": 0.0,
        "cone_cfg": {"taper_length_m": 32.9, "lane_width_m": 3.9},
    })
    return s


def test_generic_payload_moves_to_zones():
    out, changed = migrate_scene(_waymo_style())
    assert changed
    assert "workzone" not in out
    assert out["zones"]["obstacles"] == [{"x": 1.0, "y": 2.0, "z": 0.0}]
    assert out["zones"]["keepout_boxes"][0]["host_polyline_id"] == 42
    assert out["sf_version"] == SF_VERSION


def test_domain_provenance_moves_to_the_workzone_extension():
    out, _ = migrate_scene(_synthetic_style())
    ext = out["extensions"]["workzone"]
    assert ext["length_m"] == 21.5
    assert ext["center_x"] == 0.0
    assert ext["cone_cfg"]["taper_length_m"] == 32.9
    # the generic speed limit stays generic
    assert out["zones"]["speed_limit_mps"] == 11.0
    assert "speed_limit_mps" not in ext


def test_migration_is_idempotent():
    once, _ = migrate_scene(_synthetic_style())
    twice, changed = migrate_scene(dict(once))
    assert changed is False
    assert twice == once


def test_nothing_is_silently_dropped():
    scene = _waymo_style()
    scene["workzone"]["some_future_key"] = {"a": 1}
    out, _ = migrate_scene(scene)
    # unrecognised keys are workzone-specific by definition -> extension, not bin
    assert out["extensions"]["workzone"]["some_future_key"] == {"a": 1}


def test_scene_without_a_workzone_block_is_only_stamped():
    scene = {"meta": {}, "road": {"polylines": []}, "agents": {}}
    out, changed = migrate_scene(dict(scene))
    assert changed
    assert out["sf_version"] == SF_VERSION
    assert "zones" not in out and "extensions" not in out
    _, again = migrate_scene(out)
    assert again is False


def test_null_workzone_block_does_not_crash():
    scene = {"meta": {}, "road": {}, "agents": {}, "workzone": None}
    out, changed = migrate_scene(scene)
    assert changed
    assert "workzone" not in out
    assert out["sf_version"] == SF_VERSION
