"""A config written against the old workzone-flavoured key names must be loud."""

from src.scene_factory_config_compat import (
    RENAMED_CONFIG_KEYS,
    find_renamed_keys,
    warn_on_renamed_keys,
)


def test_clean_config_reports_nothing():
    cfg = {"reward": {"reward_obstacle_penalty_enable": True}, "env": {"num_envs": 4}}
    assert find_renamed_keys(cfg) == []


def test_stale_key_is_detected_with_its_replacement():
    cfg = {"reward": {"reward_workzone_cone_penalty_enable": True}}
    found = find_renamed_keys(cfg)
    assert found == [("reward", "reward_workzone_cone_penalty_enable",
                      "reward_obstacle_penalty_enable")]


def test_every_renamed_key_is_detected():
    # One section per key so a typo in the table cannot hide.
    cfg = {f"s{i}": {old: 1} for i, old in enumerate(RENAMED_CONFIG_KEYS)}
    assert len(find_renamed_keys(cfg)) == len(RENAMED_CONFIG_KEYS)


def test_non_mapping_sections_are_skipped():
    cfg = {"reward": None, "scene_factory": ["a", "b"], "env": 3}
    assert find_renamed_keys(cfg) == []


def test_warn_returns_findings_and_prints(capsys):
    cfg = {"reward": {"reward_workzone_box_entry_penalty": -5.0}}
    found = warn_on_renamed_keys(cfg, config_path="configs/x.yaml")
    assert len(found) == 1
    out = capsys.readouterr().out
    assert "reward_keepout_entry_penalty" in out
    assert "configs/x.yaml" in out


def test_no_renamed_key_maps_to_itself():
    for old, new in RENAMED_CONFIG_KEYS.items():
        assert old != new
