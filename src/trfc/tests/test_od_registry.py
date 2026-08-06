"""A research package must be able to add an OD mode without editing the env."""

import pytest

from src.scene_factory_od_registry import (
    ODRequest,
    available_od_modes,
    get_od_mode,
    register_od_mode,
    unregister_od_mode,
)


@pytest.fixture
def clean_mode():
    name = "_test_mode"
    unregister_od_mode(name)
    yield name
    unregister_od_mode(name)


def test_unknown_mode_is_not_registered():
    assert get_od_mode("definitely_not_a_mode") is None


def test_register_then_lookup(clean_mode):
    def sampler(request):
        return [request.env_id]

    register_od_mode(clean_mode, sampler)
    assert get_od_mode(clean_mode) is sampler
    assert clean_mode in available_od_modes()


def test_double_registration_is_refused(clean_mode):
    register_od_mode(clean_mode, lambda r: [])
    # two packages claiming one name would make behaviour depend on import order
    with pytest.raises(ValueError, match="already registered"):
        register_od_mode(clean_mode, lambda r: [])


def test_override_is_allowed_when_explicit(clean_mode):
    register_od_mode(clean_mode, lambda r: ["first"])
    register_od_mode(clean_mode, lambda r: ["second"], override=True)
    assert get_od_mode(clean_mode)(None) == ["second"]


def test_empty_name_is_rejected():
    with pytest.raises(ValueError):
        register_od_mode("   ", lambda r: [])


def test_request_carries_what_a_sampler_needs(clean_mode):
    seen = {}

    def sampler(request: ODRequest):
        seen.update(
            scene=request.scene_cfg, n=request.num_agents, seed=request.seed,
            env_id=request.env_id, cfg=request.cfg,
        )
        return ["ok"]

    register_od_mode(clean_mode, sampler)
    out = get_od_mode(clean_mode)(
        ODRequest(scene_cfg={"a": 1}, num_agents=8, bounds_size_m=200.0,
                  seed=42, env_id=3, cfg=object(), env=None)
    )
    assert out == ["ok"]
    assert seen["n"] == 8 and seen["seed"] == 42 and seen["env_id"] == 3
