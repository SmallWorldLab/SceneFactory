"""Registry of origin-destination samplers.

An OD sampler decides where each agent starts and where it is trying to get to.
The backbone ships the general ones -- follow a lane centre, merge across a
multi-lane closure -- but a research line usually wants its own: a workzone
approach convoy, a parking manoeuvre, whatever the study is about.

Before this, adding a mode meant editing an ``if mode == ...`` chain inside a
6,500-line environment, which is the mechanism that pulled domain vocabulary
into the backbone in the first place.  Registering instead means the backbone
never learns the name of anyone's research.

Usage from a research package::

    from src.scene_factory_od_registry import register_od_mode

    def sample_my_mode(request):
        ...
        return samples          # list of per-agent (start, goal) records

    register_od_mode("my_mode", sample_my_mode)

then set ``random_od_mode: my_mode`` in the config.  Registration happens on
import, so the package defining the mode must be imported before the env
resets -- importing it in the training entry point is enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class ODRequest:
    """Everything a sampler needs to place one world's agents.

    ``cfg`` and ``env`` are passed so a mode can read its own config fields and
    use environment helpers (per-map capacity, for instance) without the
    backbone having to know which fields belong to which mode.
    """

    scene_cfg: dict
    num_agents: int
    bounds_size_m: float
    seed: int
    env_id: int
    cfg: Any
    env: Any


class ODSampler(Protocol):
    def __call__(self, request: ODRequest) -> list: ...


_REGISTRY: dict[str, ODSampler] = {}


def register_od_mode(name: str, sampler: ODSampler, *, override: bool = False) -> None:
    """Make ``name`` usable as ``random_od_mode``.

    Refuses to shadow an existing mode unless ``override=True``.  Two packages
    silently claiming the same name would make which sampler runs depend on
    import order, which is exactly the kind of bug that only shows up as
    inexplicable spawn behaviour.
    """
    key = str(name).strip()
    if not key:
        raise ValueError("OD mode name must be a non-empty string")
    if key in _REGISTRY and not override:
        raise ValueError(
            f"OD mode {key!r} is already registered. Pass override=True if that is intended."
        )
    _REGISTRY[key] = sampler


def unregister_od_mode(name: str) -> None:
    """Remove a mode.  Mainly for tests."""
    _REGISTRY.pop(str(name).strip(), None)


def get_od_mode(name: str) -> ODSampler | None:
    """Return the sampler for ``name``, or None if it is a built-in / unknown."""
    return _REGISTRY.get(str(name).strip())


def available_od_modes() -> list[str]:
    """Registered mode names, sorted.  Built-ins are not listed here."""
    return sorted(_REGISTRY)
