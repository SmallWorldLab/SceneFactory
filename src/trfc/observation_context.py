"""Road and weather context encodings for policy observations."""

from __future__ import annotations

from typing import Sequence


ROAD_TYPE_ORDER: tuple[str, str, str] = ("AC", "SMA", "OGFC")
DEFAULT_WATER_FILM_NORM_MM = 1.0


def canonicalize_road_type(road_type: str | None) -> str | None:
    if not isinstance(road_type, str):
        return None
    text = road_type.strip().upper()
    return text or None


def encode_weather_context(
    *,
    water_film_mm: float | None,
    road_type: str | None,
    water_film_norm_mm: float = DEFAULT_WATER_FILM_NORM_MM,
) -> list[float]:
    """Encode water-film depth plus one-hot road type."""

    water = float(water_film_mm or 0.0)
    if water_film_norm_mm > 0.0:
        water = water / float(water_film_norm_mm)

    road = canonicalize_road_type(road_type)
    one_hot = [0.0, 0.0, 0.0]
    if road in ROAD_TYPE_ORDER:
        one_hot[ROAD_TYPE_ORDER.index(road)] = 1.0

    return [float(water), *one_hot]


def weather_context_dim() -> int:
    return 1 + len(ROAD_TYPE_ORDER)


__all__: Sequence[str] = [
    "ROAD_TYPE_ORDER",
    "DEFAULT_WATER_FILM_NORM_MM",
    "canonicalize_road_type",
    "encode_weather_context",
    "weather_context_dim",
]
