"""Programmatic wet-road friction estimation based on Zhao et al. (2024).

This module implements the modified average lumped LuGre (ALL) tire-pavement
friction model from:

    Zhao, L., Zhao, H., Cai, J. (2024)
    "Tire-pavement friction modeling considering pavement texture and water film"
    International Journal of Transportation Science and Technology 14 (2024) 99-109
    doi:10.1016/j.ijtst.2023.04.001

The intended abstraction boundary is:

1. Use this module offline or in your application code.
2. Feed it water-film thickness plus road / vehicle parameters.
3. Receive effective scalar friction coefficients.
4. Pass those scalars into Isaac Sim / PhysX material friction.

This is not an HTTP or JSON API. The primary interface is:

    estimate_friction(FrictionInput(...)) -> FrictionEstimate

and the module also exposes a CLI:

    python -m src.trfc --road-type AC --water-film-mm 0.2
    python -m src.trfc --demo

Slip and steady-state
---------------------
The paper uses slip-rate-driven friction curves. For this program:

    relative_speed = abs(slip) * reference_speed

The modified ALL model evolves the average bristle deflection ``z_bar`` via:

    dz_bar/dt = v_r - theta * Y_R * [
        (sigma0 * |v_r| / g(v_r)) * z_bar + K * |w_r| * z_bar
    ]

and computes friction as:

    mu = (theta * Y_R - Y_F) * [
        theta * Y_R * sigma0 * z_bar
        + sigma1 * dz_bar/dt
        + sigma2 * v_r
    ]

where:

- ``theta`` is the texture influence coefficient.
- ``Y_R`` is the contact patch length ratio under hydrodynamic lift.
- ``Y_F`` is the hydrodynamic-force ratio under wet conditions.
- ``g(v_r)`` is the Stribeck-like steady-state function.

Notes on paper parameters
-------------------------
The PDF's Table 4 reports ``sigma0``, ``sigma2``, ``mu_c``, ``mu_s``, and the
``b1..b4`` coefficients used in ``v_s(h)``. It does not list ``sigma1`` in the
table even though it appears in the final equation, so this implementation sets
``sigma1 = 0.0`` explicitly.

Water-film input
----------------
The paper treats water-film thickness ``h`` as a known model input. To stay
inside the paper's scope, ``estimate_friction()`` requires ``water_film_mm`` and
does not infer it from precipitation. A separate helper remains available for
external use, but it is not part of the friction-estimation pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_K_RAIN_MM_PER_MMPH = 0.05
DEFAULT_MAX_FILM_MM = 3.0
DEFAULT_DRAINAGE_FACTOR = 1.0
DEFAULT_V_REF_MPS = 13.89
DEFAULT_S_REF_STATIC = 0.15
DEFAULT_S_REF_DYNAMIC = 0.8
DEFAULT_DT = 1e-3
DEFAULT_T_MAX = 0.5
DEFAULT_EPS = 1e-7
DEFAULT_EPS_Z = 1e-9


@dataclass(frozen=True)
class RoadSurfaceParameters:
    road_type: str | None
    theta_texture: float
    texture_amplitude_mm: float


ROAD_SURFACE_PRESETS = {
    "AC": RoadSurfaceParameters("AC", theta_texture=1.00, texture_amplitude_mm=0.65),
    "SMA": RoadSurfaceParameters("SMA", theta_texture=1.09, texture_amplitude_mm=0.80),
    "OGFC": RoadSurfaceParameters("OGFC", theta_texture=1.21, texture_amplitude_mm=1.08),
}


@dataclass(frozen=True)
class AllWetRoadParameters:
    """Known and calibrated parameters from Zhao et al. (2024)."""

    tire_model_id: str = "zhao2024_modified_all_default"
    params_source: str = (
        "Zhao et al. (2024), Table 4 and Table 5, "
        "Tire-pavement friction modeling considering pavement texture and water film"
    )

    # Vehicle example used in the paper's numerical study (Table 5).
    normal_load_n: float = 3469.0
    tire_pressure_pa: float = 200000.0
    tire_width_m: float = 0.205
    contact_patch_length_m: float = 0.164
    contact_patch_width_m: float = 0.125
    contact_patch_area_m2: float | None = None
    wheel_radius_m: float = 0.316
    tread_radius_m: float = 0.103

    # Common physical parameters / known parameters from Table 4.
    water_density_kg_per_m3: float = 1.05e3
    water_viscosity_pa_s: float = 1.005e-3
    minimum_spacing_m: float = 0.008e-3
    h0_ratio: float = 0.01

    # Calibrated model parameters from Table 4.
    sigma0: float = 310.78
    sigma1: float = 0.0
    sigma2: float = 0.0
    mu_c: float = 0.46
    mu_s: float = 1.51
    alpha: float = 0.5
    b1: float = 4.8916
    b2: float = -7.91
    b3: float = 3.01
    b4: float = 3.40


DEFAULT_TIRE_MODELS = {
    "zhao2024_modified_all_default": AllWetRoadParameters(),
    "zhao2024_griptester": AllWetRoadParameters(
        tire_model_id="zhao2024_griptester",
        params_source=(
            "Zhao et al. (2024), Table 4 calibration / GripTester geometry, "
            "Tire-pavement friction modeling considering pavement texture and water film"
        ),
        normal_load_n=210.0,
        tire_pressure_pa=140000.0,
        tire_width_m=0.031,
        contact_patch_length_m=0.046,
        contact_patch_width_m=0.031,
        wheel_radius_m=0.127,
        tread_radius_m=0.028,
    ),
}

PAPER_TABLE2_SPEEDS_KMH = [10, 20, 30, 40, 50, 60, 70, 80, 90]
PAPER_TABLE2_AVG_MU = [0.81, 0.79, 0.79, 0.75, 0.73, 0.75, 0.67, 0.64, 0.61]
PAPER_TABLE3_WATER_MM = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
PAPER_TABLE3_AVG_MU = [0.81, 0.78, 0.77, 0.74, 0.73, 0.72, 0.70, 0.68, 0.66, 0.62]


@dataclass(frozen=True)
class FrictionInput:
    """Programmatic input to the wet-road friction estimator."""

    road_type: str | None = "AC"
    texture_coeff: float | None = None
    texture_amplitude_mm: float | None = None
    precip_type: str | None = None
    precip_intensity_mmph: float | None = None
    water_film_mm: float | None = None
    tire_model_id: str = "zhao2024_modified_all_default"
    reference_speed_mps: float = DEFAULT_V_REF_MPS
    slip_static: float = DEFAULT_S_REF_STATIC
    slip_dynamic: float = DEFAULT_S_REF_DYNAMIC
    dt: float = DEFAULT_DT
    t_max: float = DEFAULT_T_MAX
    eps: float = DEFAULT_EPS
    eps_z: float = DEFAULT_EPS_Z
    drainage_factor: float = DEFAULT_DRAINAGE_FACTOR
    k_rain_mm_per_mmph: float = DEFAULT_K_RAIN_MM_PER_MMPH
    h_max_mm: float = DEFAULT_MAX_FILM_MM
    mu_eff_mode: str = "static"


@dataclass(frozen=True)
class MuComputationDetails:
    mu: float
    road_type: str | None
    theta_texture: float
    texture_amplitude_mm: float
    water_film_mm: float
    water_film_m: float
    h0_m: float
    reference_speed_mps: float
    slip: float
    relative_speed_mps: float
    wheel_surface_speed_mps: float
    stribeck_speed_mps: float
    g_vr: float
    y_r: float
    y_f: float
    contact_factor: float
    contact_patch_area_m2: float
    k_value: float
    z_bar: float
    dz_dt: float
    decay_rate: float
    converged: bool
    num_steps: int
    hydroplaning: bool
    tire_model_id: str
    params_source: str


@dataclass(frozen=True)
class FrictionEstimate:
    mu_static: float
    mu_dynamic: float
    mu_eff: float
    road_type: str | None
    water_film_mm: float
    theta_texture: float
    texture_amplitude_mm: float
    static: MuComputationDetails
    dynamic: MuComputationDetails
    tire_model_id: str
    params_source: str


def _resolve_params(
    params: AllWetRoadParameters | Mapping[str, Any] | None,
    tire_model_id: str | None = None,
) -> AllWetRoadParameters:
    if isinstance(params, AllWetRoadParameters):
        return params
    if isinstance(params, Mapping):
        return AllWetRoadParameters(**params)

    model_id = tire_model_id or "zhao2024_modified_all_default"
    try:
        return DEFAULT_TIRE_MODELS[model_id]
    except KeyError as exc:
        known = ", ".join(sorted(DEFAULT_TIRE_MODELS))
        raise ValueError(
            f"Unknown tire_model_id={model_id!r}. Known values: {known}"
        ) from exc


def _coerce_input(request: FrictionInput | Mapping[str, Any]) -> FrictionInput:
    if isinstance(request, FrictionInput):
        return request
    if isinstance(request, Mapping):
        return FrictionInput(
            road_type=request.get("road_type", "AC"),
            texture_coeff=request.get("texture_coeff"),
            texture_amplitude_mm=request.get("texture_amplitude_mm"),
            precip_type=request.get("precip_type"),
            precip_intensity_mmph=request.get("precip_intensity_mmph"),
            water_film_mm=request.get("water_film_mm"),
            tire_model_id=request.get("tire_model_id", "zhao2024_modified_all_default"),
            reference_speed_mps=float(
                request.get("reference_speed_mps", request.get("v_ref_mps", DEFAULT_V_REF_MPS))
            ),
            slip_static=float(
                request.get("slip_static", request.get("s_ref_static", DEFAULT_S_REF_STATIC))
            ),
            slip_dynamic=float(
                request.get("slip_dynamic", request.get("s_ref_dynamic", DEFAULT_S_REF_DYNAMIC))
            ),
            dt=float(request.get("dt", DEFAULT_DT)),
            t_max=float(request.get("t_max", DEFAULT_T_MAX)),
            eps=float(request.get("eps", DEFAULT_EPS)),
            eps_z=float(request.get("eps_z", DEFAULT_EPS_Z)),
            drainage_factor=float(request.get("drainage_factor", DEFAULT_DRAINAGE_FACTOR)),
            k_rain_mm_per_mmph=float(
                request.get("k_rain_mm_per_mmph", DEFAULT_K_RAIN_MM_PER_MMPH)
            ),
            h_max_mm=float(request.get("h_max_mm", DEFAULT_MAX_FILM_MM)),
            mu_eff_mode=str(request.get("mu_eff_mode", "static")),
        )
    raise TypeError(f"Unsupported request type: {type(request)!r}")


def water_film_from_precip(
    *,
    precip_type: str | None = None,
    precip_intensity_mmph: float | None = None,
    drainage_factor: float = DEFAULT_DRAINAGE_FACTOR,
    k_rain_mm_per_mmph: float = DEFAULT_K_RAIN_MM_PER_MMPH,
    h_max_mm: float = DEFAULT_MAX_FILM_MM,
) -> float:
    """Estimate water-film thickness from precipitation.

    The v1 helper mapping requested by the user is used directly:

        h_w_mm = clamp(k_rain * intensity * drainage_factor, 0, h_max_mm)
    """

    if precip_intensity_mmph is None:
        raise ValueError(
            "precip_intensity_mmph is required when water_film_mm is not provided"
        )
    if not 0.0 <= drainage_factor <= 1.0:
        raise ValueError(
            f"drainage_factor must be in [0, 1], got {drainage_factor!r}"
        )

    precip_kind = (precip_type or "rain").strip().lower()
    if precip_kind not in {"rain", "drizzle", "shower", "unknown"}:
        precip_kind = "unknown"

    intensity = max(float(precip_intensity_mmph), 0.0)
    if k_rain_mm_per_mmph < 0.0 or h_max_mm < 0.0:
        raise ValueError("k_rain_mm_per_mmph and h_max_mm must be non-negative")

    return min(max(k_rain_mm_per_mmph * intensity * drainage_factor, 0.0), h_max_mm)


def texture_coeff_from_road(
    road_type: str | None = None,
    texture_coeff: float | None = None,
) -> float:
    """Return the texture influence coefficient theta."""

    return _resolve_road_surface(
        road_type=road_type,
        texture_coeff=texture_coeff,
        texture_amplitude_mm=None,
    ).theta_texture


def texture_amplitude_from_road(
    road_type: str | None = None,
    texture_amplitude_mm: float | None = None,
) -> float:
    """Return pavement texture amplitude in millimeters."""

    return _resolve_road_surface(
        road_type=road_type,
        texture_coeff=None,
        texture_amplitude_mm=texture_amplitude_mm,
    ).texture_amplitude_mm


def _resolve_road_surface(
    *,
    road_type: str | None,
    texture_coeff: float | None,
    texture_amplitude_mm: float | None,
) -> RoadSurfaceParameters:
    canonical_road_type = road_type.strip().upper() if isinstance(road_type, str) else None
    preset = ROAD_SURFACE_PRESETS.get(canonical_road_type, ROAD_SURFACE_PRESETS["AC"])

    theta_texture = float(texture_coeff) if texture_coeff is not None else preset.theta_texture
    texture_amplitude = (
        float(texture_amplitude_mm)
        if texture_amplitude_mm is not None
        else preset.texture_amplitude_mm
    )

    if theta_texture <= 0.0:
        raise ValueError(f"texture_coeff must be positive, got {theta_texture!r}")
    if texture_amplitude <= 0.0:
        raise ValueError(
            f"texture_amplitude_mm must be positive, got {texture_amplitude!r}"
        )

    return RoadSurfaceParameters(
        road_type=canonical_road_type or preset.road_type,
        theta_texture=theta_texture,
        texture_amplitude_mm=texture_amplitude,
    )


def _stribeck_speed_mps(h_w_m: float, params: AllWetRoadParameters) -> float:
    return max(params.b1 * math.exp(1000.0 * params.b2 * h_w_m + params.b3) + params.b4, 1e-12)


def _g_vr(relative_speed_mps: float, stribeck_speed_mps: float, params: AllWetRoadParameters) -> float:
    exponent = -abs(relative_speed_mps / max(stribeck_speed_mps, 1e-12)) ** params.alpha
    return params.mu_c + (params.mu_s - params.mu_c) * math.exp(exponent)


def _contact_patch_area_m2(params: AllWetRoadParameters) -> float:
    if params.contact_patch_area_m2 is not None:
        area = float(params.contact_patch_area_m2)
    else:
        # The paper uses A in the denominator but does not define it explicitly.
        # Default to the nominal footprint area from the reported patch length/width.
        area = float(params.contact_patch_length_m) * float(params.contact_patch_width_m)
    if area <= 0.0:
        raise ValueError(f"contact_patch_area_m2 must be positive, got {area!r}")
    return area


def _contact_patch_length_ratio(
    vehicle_speed_mps: float,
    water_film_m: float,
    texture_amplitude_m: float,
    params: AllWetRoadParameters,
) -> float:
    if water_film_m <= 0.0:
        return 1.0

    h0_m = params.h0_ratio * water_film_m
    if h0_m <= 0.0:
        return 1.0

    # Printed in the paper as: 12 v γ r0^2 / (A π p L) * (...)
    # The paper does not define A explicitly, so use either an explicit override
    # or the nominal footprint area implied by the reported patch length/width.
    contact_patch_area_m2 = _contact_patch_area_m2(params)
    prefactor = (
        12.0
        * vehicle_speed_mps
        * params.water_viscosity_pa_s
        * params.tread_radius_m
        * params.tread_radius_m
        / (
            contact_patch_area_m2
            * math.pi
            * params.tire_pressure_pa
            * params.contact_patch_length_m
        )
    )

    term_dry = 1.0 / (
        params.minimum_spacing_m * params.minimum_spacing_m
        + 2.0 * texture_amplitude_m * params.minimum_spacing_m
    )
    term_wet = 1.0 / (
        h0_m * h0_m
        + 2.0 * texture_amplitude_m * h0_m
    )

    y_r = 1.0 - prefactor * (term_dry - term_wet)
    return min(1.0, max(y_r, 0.0))


def _hydrodynamic_force_ratio(
    vehicle_speed_mps: float,
    water_film_m: float,
    params: AllWetRoadParameters,
) -> float:
    if water_film_m <= 0.0:
        return 0.0

    l_over_2r = params.contact_patch_length_m / (2.0 * params.wheel_radius_m)
    h_over_r = water_film_m / params.wheel_radius_m
    inner = (
        l_over_2r * l_over_2r
        - h_over_r * h_over_r
        + 2.0 * h_over_r * math.sqrt(max(1.0 - l_over_2r * l_over_2r, 0.0))
    )
    hydrodynamic_term = math.sqrt(max(inner, 0.0)) - l_over_2r
    y_f = (
        params.water_density_kg_per_m3
        * params.tire_width_m
        * params.wheel_radius_m
        * vehicle_speed_mps
        * vehicle_speed_mps
        / (3.0 * params.normal_load_n)
        * hydrodynamic_term
    )
    return min(1.0, max(y_f, 0.0))


def compute_mu_all_modified(
    v_ref: float,
    slip: float,
    h_w_mm: float,
    road_type: str | None = None,
    params: AllWetRoadParameters | Mapping[str, Any] | None = None,
    *,
    tire_model_id: str | None = None,
    texture_coeff: float | None = None,
    texture_amplitude_mm: float | None = None,
    dt: float = DEFAULT_DT,
    t_max: float = DEFAULT_T_MAX,
    eps: float = DEFAULT_EPS,
    eps_z: float = DEFAULT_EPS_Z,
) -> MuComputationDetails:
    """Compute steady-state friction for a reference speed / slip pair."""

    resolved_params = _resolve_params(params, tire_model_id=tire_model_id)
    road_surface = _resolve_road_surface(
        road_type=road_type,
        texture_coeff=texture_coeff,
        texture_amplitude_mm=texture_amplitude_mm,
    )

    if dt <= 0.0 or t_max <= 0.0 or eps <= 0.0 or eps_z <= 0.0:
        raise ValueError("dt, t_max, eps, and eps_z must be positive")

    reference_speed_mps = abs(float(v_ref))
    slip_ratio = float(slip)
    water_film_mm = max(float(h_w_mm), 0.0)
    water_film_m = water_film_mm * 1e-3
    h0_m = resolved_params.h0_ratio * water_film_m
    texture_amplitude_m = road_surface.texture_amplitude_mm * 1e-3

    relative_speed_signed = slip_ratio * reference_speed_mps
    relative_speed_mps = abs(relative_speed_signed)
    # In Eq. (1) the paper defines v_r = w_r - v and labels v_r in m/s, so w_r
    # is the circumferential linear speed at the tread, not angular speed in rad/s.
    # Under the slip definition s = (R*omega - v) / v, this gives:
    #   w_r = R*omega = v * (1 + s)
    wheel_surface_speed_mps = abs(reference_speed_mps + relative_speed_signed)

    y_r = _contact_patch_length_ratio(
        vehicle_speed_mps=reference_speed_mps,
        water_film_m=water_film_m,
        texture_amplitude_m=texture_amplitude_m,
        params=resolved_params,
    )
    y_f = _hydrodynamic_force_ratio(
        vehicle_speed_mps=reference_speed_mps,
        water_film_m=water_film_m,
        params=resolved_params,
    )
    contact_patch_area_m2 = _contact_patch_area_m2(resolved_params)

    stribeck_speed_mps = _stribeck_speed_mps(water_film_m, resolved_params)
    g_vr = _g_vr(relative_speed_mps, stribeck_speed_mps, resolved_params)
    # Canudas de Wit (2003) gives the ALL approximation as K(t) = K0 / L with
    # K0 = 7/6. Together with the paper's linear-speed definition of w_r, the
    # K(t)|w_r| term is dimensionally consistent only if K has units 1/m.
    k_value = 7.0 / (6.0 * resolved_params.contact_patch_length_m)

    theta_y_r = road_surface.theta_texture * y_r
    decay_rate = theta_y_r * (
        (resolved_params.sigma0 * relative_speed_mps / max(g_vr, 1e-12))
        + k_value * wheel_surface_speed_mps
    )

    z_bar = 0.0
    dz_dt = 0.0
    converged = False
    num_steps = max(1, int(math.ceil(t_max / dt)))

    for step in range(num_steps):
        dz_dt = relative_speed_mps - decay_rate * z_bar
        if decay_rate > 0.0:
            decay = math.exp(-decay_rate * dt)
            z_next = z_bar * decay + (relative_speed_mps / decay_rate) * (1.0 - decay)
        else:
            z_next = z_bar + dt * dz_dt

        if abs(dz_dt) < eps or abs(z_next - z_bar) < eps_z:
            z_bar = z_next
            converged = True
            break
        z_bar = z_next
    else:
        step = num_steps - 1

    inner_mu = (
        theta_y_r * resolved_params.sigma0 * z_bar
        + resolved_params.sigma1 * dz_dt
        + resolved_params.sigma2 * relative_speed_mps
    )
    contact_factor = max(road_surface.theta_texture * y_r - y_f, 0.0)
    mu = max(contact_factor * inner_mu, 0.0)
    hydroplaning = contact_factor <= 0.0 or y_r <= 0.0

    return MuComputationDetails(
        mu=float(mu),
        road_type=road_surface.road_type,
        theta_texture=float(road_surface.theta_texture),
        texture_amplitude_mm=float(road_surface.texture_amplitude_mm),
        water_film_mm=float(water_film_mm),
        water_film_m=float(water_film_m),
        h0_m=float(h0_m),
        reference_speed_mps=float(reference_speed_mps),
        slip=float(slip_ratio),
        relative_speed_mps=float(relative_speed_mps),
        wheel_surface_speed_mps=float(wheel_surface_speed_mps),
        stribeck_speed_mps=float(stribeck_speed_mps),
        g_vr=float(g_vr),
        y_r=float(y_r),
        y_f=float(y_f),
        contact_factor=float(contact_factor),
        contact_patch_area_m2=float(contact_patch_area_m2),
        k_value=float(k_value),
        z_bar=float(z_bar),
        dz_dt=float(dz_dt),
        decay_rate=float(decay_rate),
        converged=bool(converged),
        num_steps=int(step + 1),
        hydroplaning=bool(hydroplaning),
        tire_model_id=resolved_params.tire_model_id,
        params_source=resolved_params.params_source,
    )


def estimate_friction(request: FrictionInput | Mapping[str, Any]) -> FrictionEstimate:
    """Estimate static, dynamic, and effective friction from program input."""

    inputs = _coerce_input(request)
    params = _resolve_params(None, tire_model_id=inputs.tire_model_id)

    if inputs.water_film_mm is None:
        raise ValueError(
            "water_film_mm must be provided. "
            "Precipitation-to-water-film conversion is intentionally disabled "
            "in the friction-estimation pipeline."
        )
    water_film_mm = float(inputs.water_film_mm)
    if water_film_mm < 0.0:
        raise ValueError(f"water_film_mm must be non-negative, got {water_film_mm!r}")

    static_details = compute_mu_all_modified(
        v_ref=inputs.reference_speed_mps,
        slip=inputs.slip_static,
        h_w_mm=water_film_mm,
        road_type=inputs.road_type,
        params=params,
        texture_coeff=inputs.texture_coeff,
        texture_amplitude_mm=inputs.texture_amplitude_mm,
        dt=inputs.dt,
        t_max=inputs.t_max,
        eps=inputs.eps,
        eps_z=inputs.eps_z,
    )
    dynamic_details = compute_mu_all_modified(
        v_ref=inputs.reference_speed_mps,
        slip=inputs.slip_dynamic,
        h_w_mm=water_film_mm,
        road_type=inputs.road_type,
        params=params,
        texture_coeff=inputs.texture_coeff,
        texture_amplitude_mm=inputs.texture_amplitude_mm,
        dt=inputs.dt,
        t_max=inputs.t_max,
        eps=inputs.eps,
        eps_z=inputs.eps_z,
    )

    mu_eff_mode = inputs.mu_eff_mode.strip().lower()
    mu_eff = dynamic_details.mu if mu_eff_mode == "dynamic" else static_details.mu

    return FrictionEstimate(
        mu_static=float(static_details.mu),
        mu_dynamic=float(dynamic_details.mu),
        mu_eff=float(mu_eff),
        road_type=static_details.road_type,
        water_film_mm=float(water_film_mm),
        theta_texture=float(static_details.theta_texture),
        texture_amplitude_mm=float(static_details.texture_amplitude_mm),
        static=static_details,
        dynamic=dynamic_details,
        tire_model_id=params.tire_model_id,
        params_source=params.params_source,
    )


def _format_single_estimate(estimate: FrictionEstimate) -> str:
    return "\n".join(
        [
            f"road_type: {estimate.road_type}",
            f"water_film_mm: {estimate.water_film_mm:.3f}",
            f"theta_texture: {estimate.theta_texture:.3f}",
            f"mu_static: {estimate.mu_static:.4f}",
            f"mu_dynamic: {estimate.mu_dynamic:.4f}",
            f"mu_eff: {estimate.mu_eff:.4f}",
            f"static_y_r: {estimate.static.y_r:.4f}",
            f"static_y_f: {estimate.static.y_f:.4f}",
            f"static_v_s_mps: {estimate.static.stribeck_speed_mps:.4f}",
        ]
    )


def _format_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> str:
    materialized_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in materialized_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    lines = [render(headers), render(["-" * width for width in widths])]
    lines.extend(render(row) for row in materialized_rows)
    return "\n".join(lines)


def _rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        return 0.0
    return math.sqrt(
        sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
    )


def run_demo() -> str:
    """Return a multi-scenario demo string showing friction trends."""

    water_rows = []
    for water_film_mm in (0.0, 0.1, 0.2, 0.4, 0.6):
        estimate = estimate_friction(
            FrictionInput(
                road_type="AC",
                water_film_mm=water_film_mm,
            )
        )
        water_rows.append(
            [
                f"{estimate.water_film_mm:.3f}",
                f"{estimate.mu_static:.4f}",
                f"{estimate.mu_dynamic:.4f}",
            ]
        )

    road_rows = []
    for road_type in ("AC", "SMA", "OGFC"):
        estimate = estimate_friction(
            FrictionInput(
                road_type=road_type,
                water_film_mm=0.2,
            )
        )
        road_rows.append(
            [
                road_type,
                f"{estimate.texture_amplitude_mm:.2f}",
                f"{estimate.theta_texture:.2f}",
                f"{estimate.mu_static:.4f}",
                f"{estimate.mu_dynamic:.4f}",
            ]
        )

    return "\n\n".join(
        [
            "Water-film sweep at road_type=AC, v_ref=13.89 m/s",
            _format_table(
                headers=("water_mm", "mu_static", "mu_dynamic"),
                rows=water_rows,
            ),
            "Road-type sweep at water film = 0.20 mm, v_ref=13.89 m/s",
            _format_table(
                headers=(
                    "road_type",
                    "texture_mm",
                    "theta",
                    "mu_static",
                    "mu_dynamic",
                ),
                rows=road_rows,
            ),
        ]
    )


def _paper_validation_predictions() -> tuple[list[float], list[float]]:
    speed_predictions = []
    for speed_kmh in PAPER_TABLE2_SPEEDS_KMH:
        estimate = estimate_friction(
            FrictionInput(
                road_type="AC",
                texture_coeff=1.0,
                texture_amplitude_mm=0.64,
                water_film_mm=0.25,
                reference_speed_mps=speed_kmh / 3.6,
                slip_static=0.15,
                slip_dynamic=0.8,
                tire_model_id="zhao2024_griptester",
            )
        )
        speed_predictions.append(estimate.mu_static)

    water_predictions = []
    for water_mm in PAPER_TABLE3_WATER_MM:
        estimate = estimate_friction(
            FrictionInput(
                road_type="AC",
                texture_coeff=1.0,
                texture_amplitude_mm=0.64,
                water_film_mm=water_mm,
                reference_speed_mps=50.0 / 3.6,
                slip_static=0.15,
                slip_dynamic=0.8,
                tire_model_id="zhao2024_griptester",
            )
        )
        water_predictions.append(estimate.mu_static)

    return speed_predictions, water_predictions


def paper_validation_report() -> str:
    """Compare the model against the paper's published average curves."""

    speed_predictions, water_predictions = _paper_validation_predictions()
    speed_rmse = _rmse(PAPER_TABLE2_AVG_MU, speed_predictions)
    water_rmse = _rmse(PAPER_TABLE3_AVG_MU, water_predictions)

    speed_rows = [
        [f"{speed}", f"{paper:.2f}", f"{pred:.4f}", f"{pred - paper:+.4f}"]
        for speed, paper, pred in zip(
            PAPER_TABLE2_SPEEDS_KMH, PAPER_TABLE2_AVG_MU, speed_predictions
        )
    ]
    water_rows = [
        [f"{water:.2f}", f"{paper:.2f}", f"{pred:.4f}", f"{pred - paper:+.4f}"]
        for water, paper, pred in zip(
            PAPER_TABLE3_WATER_MM, PAPER_TABLE3_AVG_MU, water_predictions
        )
    ]

    return "\n\n".join(
        [
            "Validation against Zhao et al. (2024) published average curves",
            "Speed sweep, water film = 0.25 mm, GripTester geometry",
            _format_table(
                headers=("speed_kmh", "paper_mu", "model_mu", "delta"),
                rows=speed_rows,
            ),
            f"speed_rmse: {speed_rmse:.4f}",
            "Water-film sweep, speed = 50 km/h, GripTester geometry",
            _format_table(
                headers=("water_mm", "paper_mu", "model_mu", "delta"),
                rows=water_rows,
            ),
            f"water_rmse: {water_rmse:.4f}",
        ]
    )


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _line_plot_elements(
    *,
    x: Sequence[float],
    series: Sequence[tuple[str, str, Sequence[float]]],
    title: str,
    x_label: str,
    y_label: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> str:
    left = x0 + 70
    top = y0 + 30
    plot_width = width - 120
    plot_height = height - 90
    bottom = top + plot_height
    right = left + plot_width

    x_min = min(x)
    x_max = max(x)
    y_values = [value for _, _, values in series for value in values]
    y_min = min(y_values)
    y_max = max(y_values)
    y_pad = max(0.05, 0.1 * (y_max - y_min if y_max > y_min else 1.0))
    y_min = max(0.0, y_min - y_pad)
    y_max = y_max + y_pad

    def sx(value: float) -> float:
        if x_max == x_min:
            return left + plot_width / 2.0
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        if y_max == y_min:
            return bottom - plot_height / 2.0
        return bottom - (value - y_min) / (y_max - y_min) * plot_height

    elements = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + 18}" font-size="16" '
        f'text-anchor="middle">{_svg_escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="white" stroke="#555" stroke-width="1"/>',
    ]

    for tick in range(5):
        y_value = y_min + tick * (y_max - y_min) / 4.0
        y_pixel = sy(y_value)
        elements.append(
            f'<line x1="{left}" y1="{y_pixel:.1f}" x2="{right}" y2="{y_pixel:.1f}" '
            'stroke="#ddd" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y_pixel + 4:.1f}" font-size="11" '
            f'text-anchor="end">{y_value:.2f}</text>'
        )

    for idx, x_value in enumerate(x):
        x_pixel = sx(x_value)
        elements.append(
            f'<line x1="{x_pixel:.1f}" y1="{top}" x2="{x_pixel:.1f}" y2="{bottom}" '
            'stroke="#f0f0f0" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x_pixel:.1f}" y="{bottom + 18}" font-size="11" '
            f'text-anchor="middle">{x_value:g}</text>'
        )

    elements.append(
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height - 8}" font-size="12" '
        f'text-anchor="middle">{_svg_escape(x_label)}</text>'
    )
    elements.append(
        f'<text x="{x0 + 18}" y="{y0 + height / 2:.1f}" font-size="12" '
        f'text-anchor="middle" transform="rotate(-90 {x0 + 18},{y0 + height / 2:.1f})">'
        f'{_svg_escape(y_label)}</text>'
    )

    legend_y = y0 + 20
    legend_x = right - 140
    for idx, (label, color, values) in enumerate(series):
        points = " ".join(f"{sx(xv):.1f},{sy(yv):.1f}" for xv, yv in zip(x, values))
        elements.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>'
        )
        for xv, yv in zip(x, values):
            elements.append(
                f'<circle cx="{sx(xv):.1f}" cy="{sy(yv):.1f}" r="3" fill="{color}"/>'
            )
        ly = legend_y + idx * 16
        elements.append(
            f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 18}" y2="{ly}" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
        elements.append(
            f'<text x="{legend_x + 24}" y="{ly + 4}" font-size="11">'
            f'{_svg_escape(label)}</text>'
        )

    return "\n".join(elements)


def _bar_plot_elements(
    *,
    categories: Sequence[str],
    values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> str:
    left = x0 + 70
    top = y0 + 30
    plot_width = width - 110
    plot_height = height - 90
    bottom = top + plot_height
    right = left + plot_width
    y_max = max(values) * 1.15 if values else 1.0

    def sy(value: float) -> float:
        return bottom - (value / y_max) * plot_height

    elements = [
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + 18}" font-size="16" '
        f'text-anchor="middle">{_svg_escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
        'fill="white" stroke="#555" stroke-width="1"/>',
    ]

    for tick in range(5):
        value = tick * y_max / 4.0
        y_pixel = sy(value)
        elements.append(
            f'<line x1="{left}" y1="{y_pixel:.1f}" x2="{right}" y2="{y_pixel:.1f}" '
            'stroke="#ddd" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y_pixel + 4:.1f}" font-size="11" '
            f'text-anchor="end">{value:.2f}</text>'
        )

    count = max(len(categories), 1)
    slot = plot_width / count
    bar_width = slot * 0.5
    colors = ["#3366cc", "#ff9900", "#109618"]
    for idx, (category, value) in enumerate(zip(categories, values)):
        cx = left + slot * (idx + 0.5)
        x_bar = cx - bar_width / 2.0
        y_bar = sy(value)
        bar_height = bottom - y_bar
        color = colors[idx % len(colors)]
        elements.append(
            f'<rect x="{x_bar:.1f}" y="{y_bar:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{color}" opacity="0.85"/>'
        )
        elements.append(
            f'<text x="{cx:.1f}" y="{bottom + 18}" font-size="11" '
            f'text-anchor="middle">{_svg_escape(category)}</text>'
        )
        elements.append(
            f'<text x="{cx:.1f}" y="{y_bar - 6:.1f}" font-size="11" '
            f'text-anchor="middle">{value:.3f}</text>'
        )

    elements.append(
        f'<text x="{x0 + width / 2:.1f}" y="{y0 + height - 8}" font-size="12" '
        f'text-anchor="middle">{_svg_escape(x_label)}</text>'
    )
    elements.append(
        f'<text x="{x0 + 18}" y="{y0 + height / 2:.1f}" font-size="12" '
        f'text-anchor="middle" transform="rotate(-90 {x0 + 18},{y0 + height / 2:.1f})">'
        f'{_svg_escape(y_label)}</text>'
    )

    return "\n".join(elements)


def write_demo_svg(path: str | Path) -> Path:
    """Write a dependency-free SVG with water-film and road-type trend plots."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    water_x = [0.0, 0.1, 0.2, 0.4, 0.6]
    water_static = []
    for water_film_mm in water_x:
        estimate = estimate_friction(
            FrictionInput(
                road_type="AC",
                water_film_mm=water_film_mm,
            )
        )
        water_static.append(estimate.mu_static)

    road_categories = ["AC", "SMA", "OGFC"]
    road_static = []
    for road_type in road_categories:
        estimate = estimate_friction(
            FrictionInput(
                road_type=road_type,
                water_film_mm=0.2,
            )
        )
        road_static.append(estimate.mu_static)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="520" viewBox="0 0 1200 520">
<rect width="1200" height="520" fill="#fafafa"/>
<text x="600" y="24" font-size="20" text-anchor="middle">TRFC demo trends</text>
{_line_plot_elements(
    x=water_x,
    series=(("mu_static", "#3366cc", water_static),),
    title="Water film vs friction (AC)",
    x_label="water film thickness (mm)",
    y_label="friction coefficient",
    x0=20,
    y0=40,
    width=560,
    height=440,
)}
{_bar_plot_elements(
    categories=road_categories,
    values=road_static,
    title="Road type vs friction (water film = 0.20 mm)",
    x_label="road type",
    y_label="mu_static",
    x0=620,
    y0=40,
    width=560,
    height=440,
)}
</svg>
"""
    output.write_text(svg, encoding="utf-8")
    return output


def write_validation_svg(path: str | Path) -> Path:
    """Write a dependency-free SVG comparing model curves to the paper averages."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    speed_predictions, water_predictions = _paper_validation_predictions()
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="520" viewBox="0 0 1200 520">
<rect width="1200" height="520" fill="#fafafa"/>
<text x="600" y="24" font-size="20" text-anchor="middle">TRFC paper comparison</text>
{_line_plot_elements(
    x=[float(v) for v in PAPER_TABLE2_SPEEDS_KMH],
    series=(
        ("paper avg", "#dc3912", PAPER_TABLE2_AVG_MU),
        ("model", "#3366cc", speed_predictions),
    ),
    title="Speed sweep, water film = 0.25 mm",
    x_label="speed (km/h)",
    y_label="friction coefficient",
    x0=20,
    y0=40,
    width=560,
    height=440,
)}
{_line_plot_elements(
    x=PAPER_TABLE3_WATER_MM,
    series=(
        ("paper avg", "#dc3912", PAPER_TABLE3_AVG_MU),
        ("model", "#3366cc", water_predictions),
    ),
    title="Water-film sweep, speed = 50 km/h",
    x_label="water film thickness (mm)",
    y_label="friction coefficient",
    x0=620,
    y0=40,
    width=560,
    height=440,
)}
</svg>
"""
    output.write_text(svg, encoding="utf-8")
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wet-road friction estimation program")
    parser.add_argument("--road-type", default="AC")
    parser.add_argument("--texture-coeff", type=float, default=None)
    parser.add_argument("--texture-amplitude-mm", type=float, default=None)
    parser.add_argument("--tire-model-id", default="zhao2024_modified_all_default")
    parser.add_argument("--water-film-mm", type=float, default=None)
    parser.add_argument("--reference-speed-mps", type=float, default=DEFAULT_V_REF_MPS)
    parser.add_argument("--slip-static", type=float, default=DEFAULT_S_REF_STATIC)
    parser.add_argument("--slip-dynamic", type=float, default=DEFAULT_S_REF_DYNAMIC)
    parser.add_argument("--mu-eff-mode", choices=("static", "dynamic"), default="static")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--plot", metavar="SVG_PATH", default=None)
    parser.add_argument("--validate", metavar="SVG_PATH", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.demo:
        print(run_demo())
        if args.plot:
            path = write_demo_svg(args.plot)
            print(f"\nwrote_plot: {path}")
        if args.validate:
            print()
            print(paper_validation_report())
            path = write_validation_svg(args.validate)
            print(f"\nwrote_validation_plot: {path}")
        return 0

    if args.validate:
        print(paper_validation_report())
        path = write_validation_svg(args.validate)
        print(f"\nwrote_validation_plot: {path}")
        return 0

    if args.water_film_mm is None:
        parser.error("--water-film-mm is required unless --demo or --validate is used")

    estimate = estimate_friction(
        FrictionInput(
            road_type=args.road_type,
            texture_coeff=args.texture_coeff,
            texture_amplitude_mm=args.texture_amplitude_mm,
            water_film_mm=args.water_film_mm,
            tire_model_id=args.tire_model_id,
            reference_speed_mps=args.reference_speed_mps,
            slip_static=args.slip_static,
            slip_dynamic=args.slip_dynamic,
            mu_eff_mode=args.mu_eff_mode,
        )
    )
    print(_format_single_estimate(estimate))
    if args.plot:
        path = write_demo_svg(args.plot)
        print(f"\nwrote_plot: {path}")
    return 0
