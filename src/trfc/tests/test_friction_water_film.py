"""Regression tests for the Eq. (12) / Y_R water-film fix in friction_api.

Guards the defect described in docs/friction_model.md: `A` in
Eq. (12) was being passed the nominal contact-patch AREA (m^2), which is
dimensionally inconsistent -- dimensional analysis of Eq. (12) shows A must be
dimensionless -- and inflated the squeeze-film correction ~49x.  The result was
Y_R flipping sign at h0 = h_min, i.e. a hard mu -> 0 cliff at
h = h_min / 0.01 = 0.8 mm, identical at every speed.

Paper: Zhao, L., Zhao, H., Cai, J. (2024), Int. J. Transportation Science and
Technology 14, 99-109, doi:10.1016/j.ijtst.2023.04.001

Two known gaps remain after this fix and are asserted here as CURRENT BEHAVIOUR
so that closing them trips these tests deliberately rather than silently:
  * the paper's own Sec. 3.2 ("Y_R approximately equal to 1" at h=1mm, v=100km/h)
    is not reachable with the A that reproduces its Fig. 6 hydroplaning speeds --
    the two constraints differ by 2.4x, an internal inconsistency in the paper;
  * the lubrication branch (Eq. 11) is independently too weak; with Y_R pinned to
    1.0 it delivers only a ~38% mu drop at 1 mm / 100 km/h against the paper's 61%.
"""
from __future__ import annotations

import math

import pytest

from src.trfc.friction_api import (
    PAPER_TABLE2_AVG_MU,
    PAPER_TABLE3_AVG_MU,
    AllWetRoadParameters,
    _paper_validation_predictions,
    compute_mu_all_modified,
)

SLIP = 0.15


def mu(v_kmh: float, h_mm: float, road: str = "AC") -> float:
    return compute_mu_all_modified(
        v_ref=v_kmh / 3.6, slip=SLIP, h_w_mm=h_mm, road_type=road
    ).mu


def _rmse(a, b) -> float:
    return (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5


def test_squeeze_film_coefficient_is_dimensionless_and_calibrated():
    """A is a bare number, not an area. Regression guard on the actual bug."""
    a = AllWetRoadParameters().squeeze_film_coefficient_a
    assert isinstance(a, float)
    # Calibrated by weighted least squares on Fig. 6 hydroplaning speeds.
    assert 3.5 <= a <= 4.5
    # The old broken value was a contact-patch area, ~0.02 m^2. Two orders below.
    assert a > 1.0, "A looks like an area again -- Eq. (12) regression"


def test_no_cliff_at_the_old_0p8mm_step():
    """mu was 0.804 at 0.80 mm and exactly 0.000 at 0.82 mm -- a step function.

    The cliff sat at h = h_min / h0_ratio = 0.008 / 0.01 = 0.8 mm, where the two
    reciprocal terms of Eq. (12) cross and the bracket flips sign.  Test for an
    actual DISCONTINUITY there (vanishing step size), not merely for a small
    finite-interval change: the curve legitimately has real curvature in this
    region from the exponential v_s(h) of Eq. (11).
    """
    for v in (60.0, 100.0):
        left, right = mu(v, 0.7995), mu(v, 0.8005)
        assert left > 0.0 and right > 0.0, f"mu still collapses at 0.8 mm, v={v}"
        assert abs(left - right) < 1e-3, (
            f"discontinuity at the old cliff, v={v}: {left:.6f} -> {right:.6f}"
        )
        # and it must not have merely moved: mu stays healthy well past 0.8 mm
        assert mu(v, 1.0) > 0.4, f"mu collapsed shortly after 0.8 mm at {v} km/h"


def test_mu_is_monotone_decreasing_in_water_film():
    films = [0.0, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0]
    for v in (60.0, 80.0, 100.0):
        vals = [mu(v, h) for h in films]
        for lo, hi, a, b in zip(films, films[1:], vals, vals[1:]):
            assert b <= a + 1e-9, f"mu rose from {lo} to {hi} mm at {v} km/h"


def test_mu_is_monotone_decreasing_in_speed():
    for h in (0.5, 1.0, 2.0):
        vals = [mu(v, h) for v in (40.0, 60.0, 80.0, 100.0, 120.0)]
        for a, b in zip(vals, vals[1:]):
            assert b <= a + 1e-9, f"mu rose with speed at h={h} mm"


def test_hydroplaning_onset_is_speed_dependent():
    """Criterion 4. Before the fix this was 0.85 mm at every speed -- proof on its
    own that the speed-linear Eq. (12) prefactor was not doing anything."""
    onsets = {}
    for v in (60.0, 80.0, 100.0, 120.0):
        h = 0.05
        while h <= 25.0:
            if mu(v, round(h, 3)) <= 0.0:
                onsets[v] = round(h, 3)
                break
            h += 0.05

    # 60 km/h must NOT hydroplane within 20 mm -- Fig. 6 right panel shows the
    # 60 km/h curve still at mu ~ 0.22 at h = 20 mm.
    assert 60.0 not in onsets or onsets[60.0] > 20.0, (
        f"60 km/h hydroplanes at {onsets.get(60.0)} mm; Fig. 6 says it should not"
    )
    # the faster speeds must, and progressively sooner
    fast = [onsets.get(v) for v in (80.0, 100.0, 120.0)]
    assert all(x is not None for x in fast), f"missing hydroplaning onsets: {onsets}"
    for a, b in zip(fast, fast[1:]):
        assert b < a, f"onset must fall as speed rises, got {onsets}"


def test_fig6_hydroplaning_speeds_reproduced():
    """The anchors A was calibrated on. Paper Fig. 6 left panel x-axis crossings."""
    for h_mm, v_paper_kmh in ((5.0, 100.0), (10.0, 90.0)):
        v = 40.0
        crossing = None
        while v <= 140.0:
            if mu(v, h_mm) <= 0.0:
                crossing = v
                break
            v += 1.0
        assert crossing is not None, f"no crossing found at h={h_mm} mm"
        assert abs(crossing - v_paper_kmh) <= 10.0, (
            f"h={h_mm} mm crossing {crossing} km/h vs paper {v_paper_kmh} km/h"
        )


def test_wet_range_spans_the_useful_friction_band():
    """Why this fix matters downstream: the weather axis must reach mu ~0.3-0.5,
    the band where friction actually binds for the SceneFactory vehicle.  Before
    the fix the only reachable values were >=0.80 or exactly 0."""
    values = [mu(60.0, h) for h in (1.0, 2.0, 3.0, 5.0)]
    assert any(0.25 <= x <= 0.55 for x in values), (
        f"no water film yields mu in [0.25, 0.55] at 60 km/h: {values}"
    )


def test_table2_table3_rmse_does_not_regress():
    """Criterion 2. The fix must not disturb the calibrated 0.05-0.50 mm range,
    which sits entirely below the old cliff."""
    speed_pred, water_pred = _paper_validation_predictions()
    # Tightened to the post-alpha-calibration values so the improvement is locked
    # in, not merely the old 0.0611 / 0.0554 ceiling. Paper reports 0.023 for both;
    # the water sweep now beats that, the speed sweep is still ~1.8x it.
    assert _rmse(PAPER_TABLE2_AVG_MU, speed_pred) <= 0.0410
    assert _rmse(PAPER_TABLE3_AVG_MU, water_pred) <= 0.0210


def test_dry_is_untouched_by_the_yr_fix():
    """h = 0 short-circuits before Eq. (12), so the A fix cannot move dry mu.

    Dry mu DID move (1.0707 -> 1.1577) but only via the alpha recalibration, which
    changes g(v_r) at every water film including zero.  Pinned here so a further
    change to either parameter is deliberate."""
    assert mu(100.0, 0.0) == pytest.approx(1.1577, abs=1e-3)


# ---------------------------------------------------------------------------
# Known-unfixed gaps, asserted as current behaviour.
# ---------------------------------------------------------------------------

def test_known_gap_paper_internal_inconsistency_on_A():
    """Sec. 3.2 says Y_R ~ 1 at h=1mm/100km/h; Fig. 6's hydroplaning speeds say
    A = 4.05, which forces Y_R = 0.84 there.  Satisfying Sec. 3.2 would need
    A >= 6.63 -- 1.6x larger -- which would push every hydroplaning speed off
    Fig. 6.  No single A satisfies both; the paper is internally inconsistent.
    We calibrate to Fig. 6 because it is a full 2-panel dataset rather than one
    qualitative sentence."""
    est = compute_mu_all_modified(v_ref=100 / 3.6, slip=SLIP, h_w_mm=1.0, road_type="AC")
    assert est.y_r == pytest.approx(0.836, abs=0.02)
    assert est.y_f == pytest.approx(0.058, abs=0.01)  # Y_F ~ 0 agrees with the paper


def test_known_gap_lubrication_branch_too_weak():
    """Eq. (11) matches the paper verbatim (b1..b4 = Table 4), but its regression
    overshoots v_s at small h -- the paper's own Fig. 3 shows v_s ~ 37 m/s at
    h = 0 while Eq. (11) yields 102.65 -- so the Stribeck term under-delivers.
    Total drop at 1 mm / 100 km/h is ~54% against the paper's stated 61%."""
    dry = mu(100.0, 0.0)
    wet = mu(100.0, 1.0)
    drop_pct = 100.0 * (1.0 - wet / dry)
    assert 51.0 <= drop_pct <= 58.0, f"drop {drop_pct:.1f}% moved; paper says 61%"

    params = AllWetRoadParameters()
    v_s_dry = params.b1 * math.exp(params.b3) + params.b4
    assert v_s_dry == pytest.approx(102.65, abs=0.5)
