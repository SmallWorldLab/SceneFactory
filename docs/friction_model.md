# The weather-to-friction model

SceneFactory maps a weather description (precipitation type/intensity, water-film
depth) and a road surface type onto a scalar friction coefficient μ, which is
applied as the PhysX friction of that world's ground.

Implementation: `src/trfc/friction_api.py`
Entry point: `estimate_friction(FrictionInput) -> FrictionEstimate`
CLI: `python -m src.trfc --road-type AC --water-film-mm 0.2`

The model is the modified ALL (Average Lumped LuGre) formulation of:

> Zhao, L., Zhao, H., Cai, J. (2024). *Tire-pavement friction modeling considering
> pavement texture and water film.* International Journal of Transportation
> Science and Technology **14**, 99–109. doi:10.1016/j.ijtst.2023.04.001

---

## Correction to Eq. (12) — the coefficient `A`

**This is a post-submission correction to the code.** It changes μ in the wet
regime and therefore affects any result computed with water films above ~0.8 mm.

### The defect

Eq. (12) of the source paper introduces a coefficient `A` in the denominator but
never defines it: the where-clause following Eq. (13) defines γ, r₀, p, L, h_min,
h₀, ε, ρ and b, and omits `A`.

The original implementation guessed that `A` was the nominal tyre contact-patch
**area** (L × B = 0.0205 m²). Dimensional analysis of Eq. (12) shows it cannot be:

```
[12 v γ r₀² / (A π p L)] × [1/(h² + 2εh)]
  = (m/s · Pa·s · m²) / (A · Pa · m) × m⁻²
  = (m²/A) × m⁻²  =  1/A
```

`Y_R` is a length *ratio*, so the bracketed product must be dimensionless, so `A`
must be dimensionless. Dividing by 0.0205 m² inflated the squeeze-film correction
by ~49×. That drove `Y_R` negative for any `h₀ > h_min` and produced a hard,
speed-independent step to **μ = 0 at exactly h = h_min/0.01 = 0.8 mm**.

### The fix

- `AllWetRoadParameters.squeeze_film_coefficient_a = 4.05` (dimensionless) is now
  used in Eq. (12). `contact_patch_area_m2` is retained as a reported diagnostic
  only and no longer enters the computation path.
- `AllWetRoadParameters.alpha = 0.90` (was 0.5). α is the Stribeck exponent in
  `g(v_r) = μ_c + (μ_s − μ_c)·exp(−|v_r/v_s|^α)`. The paper's Table 4 lists σ₀, σ₂,
  μ_c, μ_s and β₁…β₄ and omits α; 0.5 was an assumption. 0.90 is fitted to the
  paper's own Tables 2 and 3.

`A` is calibrated against the left panel of the paper's Fig. 6, whose x-axis
crossings the paper defines as "the hydroplaning velocity of the vehicle under the
corresponding water film thickness". The calibration condition is **μ = 0, not
Y_R = 0**: hydroplaning in Eq. (15) occurs when θ·Y_R = Y_F, which is reached while
Y_R is still positive.

| water film | paper crossing | model | error |
|---|---|---|---|
| 5 mm | 100 km/h | 103.8 | +3.8 |
| 10 mm | 90 km/h | 85.5 | −4.5 |
| 20 mm | ~77 km/h | 71.5 | −6.0 |

α and `A` are nearly orthogonal: refitting `A` at each α returns 4.05 throughout,
because α governs the calibrated sub-millimetre range and `A` governs the
hydroplaning crossings at 5–20 mm.

---

## Reproducing the validation

```bash
PYTHONPATH=. python -m pytest src/trfc/tests/test_friction_water_film.py -q
PYTHONPATH=. python scripts/trfc_paper_validation.py --out artifacts/trfc_validation
```

The first is an 11-case regression suite pinning the corrected behaviour. The
second checks the model against the source paper across and beyond the paper's
calibrated range, and writes `validation_extended.txt`, `paper_validation_report.txt`
and `mu_vs_water_film.csv`.

`scripts/digitize_zhao_fig6.py` and `scripts/plot_trfc_vs_paper.py` reproduce the
Fig. 6 comparison. The digitiser requires **your own copy of the published
figure** — the figure is copyrighted by the publisher and is not redistributed here.

### What the validation currently reports

Four acceptance criteria; **three pass, one fails.** The failure is stated here
rather than omitted.

| # | criterion | before fix | after fix |
|---|---|---|---|
| 1 | §3.2 point condition (Y_R≈1, Y_F≈0, 61 % drop at 1 mm / 100 km/h) | FAIL | **FAIL** |
| 2 | Table 2/3 RMSE does not regress | PASS | PASS |
| 3 | no exact-zero μ through 20 mm at 60 km/h | FAIL | **PASS** |
| 4 | hydroplaning onset falls as speed rises | FAIL | **PASS** |

RMSE against the paper's own regression data improves:

| | before (α=0.5) | after (α=0.9) | paper reports |
|---|---|---|---|
| Table 2 speed sweep | 0.0611 | **0.0405** | 0.023 |
| Table 3 water sweep | 0.0554 | **0.0206** | 0.023 |

**Why criterion 1 still fails.** No single value of `A` satisfies both of the
paper's own statements:

- Fig. 6's hydroplaning velocities require `A ≈ 4.05`
- §3.2's "Y_R is approximately equal to 1" at h = 1 mm, v = 100 km/h requires `A ≥ 6.63`

— a 1.6× disagreement. We calibrate to Fig. 6 because it is a two-panel dataset
covering the whole (v, h) domain, rather than one qualitative sentence about a
single operating point. At `A = 4.05` the model gives Y_R = 0.836 and a 54 % μ drop
at that condition, against the paper's "≈1" and "61 %". This is an inconsistency in
the source, not in this implementation.

### Scope of the validation claim

This validates the implementation against the **published curves of the cited
paper**. It is *not* a validation against measured tyre-pavement friction data,
and no claim of that kind should be made on the basis of these scripts.
