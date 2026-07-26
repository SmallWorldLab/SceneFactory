#!/usr/bin/env python3
"""Validate src/trfc against the cited paper, in-range AND beyond.

    Zhao, L., Zhao, H., Cai, J. (2024) "Tire-pavement friction modeling
    considering pavement texture and water film", Int. J. Transportation Science
    and Technology 14 (2024) 99-109, doi:10.1016/j.ijtst.2023.04.001

The module already ships `paper_validation_report()`, but it only exercises the
paper's Table 2 / Table 3 ranges (water film 0.05-0.50 mm, 10-90 km/h).  The
known Y_R defect lives ABOVE 0.8 mm, so that report passes while the model is
badly wrong in exactly the regime SceneFactory trains and evaluates in.

This script adds the out-of-range checks and emits explicit PASS/FAIL against the
acceptance criteria for the Eq. (12) correction, so the same command both
documents the defect and verifies the fix.

Usage:
    PYTHONPATH=. python scripts/trfc_paper_validation.py [--out DIR]
Exit code is 0 if every criterion passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.trfc.friction_api import (
    PAPER_TABLE2_AVG_MU,
    PAPER_TABLE2_SPEEDS_KMH,
    PAPER_TABLE3_AVG_MU,
    PAPER_TABLE3_WATER_MM,
    compute_mu_all_modified,
    paper_validation_report,
)

SPEEDS_KMH = [60, 80, 100, 120]
FILMS_MM = [0.0, 0.25, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0, 20.0]
SLIP = 0.15

# Paper Fig. 6 right panel, read off the printed figure.  APPROXIMATE -- used for
# a shape comparison only, never as a pass/fail threshold.  Re-digitize from the
# PDF before treating any single value as authoritative.
FIG6_APPROX = {
    1.0: {60: 0.85, 80: 0.75, 100: 0.60, 120: 0.45},
    2.0: {60: 0.72, 80: 0.55, 100: 0.35, 120: 0.10},
    5.0: {60: 0.55, 80: 0.30, 100: 0.05, 120: 0.00},
    10.0: {60: 0.38, 80: 0.05, 100: 0.00, 120: 0.00},
    20.0: {60: 0.22, 80: 0.00, 100: 0.00, 120: 0.00},
}


def mu(v_kmh: float, h_mm: float, road: str = "AC"):
    return compute_mu_all_modified(
        v_ref=v_kmh / 3.6, slip=SLIP, h_w_mm=h_mm, road_type=road
    )


def rmse(a, b) -> float:
    return (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/trfc_validation")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    results: list[tuple[str, bool, str]] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("TRFC validation against Zhao et al. (2024)")
    emit("=" * 78)

    # ---- Criterion 1: paper section 3.2 point condition -------------------
    emit("\n[1] Paper section 3.2: at v=100 km/h, h=1 mm --")
    emit('    "Y_R is approximately equal to 1 and Y_F is approximately equal to 0"')
    emit('    "the friction coefficient drops ... can be as high as 61%"')
    d = mu(100, 1.0)
    dry = mu(100, 0.0)
    drop = 100.0 * (1.0 - d.mu / dry.mu) if dry.mu > 0 else float("nan")
    emit(f"    Y_R = {d.y_r:.4f}   (paper ~1, accept 0.90-1.00)")
    emit(f"    Y_F = {d.y_f:.4f}   (paper ~0, accept 0.00-0.10)")
    emit(f"    mu  = {d.mu:.4f}  vs dry {dry.mu:.4f}  -> drop {drop:.0f}% (paper 61%, accept 56-66)")
    ok1 = 0.90 <= d.y_r <= 1.00 and 0.0 <= d.y_f <= 0.10 and 56.0 <= drop <= 66.0
    results.append(("1. section 3.2 point condition (Y_R~1, Y_F~0, 61% drop)", ok1, ""))

    # ---- Criterion 2: in-range calibration quality ------------------------
    emit("\n[2] In-range calibration vs paper Tables 2 and 3 (GripTester geometry)")
    from src.trfc.friction_api import _paper_validation_predictions

    sp, wp = _paper_validation_predictions()
    r_speed = rmse(PAPER_TABLE2_AVG_MU, sp)
    r_water = rmse(PAPER_TABLE3_AVG_MU, wp)
    emit(f"    speed sweep RMSE {r_speed:.4f}   (baseline 0.0611, paper reports 0.023)")
    emit(f"    water sweep RMSE {r_water:.4f}   (baseline 0.0554, paper reports 0.023)")
    ok2 = r_speed <= 0.0611 + 1e-6 and r_water <= 0.0554 + 1e-6
    results.append(("2. Table 2/3 RMSE does not regress from baseline", ok2, ""))

    # ---- Criterion 3: no exact zeros inside the paper's demonstrated range -
    emit("\n[3] Paper Fig. 6 shows mu > 0 out to h = 20 mm at v = 60 km/h")
    zeros = [h for h in FILMS_MM if h > 0 and mu(60, h).mu <= 0.0]
    emit(f"    h (mm) with mu == 0 at 60 km/h: {zeros if zeros else 'none'}")
    ok3 = not zeros
    results.append(("3. no exact-zero mu through 20 mm at 60 km/h", ok3, ""))

    # ---- Criterion 4: hydroplaning onset must be speed-dependent ----------
    emit("\n[4] Hydroplaning onset must fall as speed rises (paper: 'hydroplaning velocity')")
    onsets: dict[int, float | None] = {}
    for v in SPEEDS_KMH:
        onset = None
        h = 0.05
        while h <= 25.0:
            if mu(v, round(h, 3)).mu <= 0.0:
                onset = round(h, 3)
                break
            h += 0.05
        onsets[v] = onset
        emit(f"    {v:3d} km/h -> onset {onset if onset is not None else '> 25 mm'}")
    got = [onsets[v] for v in SPEEDS_KMH]
    ok4 = all(a is not None and b is not None and b < a for a, b in zip(got, got[1:])) or (
        got[0] is None and any(x is not None for x in got[1:])
    )
    results.append(("4. onset strictly decreases with speed", ok4, ""))

    # ---- Fig. 6 shape comparison (informational, never pass/fail) ---------
    emit("\n[i] Fig. 6 right-panel shape comparison (approximate figure readings)")
    emit(f"    {'h_mm':>6} " + "".join(f"{v:>8}km/h" for v in SPEEDS_KMH))
    for h in [1.0, 2.0, 5.0, 10.0, 20.0]:
        ours = "".join(f"{mu(v, h).mu:12.3f}" for v in SPEEDS_KMH)
        paper = "".join(f"{FIG6_APPROX[h][v]:12.2f}" for v in SPEEDS_KMH)
        emit(f"    {h:6.1f} ours {ours}")
        emit(f"    {'':6s} papr {paper}")

    # ---- full grid CSV ---------------------------------------------------
    csv_path = out / "mu_vs_water_film.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["road_type", "water_film_mm", "speed_kmh", "mu", "y_r", "y_f", "hydroplaning"])
        for road in ["AC", "SMA", "OGFC"]:
            for h in FILMS_MM:
                for v in SPEEDS_KMH:
                    d = mu(v, h, road)
                    w.writerow([road, h, v, f"{d.mu:.6f}", f"{d.y_r:.6f}",
                                f"{d.y_f:.6f}", int(d.hydroplaning)])

    # ---- verdict ---------------------------------------------------------
    emit("\n" + "=" * 78)
    emit("VERDICT")
    for name, ok, _ in results:
        emit(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    all_ok = all(ok for _, ok, _ in results)
    emit(f"\n  overall: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        emit("  See docs/friction_model.md for the Eq. (12) derivation.")

    (out / "validation_extended.txt").write_text("\n".join(lines) + "\n")
    (out / "paper_validation_report.txt").write_text(paper_validation_report() + "\n")
    print(f"\n[artifacts] {out}/validation_extended.txt")
    print(f"[artifacts] {out}/paper_validation_report.txt")
    print(f"[artifacts] {csv_path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
