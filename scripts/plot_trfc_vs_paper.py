#!/usr/bin/env python3
"""Reviewer-facing figure: SceneFactory's TRFC module vs Zhao et al. (2024).

Three panels:
  (a) Fig. 6 left panel -- mu vs velocity at h = 1, 5, 10, 20 mm.  Paper values
      are DIGITIZED from the published raster figure by
      scripts/digitize_zhao_fig6.py, not eyeballed.
  (b) residuals for (a), so the h = 1 mm failure is visible rather than buried.
  (c) Tables 2 and 3 -- the paper's PRINTED calibration numbers, which are exact
      transcriptions, not figure readings.

Honesty notes rendered onto the figure itself:
  * two model parameters (A, alpha) are unspecified by the paper and were fitted
    to the paper's own published output;
  * the h = 1 mm disagreement is a known unfixed defect in the lubrication branch.

Usage:
    PYTHONPATH=. python scripts/plot_trfc_vs_paper.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.trfc.friction_api import (
    PAPER_TABLE2_AVG_MU,
    PAPER_TABLE2_SPEEDS_KMH,
    PAPER_TABLE3_AVG_MU,
    PAPER_TABLE3_WATER_MM,
    _paper_validation_predictions,
    compute_mu_all_modified,
)

DIGITIZED = Path("artifacts/trfc_validation/zhao_fig6_digitized.json")
SERIES = [("h=1mm", 1.0, "#3b4a8c"), ("h=5mm", 5.0, "#b2342e"),
          ("h=10mm", 10.0, "#e2b236"), ("h=20mm", 20.0, "#1a1a1a")]


def mu(v_kmh: float, h_mm: float) -> float:
    return compute_mu_all_modified(
        v_ref=v_kmh / 3.6, slip=0.15, h_w_mm=h_mm, road_type="AC"
    ).mu


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/trfc_validation/trfc_vs_paper.png")
    args = ap.parse_args()

    if not DIGITIZED.exists():
        raise SystemExit(
            f"missing {DIGITIZED}. Run:\n"
            "  pdfimages -f 8 -l 8 -png <paper.pdf> /tmp/fig6/p8\n"
            "  PYTHONPATH=. python scripts/digitize_zhao_fig6.py --image /tmp/fig6/p8-000.png"
        )
    dig = json.loads(DIGITIZED.read_text())["left_mu_vs_velocity_kmh"]

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.28)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))

    vs = list(range(10, 125, 2))
    all_err = []
    for label, h, colour in SERIES:
        pts = {int(k): v for k, v in dig[label].items()}
        xs = sorted(pts)
        ax0.plot(xs, [pts[x] for x in xs], "o", color=colour, ms=6,
                 mec="white", mew=0.8, label=f"paper {label}", zorder=3)
        ax0.plot(vs, [mu(v, h) for v in vs], "-", color=colour, lw=1.8, alpha=0.85,
                 label=f"ours {label}", zorder=2)
        use = [x for x in xs if x >= 30]
        err = [mu(x, h) - pts[x] for x in use]
        all_err += err
        ax1.plot(use, err, "o-", color=colour, ms=4, lw=1.4, label=label)

    ax0.set_xlabel("Velocity (km/h)")
    ax0.set_ylabel("Friction coefficient $\\mu$")
    ax0.set_title("(a) vs Zhao Fig. 6 (digitized)\nmarkers = paper, lines = ours")
    ax0.set_ylim(0, 1.0)
    ax0.set_xlim(5, 125)
    ax0.grid(alpha=0.25)
    ax0.legend(fontsize=6.5, ncol=2, loc="upper right")

    rmse = (sum(e * e for e in all_err) / len(all_err)) ** 0.5
    ax1.axhline(0, color="k", lw=1)
    ax1.axhspan(-0.05, 0.05, color="green", alpha=0.10)
    ax1.set_xlabel("Velocity (km/h)")
    ax1.set_ylabel("ours - paper")
    ax1.set_title(f"(b) residuals, v $\\geq$ 30 km/h\noverall RMSE {rmse:.3f}")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7)
    ax1.text(0.03, 0.03,
             "h=1mm is lubrication-dominated:\nknown unfixed defect in Eq. (11)",
             transform=ax1.transAxes, fontsize=7, va="bottom",
             bbox=dict(fc="#fff3cd", ec="#d0a800", lw=0.6, alpha=0.95))

    sp, wp = _paper_validation_predictions()
    ax2.plot(PAPER_TABLE2_AVG_MU, sp, "o", color="#3b4a8c", ms=7, mec="white",
             label="Table 2 (speed sweep)")
    ax2.plot(PAPER_TABLE3_AVG_MU, wp, "s", color="#b2342e", ms=7, mec="white",
             label="Table 3 (water sweep)")
    lims = [0.55, 0.90]
    ax2.plot(lims, lims, "k-", lw=1)
    ax2.set_xlim(*lims)
    ax2.set_ylim(*lims)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Paper measured (printed tables)")
    ax2.set_ylabel("Ours calculated")
    ax2.set_title("(c) vs printed Tables 2 & 3\n(exact transcriptions, not figure reads)")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=7, loc="upper left")

    def _rmse(a, b):
        return (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5

    ax2.text(0.97, 0.05,
             f"RMSE  speed {_rmse(PAPER_TABLE2_AVG_MU, sp):.4f}\n"
             f"        water {_rmse(PAPER_TABLE3_AVG_MU, wp):.4f}\n"
             f"paper reports 0.023",
             transform=ax2.transAxes, fontsize=7, ha="right", va="bottom",
             bbox=dict(fc="white", ec="0.6", lw=0.6))

    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.5, 0.02,
        "SceneFactory TRFC vs Zhao et al. (2024).  Two parameters unspecified by the paper "
        "(A in Eq. 12, Stribeck exponent $\\alpha$) were fitted to the paper's own published output.\n"
        "Panel (a) reference values are digitized from the published raster figure "
        "($\\pm$0.01); panels (c) values are exact transcriptions of printed tables.",
        fontsize=8, ha="center", va="bottom", color="0.25",
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(f"[figure] {out}")
    print(f"Fig.6 residual RMSE (v>=30): {rmse:.4f}")
    h1 = [e for (lab, h, _), e in zip(SERIES, [None] * 4)] if False else None
    for label, h, _ in SERIES:
        pts = {int(k): v for k, v in dig[label].items()}
        use = [x for x in pts if x >= 30]
        e = [mu(x, h) - pts[x] for x in use]
        print(f"  {label:8s} n={len(e):2d}  RMSE {(sum(v*v for v in e)/len(e))**0.5:.4f}")


if __name__ == "__main__":
    main()
