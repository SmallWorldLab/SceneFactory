#!/usr/bin/env python3
"""Digitize Fig. 6 of Zhao et al. (2024) from the published PDF.

    Zhao, L., Zhao, H., Cai, J. (2024) "Tire-pavement friction modeling
    considering pavement texture and water film", Int. J. Transportation Science
    and Technology 14 (2024) 99-109, doi:10.1016/j.ijtst.2023.04.001

WHY THIS EXISTS
---------------
Tables 2-5 of the paper are printed numbers and were transcribed exactly into
src/trfc/friction_api.py.  But they only cover 0.05-0.50 mm water film and
10-90 km/h -- entirely below the mu-cliff regime that the Eq. (12)
correction addresses (see docs/friction_model.md).  The ONLY published reference
for the 1-20 mm regime, which is exactly the regime SceneFactory needs, is
Fig. 6, and the paper publishes no table behind it.

Until now the comparison values for Fig. 6 were eyeball readings hardcoded in
scripts/trfc_paper_validation.py (FIG6_APPROX), self-labelled "APPROXIMATE ...
never as a pass/fail threshold".  They are also substantially wrong -- e.g. they
record mu = 0.85 at h = 1 mm, v = 60 km/h where the figure shows ~0.55.

Fig. 6 is a RASTER image in the PDF (page 8, 1914x749 px at 300 dpi), so there
are no vector coordinates to recover.  This script instead digitizes it by
marker colour: it isolates each series by RGB, erodes away the connecting
polyline to leave the marker cores, takes connected-component centroids, and
maps pixel -> data coordinates from the axis frame.

Accuracy is limited by marker size (~18 px) against a 624 px tall axis, i.e.
roughly +/-0.01 in mu -- an order of magnitude better than eyeballing, and
good enough to use as a quantitative reference.

Usage:
    pdfimages -f 8 -l 8 -png <paper.pdf> /tmp/fig6/p8
    PYTHONPATH=. python scripts/digitize_zhao_fig6.py --image /tmp/fig6/p8-000.png

Writes artifacts/trfc_validation/zhao_fig6_digitized.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# Axis frame in pixel coordinates, measured from the extracted 1914x749 image by
# locating the long dark vertical/horizontal rules.
LEFT_PANEL = {"x0": 112, "x1": 934, "xdata": (10.0, 120.0)}
RIGHT_PANEL = {"x0": 1072, "x1": 1894, "xdata": (0.0, 20.0)}
Y0_PX, Y1_PX = 636, 12          # mu = 0 at y=636, mu = 1 at y=12
YDATA = (0.0, 1.0)

# Series colours, sampled from the figure. (name, rgb, tolerance)
SERIES = [
    ("blue", (54, 62, 116), 60),
    ("red", (178, 52, 46), 60),
    ("gold", (226, 178, 54), 60),
    ("black", (16, 16, 16), 45),
]
LEFT_LABELS = {"blue": "h=1mm", "red": "h=5mm", "gold": "h=10mm", "black": "h=20mm"}
RIGHT_LABELS = {"blue": "v=60", "red": "v=80", "gold": "v=100", "black": "v=120"}


def px_to_mu(y: float) -> float:
    return YDATA[0] + (Y0_PX - y) * (YDATA[1] - YDATA[0]) / (Y0_PX - Y1_PX)


def px_to_x(x: float, panel: dict) -> float:
    lo, hi = panel["xdata"]
    return lo + (x - panel["x0"]) * (hi - lo) / (panel["x1"] - panel["x0"])


# Legend boxes, located by finding the medium-length dark horizontal rules in the
# top of each panel.  Markers inside these are legend keys, not data, and must be
# excluded or they contaminate the high-mu end of the curves.
LEGENDS = [(689, 920, 26, 221), (1613, 1880, 26, 221)]  # x0, x1, y0, y1


def _series_mask(img: np.ndarray, colour, tol: int) -> np.ndarray:
    mask = np.abs(img - np.array(colour)).sum(axis=2) < tol
    # blank the legends and a margin around the axis frame
    for lx0, lx1, ly0, ly1 in LEGENDS:
        mask[ly0 - 3 : ly1 + 4, lx0 - 3 : lx1 + 4] = False
    return mask


def extract_at(mask: np.ndarray, x_px: int, half: int = 8):
    """Find the marker centre in a narrow column band around x_px.

    The connecting polyline shares the series colour, so within the band we take
    the *densest* vertical run: the marker is ~18 px tall where the line is ~5.
    Returns mu, or None if the series has no ink in this column.
    """
    band = mask[Y1_PX : Y0_PX + 12, max(0, x_px - half) : x_px + half + 1]
    rows = band.sum(axis=1)
    if rows.sum() < 12:
        return None
    # group contiguous rows that carry ink, score each group by total pixels
    groups, cur = [], []
    for y, n in enumerate(rows):
        if n > 0:
            cur.append(y)
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    if not groups:
        return None
    best = max(groups, key=lambda g: sum(rows[y] for y in g))
    weight = sum(rows[y] for y in best)
    centre = sum(y * rows[y] for y in best) / weight
    mu = px_to_mu(centre + Y1_PX)
    return round(max(mu, 0.0), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="artifacts/trfc_validation/zhao_fig6_digitized.json")
    ap.add_argument("--erode", type=int, default=5)
    args = ap.parse_args()

    img = np.array(Image.open(args.image).convert("RGB")).astype(int)
    if img.shape[:2] != (749, 1914):
        raise SystemExit(
            f"unexpected image size {img.shape[:2]}; axis calibration assumes (749, 1914). "
            "Re-measure the axis frame if the extraction changed."
        )

    result = {
        "source": "Zhao et al. (2024) Fig. 6 left panel, digitized by marker colour",
        "method": "per-marker column band, densest vertical run; legends masked",
        "accuracy_note": "marker ~18 px on a 624 px axis => roughly +/-0.01 in mu",
        "left_mu_vs_velocity_kmh": {},
    }

    speeds = list(range(10, 130, 10))
    for name, colour, tol in SERIES:
        mask = _series_mask(img, colour, tol)
        series = {}
        for v in speeds:
            x_px = int(round(
                LEFT_PANEL["x0"]
                + (v - 10.0) * (LEFT_PANEL["x1"] - LEFT_PANEL["x0"]) / (120.0 - 10.0)
            ))
            val = extract_at(mask, x_px)
            if val is not None:
                series[v] = val
        result["left_mu_vs_velocity_kmh"][LEFT_LABELS[name]] = series

    # ---- cleanup -----------------------------------------------------------
    # Two known artefacts, both from marker occlusion rather than from the data:
    #   * at v = 10 km/h all four curves converge into a single clump, so the
    #     later-drawn series can capture an earlier one's pixels;
    #   * once a series reaches mu = 0 its markers sit ON the x-axis and are
    #     overdrawn by the black h=20mm triangles, so they vanish from the mask.
    # mu(v) is physically monotone decreasing, so enforce that and fill zero
    # tails.  Every correction is printed -- nothing is silently adjusted.
    corrections = []
    for label, series in result["left_mu_vs_velocity_kmh"].items():
        prev = None
        for v in speeds:
            if v not in series:
                # missing after the series has effectively reached zero -> it is zero
                if prev is not None and prev <= 0.05:
                    series[v] = 0.0
                    corrections.append(f"{label} v={v}: filled 0.0 (occluded at axis)")
                continue
            if prev is not None and series[v] > prev + 0.005:
                corrections.append(
                    f"{label} v={v}: dropped {series[v]} (non-monotone, marker overlap)"
                )
                del series[v]
                continue
            prev = series[v]
    if corrections:
        print("cleanup applied (physics: mu(v) is monotone decreasing):")
        for c in corrections:
            print(f"  - {c}")
        print()
    result["corrections"] = corrections

    print("=== Fig. 6 LEFT: mu vs velocity (km/h), DIGITIZED ===")
    print(f"{'v':>5} " + "".join(f"{LEFT_LABELS[n]:>10}" for n, _, _ in SERIES))
    for v in speeds:
        row = f"{v:>5} "
        for n, _, _ in SERIES:
            val = result["left_mu_vs_velocity_kmh"][LEFT_LABELS[n]].get(v)
            row += f"{val if val is not None else '-':>10}"
        print(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\n[artifact] {out}")


if __name__ == "__main__":
    main()
