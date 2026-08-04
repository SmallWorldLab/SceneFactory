#!/usr/bin/env python3
"""
Plot the brake-friction-sweep results from the acquired CSV numbers.

Standalone (no Isaac Sim): reads brake_sweep_summary.csv + brake_sweep_trajectory.csv
and renders a clean 3-panel figure for the paper:

  A  stopping distance vs mu     (measured vs Coulomb theory v0^2/2mu g)
  B  mean deceleration vs mu     (measured vs ideal mu*g and locked-skid 0.8*mu*g)
  C  braking speed vs time       (family over mu, one-hue sequential ramp)

Usage:
  python scripts/plot_brake_sweep.py [RUN_DIR]
    RUN_DIR defaults to the latest logs/rsl_rl/brake_friction_sweep/* run.
  python scripts/plot_brake_sweep.py --summary <csv> --trajectory <csv> --out <png>
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

# ---- Validated palette (dataviz reference instance, light surface) -----------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"   # axes / reference lines
GRID      = "#e1e0d9"
MEASURED  = "#2a78d6"   # categorical slot 1 (blue) -- the measured data
NHTSA_CLR = "#eb6834"   # categorical slot 2 (orange) -- empirical ground truth

# NHTSA/VRTC 1999 (braketst.pdf, Fig 3.1) shortest stopping distance from 100 km/h,
# real cars (Neon/Taurus/Town Car x 3 drivers), speed-corrected to 100 km/h.
# The 0.86/0.66 are NHTSA's *peak braking coefficient* (ASTM E1337 skid trailer) for
# dry/wet asphalt -- a tire-on-pavement measurement, NOT the same quantity as the
# sim's Coulomb mu. We use them only as the nominal friction we fed the sim, and we
# compare the *stopping distance*, not the coefficient. Ranges: full = all conditions,
# no_abs = ABS-off (driver threshold braking, the condition closest to our no-ABS sim).
NHTSA_MEASURED = {
    "Dry asphalt":  {"nominal_mu": 0.86, "full": (46.0, 69.0), "no_abs": (48.0, 69.0)},
    "Wet asphalt":  {"nominal_mu": 0.66, "full": (48.0, 95.0), "no_abs": (52.0, 95.0)},
}
V0_TARGET_MPS = 27.78  # 100 km/h, for speed-correcting sim distances (SAE J299, as NHTSA did)
# sequential blue ramp for the mu family. Ordinal use (11 discrete series) -> start
# no lighter than step 250 (#86b6ef) so even the lightest curve clears contrast on
# the light surface (dataviz palette ordinal rule).
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]
G = 9.80665


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _ramp_color(frac):
    """frac in [0,1] -> color from BLUE_RAMP (0 = light, 1 = dark)."""
    i = int(round(frac * (len(BLUE_RAMP) - 1)))
    return BLUE_RAMP[max(0, min(len(BLUE_RAMP) - 1, i))]


def _style_ax(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def make_figure(summary_rows, traj_rows, out_path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    })

    mus    = [_num(r["mu"]) for r in summary_rows]
    dmeas  = [_num(r["stopping_distance_m"]) for r in summary_rows]
    dth    = [_num(r["theoretical_distance_m"]) for r in summary_rows]
    ameas  = [_num(r["mean_decel_mps2"]) for r in summary_rows]
    v0s    = [_num(r["v0_entry_mps"]) for r in summary_rows]
    stopped = [str(r["stopped"]).strip().lower() == "true" for r in summary_rows]

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 4.9))

    # ---- Panel A: stopping distance vs mu -----------------------------------
    _style_ax(axA)
    mm = [(m, d) for m, d, s in zip(mus, dmeas, stopped) if d is not None and s]
    tt = [(m, d) for m, d in zip(mus, dth) if d is not None]
    if tt:
        axA.plot([m for m, _ in tt], [d for _, d in tt], ls="--", lw=1.6,
                 color=MUTED, label="Coulomb theory  v₀²/(2μg)", zorder=2)
    if mm:
        axA.plot([m for m, _ in mm], [d for _, d in mm], "-", lw=2.0,
                 color=MEASURED, zorder=3)
        axA.plot([m for m, _ in mm], [d for _, d in mm], "o", ms=7,
                 color=MEASURED, mec=SURFACE, mew=1.2, label="measured (sim)",
                 zorder=4)
    # (NHTSA comparison lives in its own distance-vs-surface figure, not here --
    #  Panel A is the sim's internal physics check vs Coulomb theory only.)
    # mu = 0 censored -> annotate
    if any((not s) for s in stopped):
        axA.annotate("μ=0: never stops (∞)", xy=(0.02, 0.94),
                     xycoords="axes fraction", color=INK_2, fontsize=9,
                     ha="left", va="top")
    axA.set_xlabel("surface friction  μ")
    axA.set_ylabel("stopping distance (m)")
    axA.set_title("A · Stopping distance vs μ", fontsize=11, loc="left")
    axA.invert_xaxis()
    axA.legend(frameon=False, fontsize=9, loc="upper right",
               labelcolor=INK_2)

    # ---- Panel B: mean deceleration vs mu -----------------------------------
    _style_ax(axB)
    mu_grid = [m for m in mus if m is not None and m > 0]
    if mu_grid:
        xg = sorted(mu_grid)
        axB.plot(xg, [G * m for m in xg], ls="--", lw=1.4, color=MUTED,
                 label="ideal  μg", zorder=2)
        axB.plot(xg, [0.8 * G * m for m in xg], ls=":", lw=1.6, color=MUTED,
                 label="locked skid  0.8·μg", zorder=2)
    bm = [(m, a) for m, a, s in zip(mus, ameas, stopped) if a is not None and s]
    if bm:
        axB.plot([m for m, _ in bm], [a for _, a in bm], "-", lw=2.0,
                 color=MEASURED, zorder=3)
        axB.plot([m for m, _ in bm], [a for _, a in bm], "o", ms=7,
                 color=MEASURED, mec=SURFACE, mew=1.2, label="measured (sim)",
                 zorder=4)
    axB.set_xlabel("surface friction  μ")
    axB.set_ylabel("mean deceleration (m/s²)")
    axB.set_title("B · Braking authority vs μ", fontsize=11, loc="left")
    axB.invert_xaxis()
    axB.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_2)

    # ---- Panel C: speed vs time family (sequential over mu) ------------------
    _style_ax(axC)
    umus = sorted({_num(r["mu"]) for r in traj_rows if _num(r["mu"]) is not None},
                  reverse=True)
    mu_lo, mu_hi = (min(umus), max(umus)) if umus else (0.0, 1.0)
    for mu in umus:
        pts = [(_num(r["time_s"]), _num(r["speed_mps"])) for r in traj_rows
               if _num(r["mu"]) == mu]
        pts = [(t, s) for t, s in pts if t is not None and s is not None]
        if not pts:
            continue
        frac = (mu - mu_lo) / (mu_hi - mu_lo) if mu_hi > mu_lo else 1.0  # high mu -> dark
        axC.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.5,
                 color=_ramp_color(frac), label=f"μ={mu:g}")
    axC.set_xlabel("time since brake onset (s)")
    axC.set_ylabel("speed (m/s)")
    axC.set_title("C · Braking speed vs time", fontsize=11, loc="left")
    axC.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right",
               labelcolor=INK_2)

    v0s_ok = [v for v in v0s if v is not None]
    v0_med = sorted(v0s_ok)[len(v0s_ok) // 2] if v0s_ok else 0.0
    fig.suptitle(f"PhysX vehicle braking — friction sweep "
                 f"(injected v₀ ≈ {v0_med:.1f} m/s = {v0_med * 3.6:.0f} km/h, full brake)",
                 color=INK, fontsize=12.5, x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    # Also a vector copy for the paper.
    pdf = str(Path(out_path).with_suffix(".pdf"))
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    print(f"[plot] wrote {pdf}")


def nhtsa_figure(summary_rows, run_dir):
    """External-validation figure: stopping distance by SURFACE, sim vs NHTSA.
    No mu axis -- the validated quantity is stopping distance from 100 km/h."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle
        from matplotlib.lines import Line2D
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable, skipping NHTSA figure ({e}).")
        return

    by_mu = {round(_num(r["mu"]), 3): r for r in summary_rows if _num(r["mu"]) is not None}
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    _style_ax(ax)
    surfaces = list(NHTSA_MEASURED.keys())
    w = 0.42
    for x, surf in enumerate(surfaces):
        d = NHTSA_MEASURED[surf]
        flo, fhi = d["full"]
        alo, ahi = d["no_abs"]
        ax.add_patch(Rectangle((x - w / 2, flo), w, fhi - flo, facecolor=NHTSA_CLR,
                               alpha=0.15, edgecolor="none", zorder=1))
        ax.add_patch(Rectangle((x - w / 2, alo), w, ahi - alo, facecolor=NHTSA_CLR,
                               alpha=0.34, edgecolor="none", zorder=2))
        # sim point at the nominal mu, speed-corrected to 100 km/h (as NHTSA did)
        row = by_mu.get(round(d["nominal_mu"], 3))
        if row and _num(row["stopping_distance_m"]) and _num(row["v0_entry_mps"]):
            d_raw = _num(row["stopping_distance_m"])
            v0 = _num(row["v0_entry_mps"])
            d_cor = d_raw * (V0_TARGET_MPS / v0) ** 2
            ax.plot(x, d_cor, marker="D", ms=13, color=MEASURED, mec="white",
                    mew=1.4, zorder=6)
            ax.annotate(f"sim {d_cor:.0f} m", (x, d_cor), xytext=(15, 0),
                        textcoords="offset points", va="center", color=MEASURED,
                        fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(surfaces)))
    ax.set_xticklabels([f"{s}\n(NHTSA peak μ {NHTSA_MEASURED[s]['nominal_mu']})"
                        for s in surfaces])
    ax.set_ylabel("stopping distance from 100 km/h (m)")
    ax.set_ylim(40, 100)
    ax.set_xlim(-0.6, len(surfaces) - 0.4)
    ax.set_title("External validation — 100 km/h stopping distance vs NHTSA (VRTC 1999)",
                 fontsize=12, loc="left", color=INK)
    handles = [
        Patch(facecolor=NHTSA_CLR, alpha=0.34, label="NHTSA measured — no ABS (threshold braking)"),
        Patch(facecolor=NHTSA_CLR, alpha=0.15, label="NHTSA measured — all conditions"),
        Line2D([], [], marker="D", ms=11, color=MEASURED, mec="white", mew=1.2,
               ls="none", label="SceneFactory sim (no-ABS locked skid)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left",
              labelcolor=INK_2)
    fig.text(0.01, 0.005,
             "Sim friction set to NHTSA's reported peak coefficient as the nominal input "
             "(peak coefficient ≠ the sim's Coulomb μ); distances speed-corrected to "
             "100 km/h (SAE J299). Validated quantity is stopping distance, not μ.",
             fontsize=7.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path(run_dir) / "brake_nhtsa_validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(str(Path(out).with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def _read_replicates(rep_dir):
    """Read all summary_seed*.csv in a replicates dir -> {mu: [rows...]}, plus a
    per-seed list. Each row is a dict from one seed's sweep at one mu."""
    rep_dir = Path(rep_dir)
    files = sorted(rep_dir.glob("summary_seed*.csv"))
    if not files:
        raise SystemExit(f"No summary_seed*.csv in {rep_dir}")
    by_mu = {}
    n_seeds = 0
    for f in files:
        n_seeds += 1
        for r in _read_csv(f):
            mu = _num(r["mu"])
            if mu is None:
                continue
            by_mu.setdefault(round(mu, 3), []).append(r)
    return by_mu, n_seeds, files


def _valid_inj(r):
    """A physically-valid replicate: injection took (v0 near 100 km/h), the car
    stopped, and the stop is not a solver-explosion artifact. We drop stops whose
    distance is physically impossible vs Coulomb theory (ratio far from ~1) -- the
    analog of NHTSA excluding stops with wheel lock-up / over-pedal-force."""
    v0 = _num(r["v0_entry_mps"])
    ratio = _num(r["distance_ratio"])
    return (v0 is not None and 24.0 <= v0 <= 30.0
            and str(r["stopped"]).strip().lower() == "true"
            and _num(r["stopping_distance_m"]) is not None
            and ratio is not None and 0.6 <= ratio <= 2.0)


def aggregate_replicates(rep_dir):
    """Produce spread figures across replicate seeds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle
        from matplotlib.lines import Line2D
        import random
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e})."); return

    rep_dir = Path(rep_dir)
    by_mu, n_seeds, files = _read_replicates(rep_dir)
    rng = random.Random(0)
    mus = sorted(by_mu.keys(), reverse=True)
    print(f"[plot] {n_seeds} replicate seeds, {len(mus)} mu values from {rep_dir}")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})

    # ---- Figure 1: stopping distance vs mu, all replicate points + mean -------
    fig, ax = plt.subplots(figsize=(9, 5.6))
    _style_ax(ax)
    theo_x, theo_y = [], []
    for mu in mus:
        rows = [r for r in by_mu[mu] if _valid_inj(r)]
        ds = [_num(r["stopping_distance_m"]) for r in rows]
        v0s = [_num(r["v0_entry_mps"]) for r in rows]
        if not ds:
            continue
        xj = [mu + rng.uniform(-0.006, 0.006) for _ in ds]
        ax.plot(xj, ds, "o", ms=4, color=MEASURED, alpha=0.35, mec="none", zorder=3)
        m = sum(ds) / len(ds)
        ax.plot([mu], [m], "_", ms=16, color=MEASURED, mew=2.5, zorder=4)
        v0m = sum(v0s) / len(v0s)
        theo_x.append(mu); theo_y.append((v0m * v0m) / (2.0 * mu * G))
    if theo_x:
        ax.plot(theo_x, theo_y, "--", lw=1.5, color=MUTED, zorder=2,
                label="Coulomb theory  v₀²/(2μg)")
    ax.plot([], [], "o", ms=6, color=MEASURED, alpha=0.5, mec="none",
            label=f"sim replicates ({n_seeds} seeds)")
    ax.plot([], [], "_", ms=12, color=MEASURED, mew=2.5, label="replicate mean")
    ax.set_xlabel("surface friction  μ")
    ax.set_ylabel("stopping distance from 100 km/h (m)")
    ax.set_title(f"Braking friction sweep — {n_seeds} replicate seeds per μ",
                 fontsize=12, loc="left", color=INK)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    out1 = rep_dir / "brake_replicates_vs_mu.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    fig.savefig(str(out1.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out1}")

    # ---- Figure 2: NHTSA validation with sim replicate spread ----------------
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    _style_ax(ax)
    surfaces = list(NHTSA_MEASURED.keys())
    w = 0.42
    for x, surf in enumerate(surfaces):
        d = NHTSA_MEASURED[surf]
        flo, fhi = d["full"]; alo, ahi = d["no_abs"]
        ax.add_patch(Rectangle((x - w / 2, flo), w, fhi - flo, facecolor=NHTSA_CLR,
                               alpha=0.15, edgecolor="none", zorder=1))
        ax.add_patch(Rectangle((x - w / 2, alo), w, ahi - alo, facecolor=NHTSA_CLR,
                               alpha=0.34, edgecolor="none", zorder=2))
        rows = [r for r in by_mu.get(round(d["nominal_mu"], 3), []) if _valid_inj(r)]
        # speed-correct each replicate to 100 km/h (SAE J299), as NHTSA did
        dcor = [_num(r["stopping_distance_m"]) * (V0_TARGET_MPS / _num(r["v0_entry_mps"])) ** 2
                for r in rows]
        if not dcor:
            continue
        xj = [x + rng.uniform(-0.10, 0.10) for _ in dcor]
        ax.plot(xj, dcor, "o", ms=6, color=MEASURED, alpha=0.6, mec="white", mew=0.6,
                zorder=6)
        m = sum(dcor) / len(dcor)
        ax.plot([x - w / 2, x + w / 2], [m, m], color=MEASURED, lw=2.5, zorder=7)
        lo, hi = min(dcor), max(dcor)
        ax.annotate(f"sim {m:.0f} m\n(n={len(dcor)}, {lo:.0f}–{hi:.0f})",
                    (x + w / 2, m), xytext=(12, 0), textcoords="offset points",
                    va="center", color=MEASURED, fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(surfaces)))
    ax.set_xticklabels([f"{s}\n(NHTSA peak μ {NHTSA_MEASURED[s]['nominal_mu']})"
                        for s in surfaces])
    ax.set_ylabel("stopping distance from 100 km/h (m)")
    ax.set_ylim(40, 100); ax.set_xlim(-0.6, len(surfaces) - 0.4)
    ax.set_title(f"External validation vs NHTSA — {n_seeds} sim replicate seeds",
                 fontsize=12, loc="left", color=INK)
    handles = [
        Patch(facecolor=NHTSA_CLR, alpha=0.34, label="NHTSA measured — no ABS (threshold)"),
        Patch(facecolor=NHTSA_CLR, alpha=0.15, label="NHTSA measured — all conditions"),
        Line2D([], [], marker="o", ms=7, color=MEASURED, mec="white", mew=0.6,
               ls="none", label="SceneFactory sim replicates"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper left", labelcolor=INK_2)
    fig.text(0.01, 0.005,
             "Each dot = one seed's stop, speed-corrected to 100 km/h (SAE J299). Sim friction "
             "set to NHTSA's nominal peak coefficient (peak coeff ≠ sim Coulomb μ). Validated "
             "quantity is stopping distance.", fontsize=7.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out2 = rep_dir / "brake_nhtsa_validation_replicates.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    fig.savefig(str(out2.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out2}")


def _latest_run_dir():
    cands = sorted(glob.glob("logs/rsl_rl/brake_friction_sweep/*"))
    if not cands:
        raise SystemExit("No brake_friction_sweep run dirs found under logs/rsl_rl/.")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="brake_friction_sweep run dir (default: latest)")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--replicates", default=None,
                    help="A replicates dir of summary_seed*.csv -> spread figures.")
    a = ap.parse_args()

    if a.replicates:
        aggregate_replicates(a.replicates)
        return

    run_dir = a.run_dir or _latest_run_dir()
    summary = a.summary or os.path.join(run_dir, "brake_sweep_summary.csv")
    traj    = a.trajectory or os.path.join(run_dir, "brake_sweep_trajectory.csv")
    out     = a.out or os.path.join(run_dir, "brake_sweep_figure.png")

    print(f"[plot] run_dir   = {run_dir}")
    print(f"[plot] summary   = {summary}")
    print(f"[plot] trajectory= {traj}")
    srows = _read_csv(summary)
    make_figure(srows, _read_csv(traj), out)
    nhtsa_figure(srows, run_dir)


if __name__ == "__main__":
    main()
