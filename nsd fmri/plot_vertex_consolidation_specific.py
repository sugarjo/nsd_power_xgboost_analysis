"""Vertices where consolidated images differ from BOTH new and recently seen images, in the same direction.

Test per vertex: consolidated > max(new, recent)  or  consolidated < min(new, recent).
Input: every participant's vertex t maps from plot_vertex_vs_new.py (correct
trials, raw betas, session means removed, pooled across sessions):
  a = t(consolidated hits > correct rejections)
  b = t(consolidated hits > recent hits)
Right hemisphere mapped onto the left (fsaverage xhemi) and averaged with it.

  conjunction t per participant: min(a, b) where both are positive,
                                 max(a, b) where both are negative, else 0
  group test (8 participants): one-sided one-sample t-tests of a and of b in
      the same direction; conjunction p = the larger of the two (both must hold);
      the smaller of the 'above both' and 'below both' p values is doubled
      (two directions) and FDR-corrected (Benjamini-Hochberg) across vertices.
  consistency: participants whose a and b both lie in the group direction.

Written to --out-dir:
  nsd_vertex_consolidation_specific[_maxlag<d>d].png
  nsd_vertex_consolidation_specific[_maxlag<d>d]_parcels.csv
      per parcel: % of vertices significant above / below both, mean conjunction t
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shap analysis"))
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import DIVERGING, INK, INK_MUTED, fdr_bh, load_flatmap  # noqa: E402
from plot_paper_vertex import draw_t  # noqa: E402
from plot_vertex_bilateral import label_parcels, rh_to_lh  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--max-lag-days", type=float)
    ap.add_argument("--threshold", type=float, default=3.0, help="descriptive map: |mean conjunction t| shown above")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--p-show", type=float, default=0.01, help="panel B shows uncorrected p below this "
                    "when no vertex survives FDR")
    ap.add_argument("--n-labels", type=int, default=5)
    args = ap.parse_args()
    window = f"_maxlag{args.max_lag_days:g}d" if args.max_lag_days is not None else ""

    d = np.load(args.data_dir / f"nsd_vertex_vs_new{window}_tmaps.npz")
    idx = rh_to_lh(args.cache)
    geo = load_flatmap(args.cache)
    cortex = geo.parcel != ""

    def bilateral(name):
        return (d[f"{name}__lh"] + d[f"{name}__rh"][:, idx]) / 2          # participants x left vertices

    a, b = bilateral("consolidated_vs_new"), bilateral("consolidated_vs_recent")
    n = len(a)
    conj = np.where((a > 0) & (b > 0), np.minimum(a, b), np.where((a < 0) & (b < 0), np.maximum(a, b), 0.0))
    conj_mean = conj.mean(0)
    conj_mean[~cortex] = 0

    p = {}
    for side in ("greater", "less"):
        pa = sps.ttest_1samp(a, 0, alternative=side).pvalue
        pb = sps.ttest_1samp(b, 0, alternative=side).pvalue
        p[side] = np.maximum(pa, pb)                                          # both must hold
    above = p["greater"] < p["less"]
    p_dir = np.minimum(np.minimum(p["greater"], p["less"]) * 2, 1)
    p_dir[~cortex] = np.nan
    q = fdr_bh(pd.Series(p_dir)).to_numpy()
    sig = (q < args.q) & cortex
    t_a, t_b = sps.ttest_1samp(a, 0).statistic, sps.ttest_1samp(b, 0).statistic
    group_min_t = np.where(above, np.minimum(t_a, t_b), np.maximum(t_a, t_b))     # conjunction group t
    consistent = np.where(above, ((a > 0) & (b > 0)).sum(0), ((a < 0) & (b < 0)).sum(0))

    parcel = pd.Series(geo.parcel)
    table = pd.DataFrame({
        "pct_vertices_above_both": pd.Series(sig & above).groupby(parcel).mean() * 100,
        "pct_vertices_below_both": pd.Series(sig & ~above).groupby(parcel).mean() * 100,
        "mean_conjunction_t": pd.Series(conj_mean).groupby(parcel).mean(),
        "mean_participants_consistent": pd.Series(consistent).groupby(parcel).mean(),
    }).drop("", errors="ignore")
    table.index.name = "parcel"
    table.round(2).sort_values("mean_conjunction_t").to_csv(
        args.out_dir / f"nsd_vertex_consolidation_specific{window}_parcels.csv")
    print(f"{sig.sum()} of {cortex.sum()} vertices with q < {args.q}: {int((sig & above).sum())} above both, "
          f"{int((sig & ~above).sum())} below both")
    for col in ("pct_vertices_above_both", "pct_vertices_below_both"):
        top = table[col].sort_values(ascending=False)
        print(col, top[top > 0].head(10).round(1).to_dict())

    fig, axes = plt.subplots(1, 3, figsize=(18.6, 7.6), facecolor="white",
                             gridspec_kw=dict(wspace=0.03, left=0.01, right=0.99, top=0.84, bottom=0.22))
    # A: descriptive mean conjunction t
    vmax = 10
    draw_t(axes[0], geo, conj_mean, vmax, args.threshold, geo.border)
    ranked = table["mean_conjunction_t"].sort_values()
    label_parcels(axes[0], geo, [r for r in ranked.index[-args.n_labels:][::-1] if ranked[r] > 0]
                  + [r for r in ranked.index[:args.n_labels] if ranked[r] < 0])
    axes[0].set_title(f"A   Mean conjunction t (shown where |t| > {args.threshold:g})", fontsize=11.5, color=INK,
                      loc="left")
    # B: group test, FDR (uncorrected p if nothing survives)
    if sig.any():
        shown, what = np.where(sig, group_min_t, 0.0), f"vertices with q < {args.q} ({sig.sum()})"
    else:
        unc = (p_dir < args.p_show) & cortex
        shown = np.where(unc, group_min_t, 0.0)
        what = f"uncorrected p < {args.p_show:g} ({unc.sum()}); none with q < {args.q}"
    draw_t(axes[1], geo, shown, vmax, 0, geo.border)
    lab = pd.Series(shown).groupby(parcel).mean().drop("", errors="ignore").sort_values()
    label_parcels(axes[1], geo, [r for r in lab.index[-args.n_labels:][::-1] if lab[r] > 0]
                  + [r for r in lab.index[:args.n_labels] if lab[r] < 0])
    axes[1].set_title(f"B   Group conjunction t, {what}", fontsize=11.5,
                      color=INK, loc="left")
    # C: consistency across participants (signed by group direction, shown for 6+ of 8)
    cons_signed = np.where(above, consistent, -consistent).astype(float)
    cons_signed[np.abs(cons_signed) < 6] = 0
    draw_t(axes[2], geo, cons_signed, n, 0.5, geo.border)
    axes[2].set_title(f"C   Participants with consolidated beyond both (≥ 6 of {n})", fontsize=11.5, color=INK,
                      loc="left")

    for ax, (norm, label) in zip(axes, [(Normalize(-vmax, vmax), "mean conjunction t (participants)"),
                                        (Normalize(-vmax, vmax), f"group conjunction t ({n - 1} df)"),
                                        (Normalize(-n, n), "participants (sign = direction)")]):
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + pos.width * 0.2, pos.y0 - 0.05, pos.width * 0.6, 0.022])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax, orientation="horizontal",
                          extend="both" if ax is not axes[2] else "neither")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
        cb.set_label(label, fontsize=9, color=INK)
    fig.suptitle("NSD fMRI: where consolidated images differ from both new and recently seen images",
                 fontsize=15, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.08,
             "Red: consolidated > max(new, recent); blue: consolidated < min(new, recent). Per participant and vertex: "
             "Welch t of consolidated hits vs correct rejections (new) and vs recent hits, correct trials, raw betas,\n"
             "session means removed, pooled across sessions; right hemisphere mapped onto the left (xhemi) and averaged. "
             "Conjunction t = the smaller of the two t values when both share a sign, else 0. B: one-sided t-tests "
             f"across the {n} participants\nfor both comparisons in the same direction, conjunction p = the larger p, "
             f"doubled for two directions, Benjamini-Hochberg FDR across vertices. Recent: last shown ≤ 3 h before; "
             "consolidated: > 3 h"
             + (f" and ≤ {args.max_lag_days:g} days" if args.max_lag_days is not None else "") + ".",
             fontsize=8.5, color=INK_MUTED, va="top")
    path = args.out_dir / f"nsd_vertex_consolidation_specific{window}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
