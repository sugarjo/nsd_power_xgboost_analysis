"""Consolidated beyond both new and recent: mean conjunction t, next to the same map with shuffled labels.

Fig. 4b style (Allen et al. 2022): per participant and vertex a t value,
simple mean over participants, shown where |t| > --threshold; a shuffled-label
map as control instead of a group-level test.

Per participant and vertex (correct trials, raw betas, session means removed,
pooled across sessions, right hemisphere mapped onto the left with fsaverage
xhemi and averaged):
  a = Welch t(consolidated hits > correct rejections)
  b = Welch t(consolidated hits > recent hits)
  conjunction t = min(a, b) if both > 0, max(a, b) if both < 0, else 0
Shuffled: the same with labels shuffled within each session
(extract_vertex_stats.py groups perm_a vs perm_b: hits vs correct rejections;
perm_d vs perm_c: consolidated vs recent hits).

Written to --out-dir: nsd_vertex_consolidation_shuffle[_maxlag<d>d].png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shap analysis"))
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import DIVERGING, INK, INK_MUTED, load_flatmap  # noqa: E402
from plot_paper_vertex import SUBJECTS, demean_sessions, draw_t, load_subject, welch_t  # noqa: E402
from plot_vertex_bilateral import label_parcels, rh_to_lh  # noqa: E402

PAIRS = {  # (a, b) as (positive, negative) groups
    "real": ((["consolidated"], ["cr"]), (["consolidated"], ["recent"])),
    "shuffled": ((["perm_a"], ["perm_b"]), (["perm_d"], ["perm_c"])),
}


def conjunction(a, b):
    return np.where((a > 0) & (b > 0), np.minimum(a, b), np.where((a < 0) & (b < 0), np.maximum(a, b), 0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--max-lag-days", type=float)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--vmax", type=float, default=10.0)
    ap.add_argument("--n-labels", type=int, default=5)
    args = ap.parse_args()
    window = f"_maxlag{args.max_lag_days:g}d" if args.max_lag_days is not None else ""

    idx = rh_to_lh(args.cache)
    geo = load_flatmap(args.cache)
    cortex = geo.parcel != ""
    conj = {k: [] for k in PAIRS}
    for subject in SUBJECTS:
        per_hemi = {k: {} for k in PAIRS}
        for hemi in ("lh", "rh"):
            _, n, s, ss = load_subject(args.data_dir, subject, hemi, window)
            sd, ssd = demean_sessions(n, s, ss)
            for k, (pa, pb) in PAIRS.items():
                per_hemi[k][hemi] = [welch_t(n.sum(0), sd.sum(0), ssd.sum(0), *pa),
                                     welch_t(n.sum(0), sd.sum(0), ssd.sum(0), *pb)]
        for k in PAIRS:
            a = (per_hemi[k]["lh"][0] + per_hemi[k]["rh"][0][idx]) / 2
            b = (per_hemi[k]["lh"][1] + per_hemi[k]["rh"][1][idx]) / 2
            conj[k].append(conjunction(a, b))
        print(f"subj{subject:02d}", flush=True)

    parcel = pd.Series(geo.parcel)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6), facecolor="white",
                             gridspec_kw=dict(wspace=0.03, left=0.01, right=0.99, top=0.84, bottom=0.22))
    for ax, (k, title) in zip(axes, [("real", "A   Consolidated beyond both new and recent"),
                                    ("shuffled", "B   Same, labels shuffled within session")]):
        m = np.mean(conj[k], 0)
        m[~cortex] = 0
        draw_t(ax, geo, m, args.vmax, args.threshold, geo.border)
        frac = (np.abs(m[cortex]) > args.threshold).mean()
        if k == "real":
            ranked = pd.Series(m).groupby(parcel).mean().drop("", errors="ignore").sort_values()
            label_parcels(ax, geo, [r for r in ranked.index[-args.n_labels:][::-1] if ranked[r] > 0]
                          + [r for r in ranked.index[:args.n_labels] if ranked[r] < 0])
        ax.set_title(f"{title}\nmean conjunction t, |t| > {args.threshold:g} on {frac:.1%} of cortex",
                     fontsize=11.5, color=INK, loc="left")
        print(f"{k}: |mean conjunction t| > {args.threshold:g} on {frac:.2%} of cortex; max |t| "
              f"{np.abs(m[cortex]).max():.1f}")
    pos = axes[0].get_position(), axes[1].get_position()
    mid = (pos[0].x0 + pos[1].x1) / 2
    cax = fig.add_axes([mid - 0.15, pos[0].y0 - 0.05, 0.3, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-args.vmax, args.vmax), cmap=DIVERGING), cax=cax,
                      orientation="horizontal", extend="both")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    cb.set_label("mean conjunction t (8 participants)", fontsize=9, color=INK)
    fig.suptitle("NSD fMRI: consolidated > max(new, recent) (red) or < min(new, recent) (blue)",
                 fontsize=14, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.08,
             "Per participant and vertex: Welch t of consolidated hits vs correct rejections (new) and vs recent hits; "
             "correct trials, raw betas, session means removed, pooled across sessions;\nright hemisphere mapped onto "
             "the left (xhemi) and averaged. Conjunction t = the smaller of the two when both share a sign, else 0; "
             f"mean of 8 participants, shown where |t| > {args.threshold:g}\n(as in Allen et al. 2022, Fig. 4b). "
             "B: labels shuffled within each session (hits vs correct rejections; consolidated vs recent hits). "
             "Recent: last shown ≤ 3 h before; consolidated: > 3 h"
             + (f" and ≤ {args.max_lag_days:g} days" if args.max_lag_days is not None else "") + ".",
             fontsize=8.5, color=INK_MUTED, va="top")
    path = args.out_dir / f"nsd_vertex_consolidation_shuffle{window}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
