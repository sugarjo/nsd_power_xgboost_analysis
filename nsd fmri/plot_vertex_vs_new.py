"""Is recent vs consolidated only recency? Recent and consolidated hits each against new images.

Per participant and vertex (raw betas, fsaverage, correct trials, from
extract_vertex_stats.py): each session's mean over its correct trials is
removed, trials are pooled across sessions, and a Welch t is computed for
  recent hits > correct rejections (new images)
  consolidated hits > correct rejections (new images)
  consolidated hits > recent hits
Hemispheres are averaged (right mapped onto left with the fsaverage xhemi
registration, see plot_vertex_bilateral.py), then the participants (simple
mean). Shown where |t| > --threshold on one colour scale; the parcels with
the largest positive and negative mean t are named.

Written to --out-dir:
  nsd_vertex_vs_new[_maxlag<d>d].png
  nsd_vertex_vs_new[_maxlag<d>d]_parcels.csv           parcel x contrast, bilateral group mean t
  nsd_vertex_vs_new[_maxlag<d>d]_parcels_subjects.csv  same per participant
and every participant's left / right t maps to
  data/nsd/nsd_vertex_vs_new[_maxlag<d>d]_tmaps.npz
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

CONTRASTS = {  # name: (title, positive groups, negative groups)
    "recent_vs_new": ("Recent hits > new (correct rejections)", ["recent"], ["cr"]),
    "consolidated_vs_new": ("Consolidated hits > new (correct rejections)", ["consolidated"], ["cr"]),
    "consolidated_vs_recent": ("Consolidated hits > recent hits", ["consolidated"], ["recent"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--max-lag-days", type=float, help="use the stats extracted with this lag window")
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--vmax", type=float, default=30.0)
    ap.add_argument("--n-labels", type=int, default=5)
    args = ap.parse_args()
    window = f"_maxlag{args.max_lag_days:g}d" if args.max_lag_days is not None else ""

    tmaps = {name: {"lh": [], "rh": []} for name in CONTRASTS}
    for subject in SUBJECTS:
        for hemi in ("lh", "rh"):
            sessions, n, s, ss = load_subject(args.data_dir, subject, hemi, window)
            sd, ssd = demean_sessions(n, s, ss)
            for name, (_, pos, neg) in CONTRASTS.items():
                tmaps[name][hemi].append(welch_t(n.sum(0), sd.sum(0), ssd.sum(0), pos, neg))
        print(f"subj{subject:02d}: {len(sessions)} sessions", flush=True)
    subjects = [f"subj{s:02d}" for s in SUBJECTS]
    np.savez_compressed(args.data_dir / f"nsd_vertex_vs_new{window}_tmaps.npz", subjects=np.array(subjects),
                        **{f"{name}__{h}": np.array(tmaps[name][h], np.float32) for name in CONTRASTS for h in ("lh", "rh")})

    idx = rh_to_lh(args.cache)
    geo = load_flatmap(args.cache)
    parcel = pd.Series(geo.parcel)
    group, table, subj_table = {}, {}, {}
    for name in CONTRASTS:
        bilat = (np.array(tmaps[name]["lh"]) + np.array(tmaps[name]["rh"])[:, idx]) / 2
        bilat[:, geo.parcel == ""] = 0
        group[name] = bilat.mean(0)
        per_parcel = pd.DataFrame(bilat.T, columns=subjects).groupby(parcel).mean().drop("", errors="ignore")
        table[name] = per_parcel.mean(axis=1)
        for s in subjects:
            subj_table[f"{name}__{s}"] = per_parcel[s]
    table = pd.DataFrame(table)
    table.insert(len(table.columns), "n_positive__consolidated_vs_new",
                 (pd.DataFrame({s: subj_table[f"consolidated_vs_new__{s}"] for s in subjects}) > 0).sum(axis=1))
    table.index.name = "parcel"
    table.round(3).to_csv(args.out_dir / f"nsd_vertex_vs_new{window}_parcels.csv")
    pd.DataFrame(subj_table).rename_axis("parcel").round(3).to_csv(
        args.out_dir / f"nsd_vertex_vs_new{window}_parcels_subjects.csv")

    fig, axes = plt.subplots(1, len(CONTRASTS), figsize=(6.2 * len(CONTRASTS), 7.4), facecolor="white",
                             gridspec_kw=dict(wspace=0.03, left=0.01, right=0.99, top=0.84, bottom=0.2))
    for ax, (name, (label, _, _)) in zip(axes, CONTRASTS.items()):
        draw_t(ax, geo, group[name], args.vmax, args.threshold, geo.border)
        ranked = table[name].sort_values()
        label_parcels(ax, geo, list(ranked.index[-args.n_labels:][::-1]) + list(ranked.index[:args.n_labels]))
        ax.set_title(label, fontsize=12, color=INK)
        shown = (np.abs(group[name][geo.parcel != ""]) > args.threshold).mean()
        print(f"{label}: |t| > {args.threshold:g} on {shown:.0%} of cortex; "
              f"top + {', '.join(ranked.index[-args.n_labels:][::-1])}; top - {', '.join(ranked.index[:args.n_labels])}")
    pos = axes[0].get_position(), axes[-1].get_position()
    mid = (pos[0].x0 + pos[1].x1) / 2
    cax = fig.add_axes([mid - 0.13, pos[0].y0 - 0.05, 0.26, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-args.vmax, args.vmax), cmap=DIVERGING), cax=cax,
                      orientation="horizontal", extend="both")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    cb.set_label("Welch t (mean of 8 participants, left + mirrored right hemisphere)", fontsize=9, color=INK)
    fig.suptitle("NSD fMRI: recently seen and consolidated images, each compared with new images",
                 fontsize=15, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.07,
             "Correct trials only (raw betas, fsaverage). Per participant and vertex: each session's mean removed, "
             "trials pooled across sessions, Welch t between trial types; right hemisphere mapped onto the left "
             "(xhemi) and averaged with it;\ngroup = mean of 8 participants, shown where "
             f"|t| > {args.threshold:g}. Recent: last shown ≤ 3 h before; consolidated: > 3 h"
             + (f" and ≤ {args.max_lag_days:g} days (the sEEG lag range)" if args.max_lag_days is not None else "")
             + f"; new = correct rejections. Named: the {args.n_labels} parcels with the largest positive and "
             "negative mean t.", fontsize=8.5, color=INK_MUTED, va="top")
    path = args.out_dir / f"nsd_vertex_vs_new{window}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
