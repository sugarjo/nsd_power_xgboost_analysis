"""Vertex t maps averaged over hemispheres (left + mirrored right), with the main parcels named.

Input: the per-participant t maps saved by plot_paper_vertex.py
(data/nsd/nsd_paper_vertex[_maxlag<d>d]_tmaps.npz). The right hemisphere is
mapped onto the left with FreeSurfer's left-right registration of fsaverage
(xhemi, via fsaverage_sym; files from the NSD bucket): every left vertex takes
the value of the nearest right vertex in fsaverage_sym space. Per participant
the two hemispheres are averaged, then the 8 participants (simple mean, as in
the paper). The hemispheres come from the same trials, so this reduces noise
but is not twice the sample.

Shown: pooled (paper) and pooled with session means removed, for both
contrasts, where |t| > --threshold. The --n-labels parcels with the largest
positive and the largest negative mean t (within the parcel) are named.

Written to --out-dir:
  nsd_paper_vertex_bilateral[_maxlag<d>d].png
  nsd_paper_vertex_bilateral[_maxlag<d>d]_parcels.csv   parcel x (contrast, column):
      mean over the parcel's vertices of the bilateral group t; per participant in
      ..._parcels_subjects.csv
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel.freesurfer as fs
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shap analysis"))
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import DIVERGING, INK, INK_MUTED, load_flatmap  # noqa: E402
from plot_paper_vertex import CONTRASTS, TITLES, draw_t  # noqa: E402

BUCKET = "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/freesurfer/fsaverage"
SHOWN = ["pooled", "demeaned"]


def rh_to_lh(cache: Path) -> np.ndarray:
    """Index of the right-hemisphere vertex homologous to each left vertex (fsaverage xhemi)."""
    files = {"lh": cache / "lh.fsaverage_sym.sphere.reg", "rh": cache / "xhemi" / "lh.fsaverage_sym.sphere.reg"}
    urls = {"lh": f"{BUCKET}/surf/lh.fsaverage_sym.sphere.reg", "rh": f"{BUCKET}/xhemi/surf/lh.fsaverage_sym.sphere.reg"}
    for h, f in files.items():
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(urls[h], f)
    lh, _ = fs.read_geometry(files["lh"])
    rh, _ = fs.read_geometry(files["rh"])
    return cKDTree(rh).query(lh)[1]


def label_parcels(ax, geo, parcels, color=INK):
    """Name parcels at a central vertex of their largest flat-map piece; each label takes the first
    candidate spot (above, below, sides, then further out) whose box overlaps no earlier label."""
    renderer = ax.figure.canvas.get_renderer()
    spots = [(0, 10, "center", "bottom"), (0, -10, "center", "top"), (14, 0, "left", "center"),
             (-14, 0, "right", "center")]
    for r in (24, 40, 56, 72):
        spots += [(0, r, "center", "bottom"), (0, -r, "center", "top"), (r, r / 2, "left", "center"),
                  (-r, r / 2, "right", "center"), (r, -r / 2, "left", "center"), (-r, -r / 2, "right", "center")]
    placed = []

    def overlap(box):
        return sum(max(0, min(box.x1, q.x1) - max(box.x0, q.x0)) * max(0, min(box.y1, q.y1) - max(box.y0, q.y0))
                   for q in placed)

    for name in parcels:
        sel = geo.parcel[geo.vno] == name
        if not sel.any():
            continue
        pts = geo.patch_xy[sel]
        xy = pts[np.argmin(((pts - np.median(pts, 0)) ** 2).sum(1))]
        best = None
        for dx, dy, ha, va in spots:
            ann = ax.annotate(name, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7.5,
                              color=color, arrowprops=dict(arrowstyle="-", color=INK, lw=0.5),
                              bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
            box = ann.get_window_extent(renderer).expanded(1.2, 1.4)
            cost = overlap(box)
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, ann, box)
            else:
                ann.remove()
            if cost == 0:
                break
        placed.append(best[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--max-lag-days", type=float)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--n-labels", type=int, default=5, help="parcels named per sign")
    args = ap.parse_args()
    window = f"_maxlag{args.max_lag_days:g}d" if args.max_lag_days is not None else ""

    d = np.load(args.data_dir / f"nsd_paper_vertex{window}_tmaps.npz")
    subjects = list(d["subjects"])
    idx = rh_to_lh(args.cache)
    geo = load_flatmap(args.cache)
    parcel = pd.Series(geo.parcel)

    group, rows, subj_rows = {}, {}, {}
    for name in CONTRASTS:
        for col in SHOWN:
            lh, rh = d[f"{name}__{col}__lh"], d[f"{name}__{col}__rh"]
            bilat = (lh + rh[:, idx]) / 2                                   # participants x left vertices
            bilat[:, (geo.parcel == "")] = 0
            group[name, col] = bilat.mean(0)
            per_parcel = pd.DataFrame(bilat.T, columns=subjects).groupby(parcel).mean().drop("", errors="ignore")
            rows[(name, col)] = per_parcel.mean(axis=1)
            for s in subjects:
                subj_rows[(name, col, s)] = per_parcel[s]
    table = pd.DataFrame(rows)
    table.columns = [f"{n}__{c}" for n, c in table.columns]
    table.index.name = "parcel"
    table.round(3).to_csv(args.out_dir / f"nsd_paper_vertex_bilateral{window}_parcels.csv")
    subj = pd.DataFrame(subj_rows)
    subj.columns = [f"{n}__{c}__{s}" for n, c, s in subj.columns]
    subj.index.name = "parcel"
    subj.round(3).to_csv(args.out_dir / f"nsd_paper_vertex_bilateral{window}_parcels_subjects.csv")

    fig, axes = plt.subplots(len(CONTRASTS), len(SHOWN), figsize=(13, 11.5), facecolor="white",
                             gridspec_kw=dict(wspace=0.03, hspace=0.18, left=0.06, right=0.99, top=0.9, bottom=0.17))
    for r, (name, (label, *_)) in enumerate(CONTRASTS.items()):
        for c, col in enumerate(SHOWN):
            ax = axes[r, c]
            draw_t(ax, geo, group[name, col], 40, args.threshold, geo.border)
            ranked = table[f"{name}__{col}"].sort_values()
            label_parcels(ax, geo, list(ranked.index[-args.n_labels:][::-1]) + list(ranked.index[:args.n_labels]))
            ax.set_title(TITLES[col], fontsize=11, color=INK)
            print(f"{label}, {TITLES[col]}: top + {', '.join(ranked.index[-args.n_labels:][::-1])}; "
                  f"top - {', '.join(ranked.index[:args.n_labels])}")
        axes[r, 0].text(-0.04, 0.5, f"{label}\nmean of {len(subjects)} participants, both hemispheres",
                        transform=axes[r, 0].transAxes, rotation=90, ha="right", va="center", fontsize=10, color=INK)
    pos = axes[-1, 0].get_position(), axes[-1, -1].get_position()
    mid = (pos[0].x0 + pos[1].x1) / 2
    cax = fig.add_axes([mid - 0.15, pos[0].y0 - 0.05, 0.3, 0.015])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-40, 40), cmap=DIVERGING), cax=cax,
                      orientation="horizontal", extend="both")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    cb.set_label("Welch t (mean of participants, left + mirrored right hemisphere)", fontsize=9, color=INK)
    fig.suptitle("NSD fMRI recognition memory, both hemispheres (method of Allen et al. 2022, Fig. 4b)",
                 fontsize=14, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.08,
             "Per participant and vertex: Welch t between correct trial types pooled across sessions (raw betas, "
             "fsaverage). Right hemisphere mapped onto the left with FreeSurfer's\nleft-right registration (xhemi, "
             "fsaverage_sym) and averaged with the left; group = mean of participants; shown where "
             f"|t| > {args.threshold:g}. Named: the {args.n_labels} HCP-MMP1 parcels with the\nlargest positive and "
             "negative mean t. Hits = recent + consolidated; recent ≤ 3 h, consolidated > 3 h since last shown."
             + (f" Only repeats ≤ {args.max_lag_days:g} days." if args.max_lag_days is not None else ""),
             fontsize=8.5, color=INK_MUTED, va="top")
    path = args.out_dir / f"nsd_paper_vertex_bilateral{window}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
