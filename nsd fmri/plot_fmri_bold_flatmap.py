"""Flatmaps of the BOLD differences between new, recent and consolidated images per Glasser parcel.

Uses the features of the fMRI XGBoost model (xgb_fmri.py output): per trial the
single-trial beta averaged over each parcel's vertices, z-scored per vertex
within session. Correct trials only (all of them, or the test set with
--test-only). Per participant the trial types are averaged separately, the
pairwise differences taken, then averaged across participants:
  new − recent, new − consolidated, recent − consolidated
(> 0: higher BOLD for the first type). All panels share one colour scale (the
largest |difference| over the three maps); the --n-labels largest parcels of each
map are named. With --hemispheres mean (default in xgb_fmri.py) each feature is
the average of the left and right parcel, drawn on the left flatmap.

Written to --out-dir: fmri_bold_flatmap.{svg,pdf,png} and fmri_bold_differences.csv
(participant-level differences per parcel and contrast).
"""

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shap analysis"))
from plot_new_vs_repeated import DIVERGING, INK, INK_MUTED, draw_region_map, load_flatmap  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
TYPES = {1: "new", 2: "recent", 3: "consolidated"}
PAIRS = [("new", "recent"), ("new", "consolidated"), ("recent", "consolidated")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, required=True, help="xgb_fmri.py output pickle")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--n-labels", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_pickle(args.pkl)
    X = data["features"]
    X = X[[c for c in X.columns if " | " in c]]
    if args.test_only:
        X = X.loc[data["models"]["test_data"]]
    cat = X.index.get_level_values("image_category")
    X = X[np.isin(cat, [1, 2, 3])]
    X.columns = [c.split(" | ")[0] for c in X.columns]
    means = X.groupby([X.index.get_level_values("subject"),
                       X.index.get_level_values("image_category").map(TYPES)]).mean()
    diffs = {f"{a} − {b}": means.xs(a, level=1) - means.xs(b, level=1) for a, b in PAIRS}
    pd.concat(diffs, names=["contrast", "subject"]).to_csv(args.out_dir / "fmri_bold_differences.csv")

    vmax = max(float(d.mean().abs().max()) for d in diffs.values())
    norm = TwoSlopeNorm(0, -vmax, vmax)
    geo = load_flatmap(args.cache)
    split = all(c[:2] in ("L_", "R_") for c in X.columns)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), facecolor="white", gridspec_kw=dict(wspace=0.04))
    for ax, (name, d) in zip(axes, diffs.items()):
        m = d.mean()
        if split:                                   # left-hemisphere parcels on the flatmap
            m = m[m.index.str.startswith("L_")]
            m.index = m.index.str[2:]
        draw_region_map(ax, geo, m, norm, n_labels=args.n_labels, cmap=DIVERGING)
        ax.set_title(name, fontsize=11, color=INK, loc="left")
        top = d.mean().reindex(d.mean().abs().sort_values(ascending=False).index).head(8)
        print(f"\n{name}: largest |difference| (z):\n" + top.round(3).to_string())
    cax = axes[2].inset_axes([0.45, -0.06, 0.5, 0.03])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
    cb.set_label("BOLD difference (z, within-session z-scored betas); > 0: higher for the first type",
                 fontsize=7.5, color=INK)
    from matplotlib.collections import PolyCollection
    for coll in fig.findobj(PolyCollection):
        if len(coll.get_paths()) > 1000:
            coll.set_rasterized(True)
    n = len(next(iter(diffs.values())))
    counts = pd.Series(X.index.get_level_values("image_category")).map(TYPES).value_counts()
    fig.text(0.02, -0.02, textwrap.fill(
        f"NSD fMRI, {n} participants, correct trials ({'test set' if args.test_only else 'all'}; "
        + ", ".join(f"{k} {counts.get(k, 0)}" for k in TYPES.values())
        + "). Per trial the beta averaged over each Glasser parcel's vertices (z-scored per vertex within session); "
          "per participant each trial type averaged, the difference taken, then averaged across participants. "
          f"Shared colour scale ±{vmax:.3f}; the {args.n_labels} largest parcels per map are named. "
        + ("Left-hemisphere parcels shown." if split else
           "Features were the average of the left and right parcel, shown on the left flatmap."), 230),
        fontsize=7.5, color=INK_MUTED, va="top")
    stem = args.out_dir / f"fmri_bold_flatmap{'_test' if args.test_only else ''}"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300 if ext == "png" else 600, bbox_inches="tight")
    print("wrote", stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
