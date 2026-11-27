"""Flatmaps of the fMRI XGBoost model's signed SHAP per Glasser parcel (xgb_fmri.py output).

Same quantity as the sEEG maps (shap analysis/plot_new_vs_repeated.py): per
test-set trial the signed contrast of each parcel's SHAP value (> 0 helps the
correct call), averaged within participant with the balanced trial-type
weights, then across participants.
  new_vs_repeated         SHAP(new) - (SHAP(recent) + SHAP(old))/2, new 1/2, recent 1/4, old 1/4
  recent_vs_consolidated  SHAP(old) - SHAP(recent), recent 1/2, old 1/2
One panel per contrast, each with its own colour scale up to its largest
|value|; the --n-labels largest parcels are named.

Hemispheres: with xgb_fmri.py --hemispheres mean (the default) each feature is
the average of the left and right parcel, so one value per parcel is drawn on the
left-hemisphere flatmap. With --hemispheres separate (features L_<parcel> and
R_<parcel>) the left and right values are drawn in separate rows, both on the
left flatmap (fsaverage left/right parcels correspond one to one).

Written to --out-dir: fmri_shap_flatmap[_<tag>].{svg,pdf,png} and a csv per contrast.
"""

import argparse
import sys
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
from plot_new_vs_repeated import DIVERGING, INK, INK_MUTED, draw_region_map, load_flatmap, subject_means  # noqa: E402
from signed_shap import CONTRASTS, signed_shap  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def parcel_values(data, contrast):
    """participant x parcel signed SHAP (balanced weights); columns may carry L_/R_ prefixes."""
    out = signed_shap(data, contrast, "test")
    v, miss = out["signed"], out["missing"]
    return subject_means(v, miss, "region", weights=CONTRASTS[contrast]["weights"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, required=True, help="xgb_fmri.py output pickle")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--n-labels", type=int, default=6)
    ap.add_argument("--tag", default="", help="added to the output names")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_pickle(args.pkl)
    geo = load_flatmap(args.cache)
    contrasts = ["new_vs_repeated", "recent_vs_consolidated"]
    subj = {c: parcel_values(data, c) for c in contrasts}
    cols = subj[contrasts[0]].columns
    split = all(c[:2] in ("L_", "R_") for c in cols)
    rows = [("Left hemisphere", "L_"), ("Right hemisphere", "R_")] if split else [("Hemispheres pooled", "")]

    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 5.2 * len(rows)), facecolor="white", squeeze=False,
                             gridspec_kw=dict(wspace=0.05, hspace=0.25))
    for j, c in enumerate(contrasts):
        s = subj[c]
        means = s.mean()
        vmax = float(np.abs(means).max())
        norm = TwoSlopeNorm(0, -vmax, vmax)
        table = pd.DataFrame({"mean": means, "sem": s.sem(), "n_participants": s.notna().sum()})
        table.sort_values("mean", key=np.abs, ascending=False).to_csv(
            args.out_dir / f"fmri_shap_{c}{'_' + args.tag if args.tag else ''}.csv")
        for i, (row_name, prefix) in enumerate(rows):
            vals = means[means.index.str.startswith(prefix)] if prefix else means
            vals.index = vals.index.str[len(prefix):]
            ax = axes[i, j]
            draw_region_map(ax, geo, vals, norm, n_labels=args.n_labels, cmap=DIVERGING)
            ax.set_title(f"{CONTRASTS[c]['title']}" + (f" — {row_name.lower()}" if split else ""),
                         fontsize=10.5, color=INK, loc="left")
            cax = ax.inset_axes([0.55, -0.06, 0.4, 0.03])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax, orientation="horizontal")
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
            cb.set_label(f"Net signed SHAP (log-odds); > 0 helps the correct {CONTRASTS[c]['call']} call",
                         fontsize=7.5, color=INK)
        top = means.reindex(means.abs().sort_values(ascending=False).index).head(10)
        print(f"\n{c}: colour scale ±{vmax:.4f}; top 10:\n" + top.round(4).to_string())
    from matplotlib.collections import PolyCollection
    for coll in fig.findobj(PolyCollection):
        if len(coll.get_paths()) > 1000:
            coll.set_rasterized(True)
    n = len(subj[contrasts[0]])
    import textwrap
    fig.text(0.02, -0.02, textwrap.fill(f"NSD fMRI, {n} participants, test-set trials (correct responses). Per trial the signed "
             "SHAP contrast of each Glasser parcel's feature; averaged within participant with balanced trial-type "
             "weights, then across participants. "
             + ("Left and right parcels were separate features." if split else
                "Features were the average of the left and right parcel (xgb_fmri.py --hemispheres mean), so one "
                "value per parcel is shown on the left flatmap."), 190),
             fontsize=7.5, color=INK_MUTED, va="top")
    stem = args.out_dir / f"fmri_shap_flatmap{'_' + args.tag if args.tag else ''}"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300 if ext == "png" else 600, bbox_inches="tight")
    print("wrote", stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
