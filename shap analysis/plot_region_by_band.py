"""Per-band flatmaps of each region's contribution to the new vs repeated decision.

Same aggregation as plot_new_vs_repeated.py, but per region x frequency band:
NaN features are left out, the signed mean-contrast SHAP values of a region's
band (both seconds) are summed per trial, averaged within subject (new and
repeated trials weighted equally) and then across subjects.

One flatmap per band, all seven on one colour scale: symmetric, clipped at the
98th percentile of |value| over every region x band shown. Regions covered by
fewer than --min-subjects subjects are dark gray. Amygdala and hippocampus
are drawn in the ventral glass-brain inset of each map.

Written to --out-dir:
  new_vs_repeated_region_by_band.png
  new_vs_repeated_region_by_band.csv   mean, CI and number of subjects
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from plot_new_vs_repeated import (AXIS, BANDS, DIVERGING, INK, INK_MUTED, NO_DATA_GYRUS,
                                  draw_region_map, load_flatmap, pool, subject_means)
from signed_shap import signed_shap

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BAND_NAMES = {"GammaL": "Low gamma", "GammaH": "High gamma"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--min-subjects", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out = signed_shap(pd.read_pickle(args.pkl))
    subj = subject_means(out["signed"], out["missing"], ["region", "band"])
    stats = pool(subj, args.n_boot)
    stats.index.names = ["region", "band"]
    stats.to_csv(args.out_dir / "new_vs_repeated_region_by_band.csv")

    shown = stats[stats["n_subjects"] >= args.min_subjects]
    vmax = np.percentile(np.abs(shown["mean"]), 98)
    norm = TwoSlopeNorm(0, -vmax, vmax)
    geo = load_flatmap(args.cache)

    fig, axes = plt.subplots(4, 2, figsize=(12, 15), facecolor="white",
                             gridspec_kw=dict(hspace=0.12, wspace=0.04, left=0.02, right=0.98,
                                              top=0.95, bottom=0.04))
    for i, (ax, band) in enumerate(zip(axes.ravel(), BANDS)):
        values = shown.xs(band, level="band")["mean"]
        draw_region_map(ax, geo, values, norm, n_labels=3, inset_labels=i == 0)
        ax.set_title(f"{chr(65 + i)}   {BAND_NAMES.get(band, band)}", fontsize=11, color=INK, loc="left")

    # Last cell: shared colour bar and the no-data swatch.
    key = axes.ravel()[-1]
    key.axis("off")
    cax = key.inset_axes([0.1, 0.55, 0.8, 0.07])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax,
                      orientation="horizontal", extend="both")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    cb.set_label("Net signed SHAP per region and band (log-odds)\n> 0 helps the correct new/repeated call",
                 fontsize=9, color=INK)
    key.add_patch(plt.Rectangle((0.1, 0.25), 0.04, 0.06, facecolor=NO_DATA_GYRUS, edgecolor=AXIS,
                                lw=0.5, transform=key.transAxes))
    key.text(0.16, 0.28, f"Fewer than {args.min_subjects} subjects", fontsize=9, color=INK,
             va="center", transform=key.transAxes)

    n_subj = subj.index.nunique()
    fig.suptitle("Contribution of each region to the new vs repeated decision, per frequency band",
                 fontsize=14, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.005,
             f"Mean-contrast SHAP, NaN features left out; per trial summed over both seconds of a region's band, "
             f"averaged within subject (new and repeated trials weighted equally), then across {n_subj} subjects.\n"
             f"Regions with ≥{args.min_subjects} subjects; top 3 per band by |value| named. One colour scale for all "
             f"panels, clipped at ±{vmax:.3f} (98th percentile of |value| over all regions and bands). "
             f"Flatmap of the left hemisphere, hemispheres pooled.",
             fontsize=8, color=INK_MUTED)
    fig.savefig(args.out_dir / "new_vs_repeated_region_by_band.png", dpi=200, bbox_inches="tight")
    print("wrote", args.out_dir / "new_vs_repeated_region_by_band.png")


if __name__ == "__main__":
    main()
