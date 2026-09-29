"""Flatmaps of each region's contribution for chosen band x second combinations.

Same aggregation as plot_new_vs_repeated.py, per region x band x second: NaN
features are left out, the signed mean-contrast SHAP value of the region's
feature is averaged within subject (new and repeated trials weighted equally)
and then across subjects.

One flatmap per --panels entry (default: delta and alpha, each in the first and
in the second second; up to three panels per row, four as 2 x 2), all on one colour scale: symmetric, clipped at the
98th percentile of |value| over the regions shown in any panel. Regions
covered by fewer than --min-subjects subjects are dark gray. Amygdala and
hippocampus are drawn in the ventral glass-brain inset of each map.

Written to --out-dir:
  new_vs_repeated_<band><second>_....png
  new_vs_repeated_region_band_second.csv   mean, CI and number of subjects
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from plot_new_vs_repeated import (AXIS, DIVERGING, INK, INK_MUTED, NO_DATA_GYRUS,
                                  draw_region_map, load_flatmap, pool, subject_means)
from signed_shap import signed_shap

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BAND_NAMES = {"GammaL": "Low gamma", "GammaH": "High gamma"}
ORDINAL = {1: "first", 2: "second"}


def panel(text: str) -> tuple[str, int]:
    """'Delta:1' -> ('Delta', 1)."""
    band, second = text.split(":")
    return band, int(second)


def map_grid(geo, panel_values, titles, norm, cbar_label, min_subjects, title, footer):
    """Figure with one region flatmap per panel on a shared colour scale.

    Up to three panels per row (four as 2 x 2); colour bar and the no-data
    swatch under the maps, footer text below them."""
    n = len(panel_values)
    ncols = 2 if n == 4 else min(n, 3)
    nrows = -(-n // ncols)
    height = 4.4 * nrows + 1.0            # 1 inch below the maps for colour bar and key
    bottom = 1.0 / height
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, height), facecolor="white", squeeze=False,
                             gridspec_kw=dict(wspace=0.04, hspace=0.12, left=0.02, right=0.98,
                                              top=1 - 0.8 / height, bottom=bottom))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for i, (ax, values, text) in enumerate(zip(axes.ravel(), panel_values, titles)):
        draw_region_map(ax, geo, values, norm, n_labels=5, inset_labels=i == 0)
        ax.set_title(f"{chr(65 + i)}   {text}", fontsize=11, color=INK, loc="left")

    cax = fig.add_axes([0.55, 0.55 * bottom, 0.3, 0.15 * bottom])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax,
                      orientation="horizontal", extend="both")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    cb.set_label(cbar_label, fontsize=9, color=INK)
    key = fig.add_axes([0.2, 0.55 * bottom, 0.25, 0.15 * bottom])
    key.set_xlim(0, 1); key.set_ylim(0, 1); key.axis("off")
    key.add_patch(plt.Rectangle((0, 0), 0.06, 1, facecolor=NO_DATA_GYRUS, edgecolor=AXIS, lw=0.5))
    key.text(0.09, 0.5, f"Fewer than {min_subjects} subjects", fontsize=9, color=INK, va="center")

    fig.suptitle(title, fontsize=14, color=INK, x=0.02, y=1 - 0.15 / height, ha="left", va="top")
    fig.text(0.02, -0.25 * bottom, footer, fontsize=8, color=INK_MUTED)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "real_data_bipolar_xgboost_3w_random_data.pkl")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--panels", type=panel, nargs="+", default=[("Delta", 1), ("Delta", 2), ("Alpha", 1), ("Alpha", 2)],
                    help="band:second pairs, e.g. Delta:1 Alpha:2")
    ap.add_argument("--min-subjects", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    lv = ["region", "band", "second"]
    out = signed_shap(pd.read_pickle(args.pkl))
    subj = subject_means(out["signed"], out["missing"], lv)
    stats = pool(subj, args.n_boot)
    stats.index.names = lv
    stats.to_csv(args.out_dir / "new_vs_repeated_region_band_second.csv")

    shown = stats[stats["n_subjects"] >= args.min_subjects]
    values = {p: shown.xs(p, level=["band", "second"])["mean"] for p in args.panels}
    vmax = np.percentile(np.abs(np.concatenate([v.to_numpy() for v in values.values()])), 98)
    norm = TwoSlopeNorm(0, -vmax, vmax)
    geo = load_flatmap(args.cache)

    titles = [f"{BAND_NAMES.get(b, b)}, {ORDINAL.get(sec, sec)} second" for b, sec in args.panels]
    fig = map_grid(
        geo, [values[p] for p in args.panels], titles, norm,
        "Net signed SHAP per region (log-odds); > 0 helps the correct new/repeated call",
        args.min_subjects, "Region contributions to the new vs repeated decision",
        f"Mean-contrast SHAP of one feature per region (band x second), NaN left out; averaged within "
        f"subject (new and repeated trials weighted equally), then across {subj.index.nunique()} subjects.\n"
        f"Regions with ≥{args.min_subjects} subjects; top 5 per panel by |value| named. One colour scale for "
        f"all panels, clipped at ±{vmax:.3f} (98th percentile of |value| over the regions shown). "
        f"Flatmap of the left hemisphere, hemispheres pooled.")
    name = "_".join(f"{b.lower()}{s}" for b, s in args.panels)
    path = args.out_dir / f"new_vs_repeated_{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
