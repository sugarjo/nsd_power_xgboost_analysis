"""Signed SHAP per band and second for selected regions, for one or more contrasts.

Same quantity as plot_new_vs_repeated.py panel B, but for single (merged)
regions: per trial the signed contrast of the region's features (SHAP summed
over the parcels of a merged region), averaged within subject with the
contrast's trial-type weights, then across the subjects who have the region.
Bars = mean over subjects, dots = subjects; no statistics.

Example:
  python3 plot_region_timefreq.py --regions PHA3 "VMV1+FFC+VMV2+VMV3+VVC" \
      --merged "../electrode coverages/figures/merged_regions_min4_within_RegionsROIs_fallback.csv"

Written to --out-dir: timefreq_<regions>[_<merged table>].png and .csv
(subject-level values per contrast, region, band and second).
"""

import argparse
import re
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
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import DIVERGING, INK, INK_MUTED, band_second_panel, merge_regions, subject_means  # noqa: E402
from signed_shap import CONTRASTS, signed_shap  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--regions", nargs="+", required=True, help="region names as in the merged table / maps")
    ap.add_argument("--contrasts", nargs="+", choices=list(CONTRASTS), default=list(CONTRASTS))
    ap.add_argument("--merged", type=Path, default=REPO / "electrode coverages" / "figures" /
                    "merged_regions_min4_within_RegionsROIs_fallback.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    data = pd.read_pickle(args.pkl)
    lv = ["region", "band", "second"]
    subj = {}
    for c in args.contrasts:
        out = signed_shap(data, c)
        values, missing, _ = merge_regions(out["signed"], out["missing"], args.merged)
        s = subject_means(values, missing, lv, weights=CONTRASTS[c]["weights"])
        unknown = [r for r in args.regions if r not in s.columns.get_level_values("region")]
        if unknown:
            raise SystemExit(f"unknown region(s) {unknown}; e.g. {sorted(set(s.columns.get_level_values('region')))[:20]}")
        subj[c] = s.loc[:, s.columns.get_level_values("region").isin(args.regions)]

    vmax = max(np.nanmax(np.abs(s.mean().to_numpy(float))) for s in subj.values())
    norm = TwoSlopeNorm(0, -vmax, vmax)
    lo = min(np.nanmin(s.to_numpy(float)) for s in subj.values())
    hi = max(np.nanmax(s.to_numpy(float)) for s in subj.values())
    pad = 0.06 * (hi - lo)
    nr, nc = len(args.regions), len(args.contrasts)
    fig, axes = plt.subplots(nr, nc, figsize=(7.2 * nc, 3.4 * nr + 1.0), facecolor="white", squeeze=False,
                             gridspec_kw=dict(hspace=0.55, wspace=0.12))
    rows = []
    for i, region in enumerate(args.regions):
        for j, c in enumerate(args.contrasts):
            s = subj[c].xs(region, axis=1, level="region").dropna(how="all")
            stats = pd.DataFrame({"mean": s.mean()})
            ax = axes[i, j]
            band_second_panel(ax, s, stats, norm, DIVERGING, "Net signed SHAP (log-odds)" if j == 0 else "",
                              None, whiskers=False)
            ax.set_ylim(min(lo, 0) - pad, hi + pad)
            ax.set_title(f"{region}: {CONTRASTS[c]['title']}  ({len(s)} subjects)", fontsize=10.5, color=INK,
                         loc="left")
            rows.append(s.stack(level=["band", "second"], future_stack=True).rename("value").reset_index()
                        .assign(contrast=c, region=region))
    fig.suptitle("Signed SHAP by band and second, selected regions", fontsize=13, color=INK, x=0.07, ha="left")
    fig.text(0.07, 0.01,
             "Per trial the signed contrast (> 0 helps the correct call) of the region's features, SHAP summed over the "
             "parcels of a merged region; averaged within subject with balanced trial-type weights\n"
             "(new vs repeated: new, recent, old 1/2, 1/4, 1/4; recent vs consolidated: 1/2, 1/2), then across the "
             f"subjects with the region. Bars = mean, dots = subjects. Regions: {args.merged.name}.",
             fontsize=8, color=INK_MUTED, va="top")
    tag = "_".join(re.sub(r"[^A-Za-z0-9]+", "", r) for r in args.regions)
    pd.concat(rows).to_csv(args.out_dir / f"timefreq_{tag}.csv", index=False)
    path = args.out_dir / f"timefreq_{tag}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
