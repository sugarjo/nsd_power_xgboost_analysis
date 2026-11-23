"""Direction of the effect: feature (band power) difference by band and second for selected regions.

Companion of plot_region_timefreq.py (SHAP): the same regions, bands and
seconds, but the features themselves, so the bars show whether power is
higher or lower for one trial type than for the other.

Per trial (correct responses, image_category 1/2/3; all trials, or the test
set with --test-only): the feature of each band and second averaged over the
HCP-MMP1 parcels of the listed regions that the subject has (merged-region
names from --merged are expanded to their parcels; missing values skipped).
With --combine all listed regions form one area, otherwise one panel each.

Per subject the trial types are averaged separately and combined so each type
weighs the same whatever its trial count:
  new_vs_repeated         new - (recent + old) / 2     (> 0: higher for new images)
  recent_vs_consolidated  consolidated - recent        (> 0: higher for consolidated)
then the mean over the subjects who have the area (bars) and the subjects (dots).

--scale raw (default): the features as stored (here dB power relative to
baseline), so differences are in dB; --scale zscore: each feature z-scored
within subject first, so differences are in within-subject SDs.

Example:
  python3 plot_feature_timefreq.py --contrasts new_vs_repeated --combine \\
      --regions "POS1+v23ab+POS2" "31pd+31pv+31a+d23ab"

Written to --out-dir: features_<contrast>_[combined_]<regions>.{svg,pdf,png,csv}
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
from plot_new_vs_repeated import DIVERGING, INK, INK_MUTED, band_second_panel, subject_means  # noqa: E402
from signed_shap import CONTRASTS, feature_columns  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
DIFF = {  # trial-type weights giving the difference
    "new_vs_repeated": ({"new": 1.0, "recent": -0.5, "old": -0.5}, "new − seen (recent + old)/2", "higher for new"),
    "recent_vs_consolidated": ({"old": 1.0, "recent": -1.0}, "consolidated − recent", "higher for consolidated"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--regions", nargs="+", required=True, help="merged-region or parcel names")
    ap.add_argument("--combine", action="store_true", help="treat all --regions as one area")
    ap.add_argument("--contrasts", nargs="+", choices=list(DIFF), default=["new_vs_repeated"])
    ap.add_argument("--merged", type=Path, default=REPO / "electrode coverages" / "figures" /
                    "merged_regions_min4_staged.csv")
    ap.add_argument("--scale", choices=["raw", "zscore"], default="raw")
    ap.add_argument("--test-only", action="store_true", help="only the model's test trials")
    ap.add_argument("--color-max", type=float, help="fixed colour scale (+-); default: largest |bar|")
    ap.add_argument("--unit", default="dB", help="unit of the raw features, for the axis label")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    data = pd.read_pickle(args.pkl)
    X = data["features"]
    names = [c for c in X.columns if " | " in c]
    X = X[names]
    if args.test_only:
        X = X.loc[data["models"]["test_data"]]
    X = X[X.index.get_level_values("image_category").isin([1, 2, 3])]
    X.columns = feature_columns(names)
    if args.scale == "zscore":
        g = X.groupby(level="subject")
        X = (X - g.transform("mean")) / g.transform("std")

    parcel_of = pd.read_csv(args.merged, dtype=str).set_index("parcel")["region"].to_dict()
    def parcels(region):
        ps = [p for p, r in parcel_of.items() if r == region]
        return ps or [region]
    areas = {" + ".join(args.regions): sum((parcels(r) for r in args.regions), [])} if args.combine else \
        {r: parcels(r) for r in args.regions}
    have = set(X.columns.get_level_values("region"))
    for a, ps in areas.items():
        if not set(ps) & have:
            raise SystemExit(f"no features for {a!r} (parcels {ps})")

    lv = ["band", "second"]
    results, rows = {}, []
    for c in args.contrasts:
        for a, ps in areas.items():
            sel = X.columns.get_level_values("region").isin(ps)
            per_trial = X.loc[:, sel].T.groupby(level=lv).mean().T             # mean over parcels, NaN skipped
            s = subject_means(per_trial, per_trial.isna(), lv, weights=DIFF[c][0]).dropna(how="all")
            results[c, a] = s
            rows.append(s.stack(lv, future_stack=True).rename("value").reset_index().assign(contrast=c, area=a))

    vmax = args.color_max or max(np.nanmax(np.abs(s.mean().to_numpy(float))) for s in results.values())
    norm = TwoSlopeNorm(0, -vmax, vmax)
    unit = "within-subject SD" if args.scale == "zscore" else args.unit
    nr, nc = len(areas), len(args.contrasts)
    height = 3.4 * nr + 1.8
    fig, axes = plt.subplots(nr, nc, figsize=(max(7.2 * nc, 9), height), facecolor="white", squeeze=False,
                             gridspec_kw=dict(hspace=0.55, wspace=0.15, top=1 - 0.75 / height,
                                              bottom=1.4 / height, left=0.08, right=0.98))
    for i, a in enumerate(areas):
        for j, c in enumerate(args.contrasts):
            s = results[c, a]
            ax = axes[i, j]
            band_second_panel(ax, s, pd.DataFrame({"mean": s.mean()}), norm, DIVERGING,
                              f"Feature difference ({unit})", None, whiskers=False)
            lo, hi = min(np.nanmin(s.to_numpy(float)), 0), max(np.nanmax(s.to_numpy(float)), 0)
            pad = 0.06 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"{a}: {DIFF[c][1]}  ({len(s)} subjects)", fontsize=10.5, color=INK, loc="left")
    fig.suptitle("Feature difference by band and second" + (" (z-scored within subject)" if args.scale == "zscore"
                                                            else ""), fontsize=13, color=INK, x=0.02, ha="left",
                 y=1 - 0.12 / height)
    import textwrap
    footer = (f"Correct trials ({'test set' if args.test_only else 'all trials'}); per trial the feature averaged over "
              "the area's parcels the subject has (missing skipped); per subject the trial types averaged separately "
              "and combined so each weighs the same (" + "; ".join(f"{DIFF[c][1]}: > 0 {DIFF[c][2]}"
                                                               for c in args.contrasts) +
              "), then across the subjects with the area. Bars = mean, dots = subjects. "
              f"Colour scale ±{vmax:.3g} {unit}. Regions: {args.merged.name}.")
    fig.text(0.02, 0.5 / height, textwrap.fill(footer, 150 if nc > 1 else 125), fontsize=8, color=INK_MUTED,
             va="top")
    tag = "_".join(re.sub(r"[^A-Za-z0-9]+", "", r) for r in args.regions)
    stem = args.out_dir / (f"features_{'_'.join(args.contrasts)}_{'combined_' if args.combine else ''}{tag}"
                           + ("_z" if args.scale == "zscore" else ""))
    pd.concat(rows).to_csv(stem.with_suffix(".csv"), index=False)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300 if ext == "png" else 300, bbox_inches="tight")
    print("wrote", stem.with_suffix(".svg"))
    for (c, a), s in results.items():
        m = s.mean().round(3)
        print(f"{c} | {a}: {len(s)} subjects; largest |mean| {m.abs().max()} at {m.abs().idxmax()}")


if __name__ == "__main__":
    main()
