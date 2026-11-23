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
    ap.add_argument("--combine", action="store_true",
                    help="collapse all --regions into one region (SHAP summed over them per trial)")
    ap.add_argument("--parcel-mean", action="store_true",
                    help="per trial, average SHAP over the original HCP-MMP1 parcels the patient has (instead of the sum)")
    ap.add_argument("--merged", type=Path, default=REPO / "electrode coverages" / "figures" /
                    "merged_regions_min4_within_RegionsROIs_fallback.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    data = pd.read_pickle(args.pkl)
    lv = ["region", "band", "second"]
    subj = {}
    for c in args.contrasts:
        out = signed_shap(data, c)
        if args.parcel_mean:
            # per trial and band x second: mean over the parcels (with data) of each shown region
            parcel_of = pd.read_csv(args.merged, dtype=str).set_index("parcel")["region"].to_dict()
            groups = [" + ".join(args.regions)] if args.combine else args.regions
            frames, misses = [], []
            for g in groups:
                members = args.regions if args.combine else [g]
                parcels = [p for p, r in parcel_of.items() if r in members] or members
                cols = out["signed"].columns.get_level_values("region").isin(parcels)
                v = out["signed"].loc[:, cols].where(~out["missing"].loc[:, cols].to_numpy())
                m = v.T.groupby(level=["band", "second"]).mean().T          # NaN-skipping mean over parcels
                m.columns = pd.MultiIndex.from_tuples([(g, b, sec) for b, sec in m.columns], names=lv)
                frames.append(m)
                misses.append(m.isna())
            values, missing = pd.concat(frames, axis=1), pd.concat(misses, axis=1)
        else:
            values, missing, _ = merge_regions(out["signed"], out["missing"], args.merged)
        if args.combine and not args.parcel_mean:  # one region: summed over the listed regions per trial
            combined = " + ".join(args.regions)
            rename = lambda cols: pd.MultiIndex.from_arrays(
                [[combined if r in args.regions else r for r in cols.get_level_values(0)],
                 cols.get_level_values(1), cols.get_level_values(2)], names=cols.names)
            values = values.set_axis(rename(values.columns), axis=1)
            missing = missing.set_axis(rename(missing.columns), axis=1)
        s = subject_means(values, missing, lv, weights=CONTRASTS[c]["weights"])
        wanted = [" + ".join(args.regions)] if args.combine else args.regions
        unknown = [r for r in wanted if r not in s.columns.get_level_values("region")]
        if unknown:
            raise SystemExit(f"unknown region(s) {unknown}; e.g. {sorted(set(s.columns.get_level_values('region')))[:20]}")
        subj[c] = s.loc[:, s.columns.get_level_values("region").isin(wanted)]
    regions_shown = wanted

    vmax = max(np.nanmax(np.abs(s.mean().to_numpy(float))) for s in subj.values())
    norm = TwoSlopeNorm(0, -vmax, vmax)
    lo = min(np.nanmin(s.to_numpy(float)) for s in subj.values())
    hi = max(np.nanmax(s.to_numpy(float)) for s in subj.values())
    pad = 0.06 * (hi - lo)
    nr, nc = len(regions_shown), len(args.contrasts)
    height = 3.4 * nr + 1.8
    fig, axes = plt.subplots(nr, nc, figsize=(max(7.2 * nc, 9), height), facecolor="white", squeeze=False,
                             gridspec_kw=dict(hspace=0.55, wspace=0.12, top=1 - 0.75 / height,
                                              bottom=1.35 / height, left=0.08, right=0.98))
    rows = []
    for i, region in enumerate(regions_shown):
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
    fig.suptitle("Signed SHAP by band and second" + (", regions combined" if args.combine else ", selected regions")
                 + (", mean per parcel" if args.parcel_mean else ""),
                 fontsize=13, color=INK, x=0.02, ha="left", y=1 - 0.12 / height)
    import textwrap
    footer = (
             "Per trial the signed contrast (> 0 helps the correct call) of the region's features, "
             + ("SHAP averaged over the original HCP-MMP1 parcels the subject has (per band and second), so the value "
                "does not grow with the number of parcels covered; " if args.parcel_mean else
                "SHAP summed over the parcels of a merged region; ")
             + "averaged within subject with balanced trial-type weights\n"
             "(new vs repeated: new, recent, old 1/2, 1/4, 1/4; recent vs consolidated: 1/2, 1/2), then across the "
             f"subjects with the region. Bars = mean, dots = subjects. Regions: {args.merged.name}."
              + (" Combined: the listed regions taken together; a subject counts if it has any of them."
                 if args.combine else ""))
    footer = "\n".join(textwrap.fill(line, 150 if nc > 1 else 125) for line in footer.replace("\n", " ").split("\n"))
    fig.text(0.02, 0.45 / height, footer, fontsize=8, color=INK_MUTED, va="top")
    tag = ("combined_" if args.combine else "") + ("parcelmean_" if args.parcel_mean else "") + "_".join(re.sub(r"[^A-Za-z0-9]+", "", r) for r in args.regions)
    pd.concat(rows).to_csv(args.out_dir / f"timefreq_{tag}.csv", index=False)
    path = args.out_dir / f"timefreq_{tag}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    for ext in ("svg", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
