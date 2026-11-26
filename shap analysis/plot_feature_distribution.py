"""Distribution of one band's power (the feature itself, not SHAP) for a set of parcels, by trial type.

Per correct trial (image_category 1/2/3; all trials, or the test set with
--test-only): the band power of the --band in each --seconds window, averaged
over the listed HCP-MMP1 parcels the patient has a channel in (missing skipped).
One panel per second; per trial type:
  violin = all trials pooled over patients,
  bar    = mean over patients of each patient's mean (each patient counts once),
  error  = s.e.m. over patients, dots = patients, thin lines join a patient's dots.
Values are as stored in the features (dB relative to the session baseline).

Example:
  python3 plot_feature_distribution.py --parcels 31pd 31pv 31a d23ab --band GammaH

Written to --out-dir: feature_distribution_<band>_<parcels>.{svg,pdf,png,csv}
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import AXIS, BAND_LABELS, GRID, INK, INK_MUTED  # noqa: E402
from signed_shap import feature_columns  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
TYPES = {1: ("New", "#7a7a7a"), 2: ("Recent", "#2a78d6"), 3: ("Consolidated", "#eb6834")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--parcels", nargs="+", required=True)
    ap.add_argument("--band", default="GammaH", help="Delta Theta Alpha Beta GammaL GammaH Ripple")
    ap.add_argument("--seconds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--test-only", action="store_true", help="only the model's test trials")
    ap.add_argument("--unit", default="dB")
    ap.add_argument("--clip", type=float, default=99.5, help="y range: this upper (and 100 - it lower) percentile")
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
    unknown = set(args.parcels) - set(X.columns.get_level_values("region"))
    if unknown:
        raise SystemExit(f"unknown parcels: {sorted(unknown)}")

    rows = []
    for sec in args.seconds:
        sel = (X.columns.get_level_values("region").isin(args.parcels)
               & (X.columns.get_level_values("band") == args.band)
               & (X.columns.get_level_values("second") == sec))
        v = X.loc[:, sel].mean(axis=1, skipna=True).dropna()          # mean over the patient's parcels
        rows.append(pd.DataFrame({"subject": v.index.get_level_values("subject"),
                                  "category": v.index.get_level_values("image_category").astype(int),
                                  "second": sec, "value": v.to_numpy()}))
    long = pd.concat(rows, ignore_index=True)
    tag = "_".join(re.sub(r"[^A-Za-z0-9]+", "", p) for p in args.parcels)
    stem = args.out_dir / f"feature_distribution_{args.band}_{tag}{'_test' if args.test_only else ''}"
    long.to_csv(stem.with_suffix(".csv"), index=False)

    per = long.groupby(["second", "subject", "category"]).value.mean().unstack("category")
    lo, hi = np.nanpercentile(long.value, [100 - args.clip, args.clip])
    lo, hi = min(lo, 0), max(hi, 0)
    pad = 0.04 * (hi - lo)
    lim = max(abs(lo), abs(hi))
    fig, axes = plt.subplots(1, len(args.seconds), figsize=(3.4 * len(args.seconds) + 0.4, 3.6),
                             facecolor="white", sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, sec in zip(axes[0], args.seconds):
        p = per.loc[sec]
        jit = rng.uniform(-0.1, 0.1, len(p))
        for i, (cat, (lab, col)) in enumerate(TYPES.items()):
            vals = long[(long.second == sec) & (long.category == cat)].value.to_numpy()
            vals = vals[(vals >= lo - 3 * pad) & (vals <= hi + 3 * pad)]
            parts = ax.violinplot(vals, positions=[i], widths=0.8, showextrema=False, points=200)
            for b in parts["bodies"]:
                b.set_facecolor(col); b.set_edgecolor("none"); b.set_alpha(0.18)
            ax.bar(i, p[cat].mean(), width=0.42, color=col, alpha=0.85, zorder=3)
            ax.errorbar(i, p[cat].mean(), yerr=p[cat].sem(), fmt="none", ecolor=INK, elinewidth=1.1, capsize=3,
                        zorder=5)
        for j, (_, r) in enumerate(p.iterrows()):
            ax.plot(np.arange(3) + jit[j], r[[1, 2, 3]].to_numpy(), color=INK_MUTED, lw=0.5, alpha=0.6, zorder=4)
        for i, cat in enumerate(TYPES):
            ax.scatter(i + jit, p[cat], s=12, color="white", edgecolor=INK, linewidths=0.7, zorder=6)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks(range(3), [TYPES[c][0] for c in TYPES], fontsize=8.5, color=INK)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{BAND_LABELS.get(args.band, args.band)}, {sec}{'st' if sec == 1 else 'nd'} second",
                     fontsize=9.5, color=INK, loc="left")
        ax.tick_params(axis="y", labelsize=7.5, colors=INK_MUTED)
        ax.tick_params(axis="x", length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0, 0].set_ylabel(f"{BAND_LABELS.get(args.band, args.band)} power ({args.unit})", fontsize=8.5, color=INK)
    n_subj = long.subject.nunique()
    counts = long[long.second == args.seconds[0]].category.value_counts()
    fig.suptitle(f"{' + '.join(args.parcels)}: correct trials ({n_subj} patients)", fontsize=10.5, color=INK,
                 x=0.02, ha="left", y=1.03)
    fig.text(0.02, -0.04,
             f"Per trial the feature averaged over the listed parcels the patient has. Violins: all "
             f"{'test-set ' if args.test_only else ''}correct trials pooled (n = "
             + ", ".join(f"{TYPES[c][0].lower()} {counts.get(c, 0)}" for c in TYPES)
             + f"; y range {lo:.1f} to {hi:.1f} {args.unit}, the {100 - args.clip:g}–{args.clip:g}th percentiles).\nBars: mean over patients of each patient's mean ± s.e.m.; "
               "dots = patients, lines join each patient's values.",
             fontsize=6.5, color=INK_MUTED, va="top")
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print("wrote", stem.with_suffix(".svg"))
    print(per.groupby(level="second").agg(["mean", "sem"]).round(3).to_string())


if __name__ == "__main__":
    main()
