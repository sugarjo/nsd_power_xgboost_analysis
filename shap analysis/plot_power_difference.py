"""Flatmaps of the feature (band power) difference between trial types.

Uses the feature values themselves (no SHAP): for every region x band x
second, is the feature higher on one trial type than on the other?

  --contrast new_vs_repeated (default)  new - (recent + old)/2
  --contrast recent_vs_consolidated     consolidated (old) - recent

Per feature:
  1. Scale (--scale):
       power   linear power, converted to dB (10 * log10); non-positive values
               are dropped. The difference is then a log power ratio in dB.
       log     already dB / log / z-scored; used as is (own units).
       zscore  z-scored within subject (over all of that subject's correct
               trials), so the difference is in within-subject SDs and every
               subject is on the same scale.
  2. With --merged, low-coverage parcels are merged into regions
     (merge_low_coverage.py table): per trial the scaled feature is averaged
     over the region's parcels that have data.
  3. Per subject, the values are averaged over new, recent and old trials
     separately; a trial type needs at least --min-trials trials, otherwise
     the subject is left out for that feature.
  4. Per subject the contrast is taken from these means, so each trial type
     weighs the same whatever its trial count.
  5. Subjects are pooled with equal weight: the mean over subjects, a 95%
     bootstrap CI and an exact two-sided sign-flip p-value over subjects
     (uncorrected for the number of regions, bands and seconds).

Trials: correct responses only (image_category 1, 2, 3), from all trials by
default or only the model's test trials with --test-only.

Written to --out-dir (prefix power_<contrast>, suffix "_merged" with --merged):
  <prefix><suffix>_<band><second>_....png   one flatmap per --panels entry
  <prefix><suffix>.csv                      per feature: mean, CI, n_subjects,
                                            p_signflip
  <prefix><suffix>_label_means.csv          per feature: mean over subjects of
                                            the new, recent and old values

Example (real data, power in linear units, alpha in both seconds, merged):
  python plot_power_difference.py --pkl real.pkl --contrast recent_vs_consolidated \\
      --panels Alpha:1 Alpha:2 --merged --n-labels 6
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_new_vs_repeated import load_flatmap, pool
from plot_region_band_second import BAND_NAMES, ORDINAL, map_grid, panel
from signed_shap import feature_columns

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LABELS = {1: "new", 2: "recent", 3: "old"}     # image_category of correct trials
CONTRASTS = {
    "new_vs_repeated": dict(weights={"new": 1, "recent": -0.5, "old": -0.5}, short="new − repeated",
                            title="new vs repeated", formula="new − (recent + old)/2", more="new"),
    "recent_vs_consolidated": dict(weights={"old": 1, "recent": -1}, short="consolidated − recent",
                                   title="recent vs consolidated (old)", formula="consolidated − recent",
                                   more="consolidated"),
}
UNITS = {"power": "dB", "log": "feature units", "zscore": "within-subject SD"}


def scaled_features(data: dict, test_only: bool, scale: str) -> pd.DataFrame:
    """Correct trials x (region, band, second) features on the chosen scale (step 1)."""
    X = data["features"]
    X = X[[c for c in X.columns if " | " in c]]
    if test_only:
        X = X.loc[data["models"]["test_data"]]
    X = X[X.index.get_level_values("image_category").isin(list(LABELS))]
    X.columns = feature_columns(X.columns)
    if scale == "power":
        nonpos = (X <= 0).to_numpy().sum()
        if nonpos:
            print(f"dropping {nonpos} non-positive power values before taking dB")
        return 10 * np.log10(X.where(X > 0))
    if scale == "zscore":
        g = X.groupby(level="subject")
        return (X - g.transform("mean")) / g.transform("std")
    return X


def merge_features(X: pd.DataFrame, table: Path):
    """Average each trial's features over the parcels of a merged region (step 2)."""
    regions = pd.read_csv(table, dtype=str).set_index("parcel")["region"].to_dict()
    cols = X.columns
    keys = [[regions.get(r, r) for r in cols.get_level_values("region")],
            cols.get_level_values("band"), cols.get_level_values("second")]
    merged = X.T.groupby(keys).mean().T                # NaN (no electrode) is skipped
    merged.columns.names = cols.names
    return merged, regions


def label_means(X: pd.DataFrame, min_trials: int) -> dict:
    """Trial type -> subject x feature means, NaN below --min-trials (step 3)."""
    keys = [X.index.get_level_values("subject"), X.index.get_level_values("image_category").map(LABELS)]
    means = X.groupby(keys).mean().where(X.groupby(keys).count() >= min_trials)
    means.index.names = ["subject", "label"]
    return {lab: means.xs(lab, level="label") for lab in LABELS.values()}


def sign_flip_p(subj: pd.DataFrame, n_perm: int = 100000, seed: int = 0) -> pd.Series:
    """Exact (or, above 16 subjects, Monte-Carlo) two-sided sign-flip p per column."""
    rng = np.random.default_rng(seed)
    p = pd.Series(np.nan, index=subj.columns)
    n_valid = subj.notna().sum()
    for n, cols in n_valid.groupby(n_valid).groups.items():
        if n < 2:
            continue
        signs = (np.array(list(itertools.product([1, -1], repeat=n))) if n <= 16
                 else rng.choice([1, -1], (n_perm, n)))
        for col in cols:
            v = subj[col].dropna().to_numpy()
            observed = abs(v.mean())
            null = np.abs(signs @ v / n)
            p[col] = (null >= observed - 1e-12).mean()
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--contrast", choices=list(CONTRASTS), default="new_vs_repeated")
    ap.add_argument("--panels", type=panel, nargs="+",
                    default=[("Delta", 1), ("Delta", 2), ("Alpha", 1), ("Alpha", 2)],
                    help="band:second pairs, e.g. Delta:1 Alpha:2")
    ap.add_argument("--scale", choices=list(UNITS), default="power",
                    help="power: linear power, converted to dB; log: used as is; zscore: z-scored within subject")
    ap.add_argument("--merged", type=Path, nargs="?", const=REPO / "electrode coverages" / "figures" / "merged_regions.csv",
                    help="parcel -> merged region table (merge_low_coverage.py); features averaged within merged regions")
    ap.add_argument("--min-subjects", type=int, default=3)
    ap.add_argument("--min-trials", type=int, default=5, help="per subject and trial type (new/recent/old)")
    ap.add_argument("--test-only", action="store_true", help="only the model's test trials")
    ap.add_argument("--n-labels", type=int, default=5, help="regions named per panel (largest |value|)")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    C = CONTRASTS[args.contrast]
    unit = UNITS[args.scale]
    prefix = f"power_{args.contrast}" + ("_merged" if args.merged else "")

    X = scaled_features(pd.read_pickle(args.pkl), args.test_only, args.scale)
    regions = None
    if args.merged:
        X, regions = merge_features(X, args.merged)
    by_label = label_means(X, args.min_trials)
    subj = sum(w * by_label[lab] for lab, w in C["weights"].items())     # step 4

    stats = pool(subj, args.n_boot)
    stats.index = pd.MultiIndex.from_tuples(stats.index, names=["region", "band", "second"])
    stats["p_signflip"] = sign_flip_p(subj).to_numpy()
    stats.to_csv(args.out_dir / f"{prefix}.csv")
    level = pd.DataFrame({lab: m.mean() for lab, m in by_label.items()})
    level["n_subjects"] = subj.notna().sum()
    level.index.names = ["region", "band", "second"]
    level.to_csv(args.out_dir / f"{prefix}_label_means.csv")

    shown = stats[stats["n_subjects"] >= args.min_subjects]
    values = [shown.xs(p, level=["band", "second"])["mean"] for p in args.panels]
    vmax = np.percentile(np.abs(np.concatenate([v.to_numpy() for v in values])), 98)
    norm = TwoSlopeNorm(0, -vmax, vmax)

    trials = "the model's test trials" if args.test_only else "all trials"
    scale_text = {"power": "power in dB", "log": "features as given",
                  "zscore": "features z-scored within subject"}[args.scale]
    titles = [f"{BAND_NAMES.get(b, b)}, {ORDINAL.get(s, s)} second" for b, s in args.panels]
    fig = map_grid(
        load_flatmap(args.cache), values, titles, norm,
        f"Difference, {C['short']} ({unit}); > 0: higher on {C['more']} trials",
        args.min_subjects, f"Band power on {C['title']} trials",
        f"Correct responses from {trials}; {scale_text}, averaged within subject per trial type "
        f"(≥{args.min_trials} trials each),\ndifference = {C['formula']}, then averaged across "
        f"{subj.index.nunique()} subjects. "
        + (f"Parcels with <{args.min_subjects} subjects merged with neighbours ({args.merged.name}), features "
           f"averaged within each merged region." if args.merged else "") +
        f"\nRegions with ≥{args.min_subjects} subjects; top {args.n_labels} per panel by |value| named. "
        f"One colour scale for all panels,\nclipped at ±{vmax:.2g} {unit} (98th percentile of |value| over the "
        f"regions shown). Flatmap of the left hemisphere, hemispheres pooled.",
        regions, args.n_labels)
    name = "_".join(f"{b.lower()}{s}" for b, s in args.panels)
    path = args.out_dir / f"{prefix}_{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
