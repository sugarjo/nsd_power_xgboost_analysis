"""Flatmaps of the band-power difference between new and repeated trials.

Uses the feature values themselves (no SHAP): for every region x band x
second, is power higher on new than on repeated trials?

Per feature:
  1. With --scale power (default) power is converted to dB (10 * log10) and
     non-positive values are dropped. With --scale log the features are taken
     as they are (already dB / log / z-scored), and the difference below is in
     their own units.
  2. Per subject, the dB values are averaged over new, recent and old trials
     separately (a label needs at least --min-trials trials, otherwise the
     subject is left out for that feature).
  3. Per subject: difference = new - (recent + old) / 2, so recent and old
     weigh equally whatever their trial counts. In dB this is the log ratio
     of new to repeated power; > 0 means more power on new trials.
  4. Subjects are pooled with equal weight: the mean over subjects, a 95%
     bootstrap CI and an exact two-sided sign-flip p-value over subjects
     (uncorrected for the number of regions, bands and seconds tested).

Trials: correct responses only (image_category 1, 2, 3), from all trials by
default or only the model's test trials with --test-only.

Written to --out-dir:
  power_new_vs_repeated_<band><second>_....png
  power_new_vs_repeated.csv   per feature: mean, CI, n_subjects, p_signflip
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from plot_new_vs_repeated import load_flatmap, pool
from plot_region_band_second import BAND_NAMES, ORDINAL, map_grid, panel
from signed_shap import feature_columns

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LABELS = {1: "new", 2: "recent", 3: "old"}     # image_category of correct trials


def subject_differences(data: dict, min_trials: int, test_only: bool, scale: str) -> pd.DataFrame:
    """Subject x feature table of new - (recent + old)/2 (steps 1-3)."""
    X = data["features"]
    X = X[[c for c in X.columns if " | " in c]]
    if test_only:
        X = X.loc[data["models"]["test_data"]]
    cat = X.index.get_level_values("image_category")
    X = X[cat.isin(list(LABELS))]
    X.columns = feature_columns(X.columns)

    if scale == "power":
        nonpos = (X <= 0).to_numpy().sum()
        if nonpos:
            print(f"dropping {nonpos} non-positive power values before taking dB")
        db = 10 * np.log10(X.where(X > 0))
    else:
        db = X

    keys = [X.index.get_level_values("subject"),
            X.index.get_level_values("image_category").map(LABELS)]
    means = db.groupby(keys).mean()
    counts = db.groupby(keys).count()
    means = means.where(counts >= min_trials)
    means.index.names = ["subject", "label"]
    by_label = {lab: means.xs(lab, level="label") for lab in LABELS.values()}
    return by_label["new"] - (by_label["recent"] + by_label["old"]) / 2


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
    ap.add_argument("--panels", type=panel, nargs="+",
                    default=[("Delta", 1), ("Delta", 2), ("Alpha", 1), ("Alpha", 2)],
                    help="band:second pairs, e.g. Delta:1 Alpha:2")
    ap.add_argument("--min-subjects", type=int, default=3)
    ap.add_argument("--min-trials", type=int, default=5, help="per subject and label (new/recent/old)")
    ap.add_argument("--test-only", action="store_true", help="only the model's test trials")
    ap.add_argument("--scale", choices=["power", "log"], default="power",
                    help="power: linear power, converted to dB; log: already dB/log/z-scored, used as is")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    subj = subject_differences(pd.read_pickle(args.pkl), args.min_trials, args.test_only, args.scale)
    unit = "dB" if args.scale == "power" else "feature units"
    stats = pool(subj, args.n_boot)
    stats.index = pd.MultiIndex.from_tuples(stats.index, names=["region", "band", "second"])
    stats["p_signflip"] = sign_flip_p(subj).to_numpy()
    stats.to_csv(args.out_dir / "power_new_vs_repeated.csv")

    shown = stats[stats["n_subjects"] >= args.min_subjects]
    values = [shown.xs(p, level=["band", "second"])["mean"] for p in args.panels]
    vmax = np.percentile(np.abs(np.concatenate([v.to_numpy() for v in values])), 98)
    norm = TwoSlopeNorm(0, -vmax, vmax)

    trials = "the model's test trials" if args.test_only else "all trials"
    titles = [f"{BAND_NAMES.get(b, b)}, {ORDINAL.get(s, s)} second" for b, s in args.panels]
    fig = map_grid(
        load_flatmap(args.cache), values, titles, norm,
        f"Power difference, new − repeated ({unit}); > 0: more power on new trials",
        args.min_subjects, "Band power on new vs repeated trials",
        f"Correct responses from {trials}; power in {unit}, averaged within subject per label "
        f"(≥{args.min_trials} trials each), difference = new − (recent + old)/2, then averaged across "
        f"{subj.index.nunique()} subjects.\n"
        f"Regions with ≥{args.min_subjects} subjects; top 5 per panel by |value| named. One colour scale for "
        f"all panels, clipped at ±{vmax:.3g} {unit} (98th percentile of |value| over the regions shown). "
        f"Flatmap of the left hemisphere, hemispheres pooled.")
    name = "_".join(f"{b.lower()}{s}" for b, s in args.panels)
    path = args.out_dir / f"power_new_vs_repeated_{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
