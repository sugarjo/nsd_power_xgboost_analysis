"""sEEG counterpart of nsd fmri/nsd_prevalence.py: within-patient tests and Bayesian population prevalence.

Effects (correct trials; image_category 1 = new, 2 = recent, 3 = consolidated/old):
  recent > new, consolidated > new, consolidated > recent, and the conjunction
  'consolidated beyond both' (consolidated > new AND > recent, or < both).
Units: merged regions (merge_low_coverage.py table) x band x second.

Two kinds of data, each tested within every patient:
  features  power features (data["features"]), averaged over the parcels of a
            merged region; Welch t between the two trial types, all trials of
            the patient (train + test).
  shap      SHAP values of the test set (data["models"]["test_shap_values"],
            classes new/recent/old), summed over the parcels of a merged region
            (SHAP is additive). For a pair (A vs B) the per-trial value is
            SHAP(class A) - SHAP(class B), and the test is whether it is higher
            on A trials than on B trials (Welch t): the feature pushes the
            model towards the correct one of the two classes. Only the positive
            direction is meaningful here; the conjunction is 'consolidated is
            separated from both new and recent'.
Values of features that were missing (no electrode, missing second) are left out.

Within a patient, family-wise error across that patient's features is
controlled by shuffling the two trial labels (--n-perm permutations,
maximum / minimum statistic), one-sided alpha per direction. Features need
--min-trials trials of each type.
Across patients: k of the n patients with that feature (coverage), Bayesian
prevalence (Ince et al. 2021, eLife 10:e62461): P(significant) = alpha +
(1 - alpha) gamma, uniform prior on gamma; MAP and 96% HPDI.

Written to --out-dir:
  seeg_prevalence_<data>.csv    one row per feature x effect x direction: k, n, prevalence
  seeg_prevalence_<data>.png    flatmaps: per region the band/second with the
                                largest share of patients significant
  seeg_prevalence_<data>_bands.png  per effect and band/second: regions where at
                                least half of the covered patients are significant
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import (BANDS, BAND_LABELS, DIVERGING, INK, INK_MUTED, draw_region_map,  # noqa: E402
                                  load_flatmap)
from signed_shap import feature_columns  # noqa: E402

CATEGORY = {"new": 1, "recent": 2, "consolidated": 3}
CLASS = {"new": 0, "recent": 1, "consolidated": 2}
CONTRASTS = {
    "recent_vs_new": ("Recent > new", "recent", "new"),
    "consolidated_vs_new": ("Consolidated > new", "consolidated", "new"),
    "consolidated_vs_recent": ("Consolidated > recent", "consolidated", "recent"),
}
EFFECTS = {**{c: v[0] for c, v in CONTRASTS.items()}, "beyond_both": "Consolidated beyond both"}


def prevalence(k, n, alpha, grid=np.linspace(0, 1, 2001)):
    """MAP and 96% HPDI of population prevalence given k of n significant (NaN if n = 0)."""
    if n == 0:
        return np.nan, np.nan, np.nan
    theta = alpha + (1 - alpha) * grid
    post = theta ** k * (1 - theta) ** (n - k)
    post /= post.sum()
    order = np.argsort(post)[::-1]
    inside = order[: np.searchsorted(np.cumsum(post[order]), 0.96) + 1]
    return grid[np.argmax(post)], grid[inside].min(), grid[inside].max()


def welch(L, M, X0, X2, min_trials):
    """Welch t for label matrices L (perms x trials, 1 = first type); NaN where too few trials."""
    Lp, Ln = L, 1 - L
    n1, n2 = Lp @ M, Ln @ M
    s1, s2, q1, q2 = Lp @ X0, Ln @ X0, Lp @ X2, Ln @ X2
    with np.errstate(invalid="ignore", divide="ignore"):
        m1, m2 = s1 / n1, s2 / n2
        v1 = (q1 - n1 * m1 ** 2) / (n1 - 1)
        v2 = (q2 - n2 * m2 ** 2) / (n2 - 1)
        t = (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)
    t[(n1 < min_trials) | (n2 < min_trials)] = np.nan
    return t


def within_patient(X, first, n_perm, rng, min_trials, alpha, chunk=50):
    """t per feature and FWE-significant masks (above, below) from a label permutation max statistic."""
    M = np.isfinite(X).astype(float)
    X0 = np.where(M > 0, X, 0.0)
    X2 = X0 ** 2
    lab = first.astype(float)
    t = welch(lab[None], M, X0, X2, min_trials)[0]
    valid = np.isfinite(t)
    mx, mn = [], []
    for start in range(0, n_perm, chunk):
        L = rng.permuted(np.tile(lab, (min(chunk, n_perm - start), 1)), axis=1)
        tp = welch(L, M, X0, X2, min_trials)[:, valid]
        mx.append(np.nanmax(tp, 1))
        mn.append(np.nanmin(tp, 1))
    hi, lo = np.quantile(np.concatenate(mx), 1 - alpha), np.quantile(np.concatenate(mn), alpha)
    return t, np.where(valid, t > hi, False), np.where(valid, t < lo, False), valid


def merged_keys(cols, regions):
    return [[regions.get(r, r) for r in cols.get_level_values("region")],
            cols.get_level_values("band"), cols.get_level_values("second")]


def load(data, kind, regions):
    """Trials x merged features table (index with image_category, subject)."""
    names = [c for c in data["features"].columns if " | " in c]
    if kind == "features":
        X = data["features"][names]
        X = X[X.index.get_level_values("image_category").isin(CATEGORY.values())]
        X.columns = feature_columns(names)
        out = X.T.groupby(merged_keys(X.columns, regions)).mean().T          # NaN skipped
        return out, None
    models = data["models"]
    test = models["test_data"]
    shap = models["test_shap_values"]
    keep = np.isin(test.get_level_values("image_category"), list(CATEGORY.values()))
    missing = data["features"].loc[test[keep], names].isna().to_numpy()
    cols = feature_columns(names)
    frames = {}
    for a, b in [("recent", "new"), ("consolidated", "new"), ("consolidated", "recent")]:
        v = shap[keep][..., CLASS[a]] - shap[keep][..., CLASS[b]]
        v = pd.DataFrame(np.where(missing, np.nan, v), index=test[keep], columns=cols)
        frames[a, b] = v.T.groupby(merged_keys(cols, regions)).sum(min_count=1).T
    return None, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--merged", type=Path, default=REPO / "electrode coverages" / "figures" /
                    "merged_regions_min4_within_RegionsROIs.csv")
    ap.add_argument("--data", choices=["features", "shap"], nargs="+", default=["features", "shap"])
    ap.add_argument("--exclude", nargs="+", default=["CNP21"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--min-trials", type=int, default=10)
    ap.add_argument("--min-patients", type=int, default=3, help="regions shown with at least this coverage")
    ap.add_argument("--n-labels", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = pd.read_pickle(args.pkl)
    regions = pd.read_csv(args.merged, dtype=str).set_index("parcel")["region"].to_dict()
    geo = load_flatmap(args.cache)

    for kind in args.data:
        X, frames = load(data, kind, regions)
        rows = []
        some = X if X is not None else next(iter(frames.values()))
        subjects = [s for s in some.index.get_level_values("subject").unique() if s not in args.exclude]
        features = some.columns
        sig = {}                                                    # (effect, dir) -> subjects x features
        for subject in subjects:
            for c, (_, a, b) in CONTRASTS.items():
                T = X if X is not None else frames[a, b]
                T = T[T.index.get_level_values("subject") == subject]
                cat = T.index.get_level_values("image_category").to_numpy()
                use = np.isin(cat, [CATEGORY[a], CATEGORY[b]])
                vals = T[use].to_numpy(float)
                t, up, down, valid = within_patient(vals, cat[use] == CATEGORY[a], args.n_perm, rng,
                                                    args.min_trials, args.alpha)
                for d, m in (("above", up), ("below", down)):
                    sig.setdefault((c, d), {})[subject] = np.where(valid, m, np.nan)
                print(f"{kind} {subject} {c}: {int(valid.sum())} features, FWE + {int(up.sum())}, - {int(down.sum())}",
                      flush=True)
        for d in ("above", "below"):
            sig["beyond_both", d] = {s: np.where(np.isnan(sig["consolidated_vs_new", d][s]) |
                                                 np.isnan(sig["consolidated_vs_recent", d][s]), np.nan,
                                                 sig["consolidated_vs_new", d][s] * sig["consolidated_vs_recent", d][s])
                                     for s in subjects}
        directions = ("above",) if kind == "shap" else ("above", "below")
        for (e, d), per in sig.items():
            if d not in directions:
                continue
            S = np.array([per[s] for s in subjects])                # subjects x features, NaN = not covered
            k, n = np.nansum(S, 0).astype(int), np.isfinite(S).sum(0)
            for j, f in enumerate(features):
                p = prevalence(k[j], n[j], args.alpha)
                rows.append(dict(region=f[0], band=f[1], second=f[2], effect=e, direction=d, k=k[j], n=n[j],
                                 prevalence_map=p[0], hpdi96_low=p[1], hpdi96_high=p[2]))
        table = pd.DataFrame(rows)
        table.round(3).to_csv(args.out_dir / f"seeg_prevalence_{kind}.csv", index=False)

        # flatmaps: per region, the band/second with the largest share of covered patients significant
        table["frac"] = np.where(table.n >= args.min_patients, table.k / table.n.where(table.n > 0), np.nan)
        table["signed"] = np.where(table.direction == "above", table.frac, -table.frac)
        fig, axes = plt.subplots(1, len(EFFECTS), figsize=(6.2 * len(EFFECTS), 7.6), facecolor="white",
                                 gridspec_kw=dict(wspace=0.03, left=0.01, right=0.99, top=0.78, bottom=0.2))
        norm = Normalize(-1, 1)
        summary = []
        for ax, (e, label) in zip(axes, EFFECTS.items()):
            t_e = table[(table.effect == e) & table.frac.notna()]
            best = t_e.loc[t_e.groupby("region").frac.idxmax()] if len(t_e) else t_e
            values = best.set_index("region").signed
            draw_region_map(ax, geo, values[values != 0], norm, n_labels=args.n_labels, regions=regions)
            top = best.sort_values("frac", ascending=False).head(4)
            ax.set_title(f"{label}\n" + "\n".join(f"{r.region} {BAND_LABELS.get(r.band, r.band)}{r.second} "
                                                   f"{r.k}/{r.n}" for r in top.head(3).itertuples() if r.k > 0),
                         fontsize=9.5, color=INK, va="bottom")
            summary.append(f"{label}: " + ", ".join(f"{r.region} {r.band}{r.second} {r.direction} {r.k}/{r.n} "
                                                     f"(prev {r.prevalence_map:.2f} [{r.hpdi96_low:.2f}, "
                                                     f"{r.hpdi96_high:.2f}])" for r in top.itertuples()))
        print("\n".join(summary))
        pos = axes[0].get_position(), axes[-1].get_position()
        cax = fig.add_axes([(pos[0].x0 + pos[1].x1) / 2 - 0.13, pos[0].y0 - 0.05, 0.26, 0.022])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
        cb.set_label("share of covered patients significant (red: first > second / above both; blue: reverse)"
                     if kind == "features" else "share of covered patients significant", fontsize=9, color=INK)
        what = ("power features, Welch t between trial types (all trials)" if kind == "features" else
                "SHAP (test set): SHAP(class A) − SHAP(class B) higher on A than on B trials")
        fig.suptitle(f"sEEG: within-patient tests and population prevalence ({kind})", fontsize=15, color=INK,
                     x=0.01, ha="left")
        fig.text(0.01, 0.1,
                 f"Per patient and merged region × band × second: {what}; family-wise error across the patient's "
                 f"features by label permutation ({args.n_perm}, max statistic), one-sided α = {args.alpha:g}.\n"
                 f"Map: per region the band/second with the largest share of covered patients significant (regions "
                 f"with ≥ {args.min_patients} patients; gray: none). Titles: best features, k/n patients. Bayesian "
                 "prevalence per feature in the csv (Ince et al. 2021).\nRegions: "
                 f"{args.merged.name}; patients: {len(subjects)}"
                 + (f" (without {', '.join(args.exclude)})" if args.exclude else "") + ".",
                 fontsize=8.5, color=INK_MUTED, va="top")
        path = args.out_dir / f"seeg_prevalence_{kind}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)

        # band x second: regions where at least half of the covered patients are significant
        fig, axes = plt.subplots(1, len(EFFECTS), figsize=(3.4 * len(EFFECTS), 3.6), facecolor="white",
                                 gridspec_kw=dict(wspace=0.35))
        for ax, (e, label) in zip(axes, EFFECTS.items()):
            t_e = table[(table.effect == e) & (table.frac >= 0.5)]
            grid = {d: t_e[t_e.direction == d].groupby(["band", "second"]).size().unstack()
                    .reindex(index=BANDS, columns=[1, 2]).fillna(0).astype(int) for d in ("above", "below")}
            ax.imshow(grid["above"] - grid["below"], cmap=DIVERGING, vmin=-10, vmax=10, aspect="auto")
            for i in range(len(BANDS)):
                for j in range(2):
                    a, b = grid["above"].iat[i, j], grid["below"].iat[i, j]
                    ax.text(j, i, f"{a}" + (f" / {b}" if kind == "features" else ""), ha="center", va="center",
                            fontsize=8, color=INK)
            ax.set_xticks([0, 1], ["1st s", "2nd s"])
            ax.set_yticks(range(len(BANDS)), [BAND_LABELS.get(b, b) for b in BANDS])
            ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
            ax.set_title(label, fontsize=10, color=INK)
            for s in ax.spines.values():
                s.set_visible(False)
        fig.suptitle(f"sEEG ({kind}): regions where ≥ half of the covered patients (≥ {args.min_patients}) are "
                     "significant" + (" (above / below)" if kind == "features" else ""),
                     fontsize=11, color=INK, x=0.02, ha="left", y=1.04)
        path = args.out_dir / f"seeg_prevalence_{kind}_bands.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
