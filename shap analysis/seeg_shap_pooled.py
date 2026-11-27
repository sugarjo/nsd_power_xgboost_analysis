"""Pooled (model-level) test of what the XGBoost model uses to separate new, recent and consolidated images.

The SHAP values come from one model trained on all patients, so the question
'which features does the model use?' is answered by pooling patients
(fixed effects: about this model and these patients, not the population).

For each pair A vs B (recent vs new, consolidated vs new, consolidated vs
recent), merged region x band x second, test-set correct trials:
  v      = SHAP(class A) - SHAP(class B) per trial (log-odds), summed over the
           parcels of a merged region, missing features left out;
  D_p    = mean(v on A trials) - mean(v on B trials) within patient p
           (> 0: the feature pushes the model towards the correct one of A, B);
  S      = mean of D_p over the patients who have the feature (each patient
           weighs equally, whatever its number of trials).
Null: A/B labels shuffled within each patient (--n-perm), family-wise error
across all features by the maximum statistic, one-sided (S > 0).
Conjunction 'consolidated separated from both': consolidated vs new and
consolidated vs recent both significant.
Population checks per feature: patients with D_p > 0 out of n, and
leave-one-patient-out: the smallest S when any one patient is left out.

Written to --out-dir:
  seeg_shap_pooled.csv    feature x effect: S, p_fwe, significant, n, n_positive, loo_min
  seeg_shap_pooled.png    flatmaps: per region the largest significant S (any band/second)
  seeg_shap_pooled_bands.png  per effect, region x band/second S for the significant regions
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
from plot_new_vs_repeated import BANDS, BAND_LABELS, INK, INK_MUTED, SEQUENTIAL, draw_region_map, load_flatmap  # noqa: E402
from seeg_prevalence import CATEGORY, CONTRASTS, EFFECTS, load  # noqa: E402


def patient_diffs(V, first, L=None, min_trials=5):
    """D = mean(v | A) - mean(v | B) per feature; L: perms x trials label matrix (default: real labels)."""
    M = np.isfinite(V).astype(float)
    V0 = np.where(M > 0, V, 0.0)
    L = first.astype(float)[None] if L is None else L
    na, nb = L @ M, (1 - L) @ M
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (L @ V0) / na - ((1 - L) @ V0) / nb
    d[(na < min_trials) | (nb < min_trials)] = np.nan
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--merged", type=Path, default=REPO / "electrode coverages" / "figures" /
                    "merged_regions_min4_within_RegionsROIs_fallback.csv")
    ap.add_argument("--exclude", nargs="+", default=["CNP21"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--min-trials", type=int, default=5, help="per class and patient")
    ap.add_argument("--min-patients", type=int, default=2, help="features need at least this many patients")
    ap.add_argument("--n-labels", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = pd.read_pickle(args.pkl)
    regions = pd.read_csv(args.merged, dtype=str).set_index("parcel")["region"].to_dict()
    _, frames = load(data, "shap", regions)
    features = next(iter(frames.values())).columns
    subjects = [s for s in next(iter(frames.values())).index.get_level_values("subject").unique()
                if s not in args.exclude]

    results, significant = [], {}
    for c, (label, a, b) in CONTRASTS.items():
        T = frames[a, b]
        D = np.full((len(subjects), len(features)), np.nan)
        null_sum = np.zeros((args.n_perm, len(features)))
        null_cnt = np.zeros((args.n_perm, len(features)))
        for i, s in enumerate(subjects):
            Ts = T[T.index.get_level_values("subject") == s]
            cat = Ts.index.get_level_values("image_category").to_numpy()
            use = np.isin(cat, [CATEGORY[a], CATEGORY[b]])
            V, first = Ts[use].to_numpy(float), cat[use] == CATEGORY[a]
            D[i] = patient_diffs(V, first, min_trials=args.min_trials)[0]
            L = rng.permuted(np.tile(first.astype(float), (args.n_perm, 1)), axis=1)
            Dp = patient_diffs(V, first, L, args.min_trials)
            ok = np.isfinite(Dp)
            null_sum += np.where(ok, Dp, 0)
            null_cnt += ok
        n = np.isfinite(D).sum(0)
        keep = n >= args.min_patients
        S = np.where(keep, np.nanmean(np.where(np.isfinite(D), D, np.nan), 0), np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            null = np.where(keep, null_sum / null_cnt, np.nan)
        max_null = np.nanmax(null, 1)
        p_fwe = (1 + (max_null[:, None] >= S[None]).sum(0)) / (1 + args.n_perm)
        sig = keep & (p_fwe < args.alpha)
        significant[c] = sig
        # leave one patient out: smallest S without any single patient
        loo = np.full(len(features), np.nan)
        for j in np.flatnonzero(sig):
            col = D[:, j][np.isfinite(D[:, j])]
            loo[j] = min((col.sum() - x) / (len(col) - 1) for x in col) if len(col) > 1 else np.nan
        for j, f in enumerate(features):
            results.append(dict(region=f[0], band=f[1], second=f[2], effect=c, S=S[j], p_fwe=p_fwe[j],
                                significant=bool(sig[j]), n=int(n[j]), n_positive=int(np.nansum(D[:, j] > 0)),
                                loo_min=loo[j]))
        print(f"{label}: {sig.sum()} of {keep.sum()} features significant (FWE {args.alpha}); "
              f"null max S 95% = {np.quantile(max_null, 1 - args.alpha):.4f}", flush=True)

    table = pd.DataFrame(results)
    conj = significant["consolidated_vs_new"] & significant["consolidated_vs_recent"]
    both = table[table.effect == "consolidated_vs_new"].reset_index(drop=True)
    other = table[table.effect == "consolidated_vs_recent"].reset_index(drop=True)
    conj_rows = both[["region", "band", "second"]].assign(
        effect="beyond_both", S=np.minimum(both.S, other.S), p_fwe=np.maximum(both.p_fwe, other.p_fwe),
        significant=conj, n=np.minimum(both.n, other.n), n_positive=np.minimum(both.n_positive, other.n_positive),
        loo_min=np.minimum(both.loo_min, other.loo_min))
    table = pd.concat([table, conj_rows], ignore_index=True)
    table.round(5).to_csv(args.out_dir / "seeg_shap_pooled.csv", index=False)
    print(f"Consolidated separated from both: {int(conj.sum())} features")
    for e, label in EFFECTS.items():
        top = table[(table.effect == e) & table.significant].sort_values("S", ascending=False).head(8)
        print(f"  {label}: " + ", ".join(f"{r.region} {r.band}{r.second} S={r.S:.4f} {r.n_positive}/{r.n} "
                                          f"loo={r.loo_min:.4f}" for r in top.itertuples()))

    # flatmaps
    geo = load_flatmap(args.cache)
    sig_t = table[table.significant]
    vmax = sig_t.S.quantile(0.95) if len(sig_t) else 1.0
    norm = Normalize(0, vmax)
    fig, axes = plt.subplots(1, len(EFFECTS), figsize=(6.2 * len(EFFECTS), 7.8), facecolor="white",
                             gridspec_kw=dict(wspace=0.03, left=0.01, right=0.99, top=0.78, bottom=0.2))
    for ax, (e, label) in zip(axes, EFFECTS.items()):
        t_e = sig_t[sig_t.effect == e]
        best = t_e.loc[t_e.groupby("region").S.idxmax()] if len(t_e) else t_e
        draw_region_map(ax, geo, best.set_index("region").S, norm, n_labels=args.n_labels, cmap=SEQUENTIAL,
                        regions=regions)
        top = best.sort_values("S", ascending=False).head(3)
        ax.set_title(f"{label}: {len(t_e)} features in {t_e.region.nunique()} regions\n"
                     + "\n".join(f"{r.region} {BAND_LABELS.get(r.band, r.band)}{r.second} "
                                 f"({r.n_positive}/{r.n} patients)" for r in top.itertuples()),
                     fontsize=9.5, color=INK, va="bottom")
    pos = axes[0].get_position(), axes[-1].get_position()
    cax = fig.add_axes([(pos[0].x0 + pos[1].x1) / 2 - 0.13, pos[0].y0 - 0.05, 0.26, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL), cax=cax, orientation="horizontal",
                      extend="max")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    cb.set_label("S: patient-averaged SHAP separation (log-odds), largest significant band/second", fontsize=9,
                 color=INK)
    fig.suptitle("sEEG model: features the XGBoost model uses to separate the trial types (pooled over patients)",
                 fontsize=15, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.1,
             "Test-set correct trials. Per patient and merged region × band × second: mean of SHAP(A) − SHAP(B) on A "
             "trials minus on B trials; S = mean over the patients with that feature (equal weight per patient).\n"
             f"Null: labels shuffled within each patient ({args.n_perm} permutations), family-wise error across "
             f"features by the maximum statistic, one-sided α = {args.alpha:g}. Gray: no significant feature. "
             "Beyond both: consolidated vs new and vs recent both significant.\nFixed-effects (this model, these "
             f"patients). Regions: {args.merged.name}; {len(subjects)} patients"
             + (f" (without {', '.join(args.exclude)})" if args.exclude else "") + ".",
             fontsize=8.5, color=INK_MUTED, va="top")
    fig.savefig(args.out_dir / "seeg_shap_pooled.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # band x second for the significant regions
    fig, axes = plt.subplots(1, len(EFFECTS), figsize=(3.6 * len(EFFECTS), 6.5), facecolor="white",
                             gridspec_kw=dict(wspace=0.9))
    cols = [(b, s) for b in BANDS for s in (1, 2)]
    for ax, (e, label) in zip(axes, EFFECTS.items()):
        t_e = table[table.effect == e]
        regs = t_e[t_e.significant].groupby("region").S.max().sort_values(ascending=False).index[:20]
        grid = (t_e[t_e.region.isin(regs)].assign(S=lambda x: x.S.where(x.significant))
                .pivot_table(index="region", columns=["band", "second"], values="S")
                .reindex(index=regs, columns=pd.MultiIndex.from_tuples(cols)))
        ax.imshow(grid.to_numpy(float), cmap=SEQUENTIAL, norm=norm, aspect="auto")
        ax.set_xticks(range(len(cols)), [f"{BAND_LABELS.get(b, b)} {s}" for b, s in cols], rotation=90)
        ax.set_yticks(range(len(regs)), list(regs))
        ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
        ax.set_title(f"{label}\n({len(regs)} regions)", fontsize=10, color=INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("sEEG model: significant features by band and second (S, blank = not significant)", fontsize=11,
                 color=INK, x=0.02, ha="left", y=1.0)
    fig.savefig(args.out_dir / "seeg_shap_pooled_bands.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", args.out_dir / "seeg_shap_pooled.png")


if __name__ == "__main__":
    main()
