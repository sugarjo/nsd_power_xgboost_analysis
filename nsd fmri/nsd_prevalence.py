"""Within-participant tests and Bayesian population prevalence (Ince et al. 2021, eLife 10:e62461).

Suited to NSD's design (8 participants, ~30-40 sessions each). For every
participant and vertex:
  1. per session, the mean beta difference between two trial types (raw betas,
     correct trials; within a session, so session offsets cancel);
     right hemisphere mapped onto the left (fsaverage xhemi) and averaged;
  2. one-sample t across that participant's sessions (sessions are the
     random unit, trials are not treated as independent);
  3. family-wise error control across vertices by sign-flipping sessions
     (--n-perm permutations, maximum / minimum statistic), one-sided per
     direction at --alpha.
Effects: recent > new, consolidated > new, consolidated > recent, and the
conjunction 'consolidated beyond both' (consolidated > new AND > recent, or
< both, each significant at FWE alpha in the same direction).

Across participants: k of 8 significant; Bayesian prevalence of the effect in
the population, assuming tests of sensitivity 1 and false-positive rate alpha:
P(significant) = alpha + (1 - alpha) * gamma, uniform prior on gamma; MAP and
96% highest-posterior-density interval (as in Ince et al.).
Parcel level: the same with parcel means and Bonferroni correction across parcels.

Written to --out-dir:
  nsd_prevalence[_maxlag<d>d].png
  nsd_prevalence[_maxlag<d>d]_parcels.csv   per parcel and effect/direction: participants
      significant (k), prevalence MAP and 96% HPDI, per-participant t
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shap analysis"))
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import (GYRUS, INK, INK_MUTED, NO_DATA_GYRUS, NO_DATA_SULCUS, SULCUS,  # noqa: E402
                                  draw_flat, load_flatmap)
from plot_paper_vertex import G, SUBJECTS, load_subject  # noqa: E402
from plot_vertex_bilateral import label_parcels, rh_to_lh  # noqa: E402

CONTRASTS = {
    "recent_vs_new": ("Recent > new", "recent", "cr"),
    "consolidated_vs_new": ("Consolidated > new", "consolidated", "cr"),
    "consolidated_vs_recent": ("Consolidated > recent", "consolidated", "recent"),
}


def session_diffs(data_dir, subject, window, idx, pos, neg):
    """Sessions x left vertices: per-session mean difference, hemispheres averaged (sessions with both types)."""
    out = {}
    for hemi in ("lh", "rh"):
        sessions, n, s, _ = load_subject(data_dir, subject, hemi, window)
        ok = (n[:, G[pos]] > 0) & (n[:, G[neg]] > 0)
        d = s[ok, G[pos]] / n[ok, G[pos], None] - s[ok, G[neg]] / n[ok, G[neg], None]
        out[hemi] = d
    return (out["lh"] + out["rh"][:, idx]) / 2


def t_and_null(d, n_perm, rng, chunk=100):
    """t across sessions and the sign-flip null of its max and min over columns."""
    m = len(d)
    q = (d ** 2).sum(0)
    def tstat(mean):
        var = np.maximum(q - m * mean ** 2, 1e-12) / (m - 1)
        return mean / np.sqrt(var / m)
    t = tstat(d.mean(0))
    mx, mn = [], []
    for start in range(0, n_perm, chunk):
        signs = rng.choice([-1.0, 1.0], size=(min(chunk, n_perm - start), m))
        tp = tstat(signs @ d / m)
        mx.append(tp.max(1))
        mn.append(tp.min(1))
    return t, np.concatenate(mx), np.concatenate(mn)


def prevalence(k, n, alpha, grid=np.linspace(0, 1, 2001)):
    """MAP and 96% HPDI of population prevalence gamma given k of n significant."""
    theta = alpha + (1 - alpha) * grid
    post = theta ** k * (1 - theta) ** (n - k)
    post /= post.sum()
    order = np.argsort(post)[::-1]
    inside = order[: np.searchsorted(np.cumsum(post[order]), 0.96) + 1]
    return grid[np.argmax(post)], grid[inside].min(), grid[inside].max()


def draw_counts(ax, geo, signed_k, n, border):
    cmap = LinearSegmentedColormap.from_list("cnt", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
                                                     "#f3a9a8", "#e34948", "#8f1f1f"])
    rgb = np.where((geo.sulc > 0)[:, None], NO_DATA_SULCUS, NO_DATA_GYRUS)
    medial = geo.parcel == ""
    rgb[medial] = np.where((geo.sulc[medial] > 0)[:, None], SULCUS, GYRUS)
    show = signed_k != 0
    rgb[show] = cmap((signed_k[show] + n) / (2 * n))[:, :3]
    rgb[border] = rgb[border] * 0.88 + 0.12 * np.array([0.1, 0.1, 0.1])
    draw_flat(ax, geo.patch_xy, geo.patch_faces, rgb[geo.vno])
    return cmap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--max-lag-days", type=float)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-labels", type=int, default=5)
    args = ap.parse_args()
    window = f"_maxlag{args.max_lag_days:g}d" if args.max_lag_days is not None else ""
    rng = np.random.default_rng(args.seed)

    idx = rh_to_lh(args.cache)
    geo = load_flatmap(args.cache)
    cortex = geo.parcel != ""
    parcel = pd.Series(geo.parcel)
    parcels = sorted(set(geo.parcel) - {""})
    n_sub = len(SUBJECTS)

    sig = {(c, d): [] for c in CONTRASTS for d in ("pos", "neg")}       # participants x vertices, bool
    psig = {(c, d): [] for c in CONTRASTS for d in ("pos", "neg")}      # participants x parcels, bool
    pt = {c: [] for c in CONTRASTS}                                     # participants x parcels, t
    for subject in SUBJECTS:
        for c, (_, pos, neg) in CONTRASTS.items():
            d = session_diffs(args.data_dir, subject, window, idx, pos, neg)[:, cortex]
            finite = np.isfinite(d).all(0)                  # a few vertices lack data in some participants
            t, mx, mn = t_and_null(d[:, finite], args.n_perm, rng)
            full = np.zeros(len(cortex))
            full[np.flatnonzero(cortex)[finite]] = t
            sig[c, "pos"].append(np.where(cortex, full > np.quantile(mx, 1 - args.alpha), False))
            sig[c, "neg"].append(np.where(cortex, full < np.quantile(mn, args.alpha), False))
            # parcel level: parcel means per session, one-sided t, Bonferroni across parcels
            dp = pd.DataFrame(d.T, index=geo.parcel[cortex]).groupby(level=0).mean().reindex(parcels).T.to_numpy()
            # (pandas mean skips NaN vertices)
            res = sps.ttest_1samp(dp, 0)
            pt[c].append(res.statistic)
            p_one = sps.t.sf(np.abs(res.statistic), len(dp) - 1)
            psig[c, "pos"].append((res.statistic > 0) & (p_one < args.alpha / len(parcels)))
            psig[c, "neg"].append((res.statistic < 0) & (p_one < args.alpha / len(parcels)))
            print(f"subj{subject:02d} {c}: {len(d)} sessions, FWE vertices + {sig[c, 'pos'][-1].sum()}, "
                  f"- {sig[c, 'neg'][-1].sum()}; parcels + {psig[c, 'pos'][-1].sum()}, - {psig[c, 'neg'][-1].sum()}",
                  flush=True)

    effects = {c: CONTRASTS[c][0] for c in CONTRASTS}
    for d in ("pos", "neg"):                                          # conjunction: beyond both
        sig["beyond_both", d] = [a & b for a, b in zip(sig["consolidated_vs_new", d], sig["consolidated_vs_recent", d])]
        psig["beyond_both", d] = [a & b for a, b in zip(psig["consolidated_vs_new", d],
                                                         psig["consolidated_vs_recent", d])]
    effects["beyond_both"] = "Consolidated beyond both (conjunction)"

    # parcel table
    rows = {}
    for e in effects:
        for d, word in (("pos", "above"), ("neg", "below")):
            k = np.sum(psig[e, d], 0)
            prev = [prevalence(int(x), n_sub, args.alpha) for x in k]
            rows[(e, word, "k")] = k
            rows[(e, word, "prevalence_map")] = [p[0] for p in prev]
            rows[(e, word, "hpdi96_low")] = [p[1] for p in prev]
            rows[(e, word, "hpdi96_high")] = [p[2] for p in prev]
    for c in CONTRASTS:
        for i, s in enumerate(SUBJECTS):
            rows[(c, "t", f"subj{s:02d}")] = pt[c][i]
    table = pd.DataFrame(rows, index=parcels)
    table.columns = ["__".join(x) for x in table.columns]
    table.index.name = "parcel"
    table.round(3).to_csv(args.out_dir / f"nsd_prevalence{window}_parcels.csv")

    # figure: signed number of participants with FWE-significant vertex effect
    fig, axes = plt.subplots(1, 4, figsize=(24, 7.4), facecolor="white",
                             gridspec_kw=dict(wspace=0.03, left=0.01, right=0.99, top=0.82, bottom=0.24))
    lines = []
    for ax, (e, label) in zip(axes, effects.items()):
        kp, kn = np.sum(sig[e, "pos"], 0), np.sum(sig[e, "neg"], 0)
        signed = np.where(kp >= kn, kp, -kn).astype(float)
        cmap = draw_counts(ax, geo, signed, n_sub, geo.border)
        best = pd.Series(np.abs(signed)).groupby(parcel).mean().drop("", errors="ignore")
        sgn = pd.Series(signed).groupby(parcel).mean().drop("", errors="ignore")
        top = best[best > 0].sort_values(ascending=False).index[: 2 * args.n_labels]
        label_parcels(ax, geo, list(top))
        kmax = int(np.abs(signed).max())
        p_map, lo, hi = prevalence(kmax, n_sub, args.alpha)
        ax.set_title(f"{label}\nmax {kmax}/{n_sub} participants: prevalence {p_map:.2f} [{lo:.2f}, {hi:.2f}]",
                     fontsize=11, color=INK)
        lines.append(f"{label}: vertices with k>=1 {(np.abs(signed) >= 1).sum()}, k>=4 {(np.abs(signed) >= 4).sum()}, "
                     f"k=8 {(np.abs(signed) == 8).sum()}; most consistent parcels: "
                     + ", ".join(f"{p} ({sgn[p]:+.1f})" for p in top[:8]))
    print("\n".join(lines))
    cax = fig.add_axes([0.35, 0.16, 0.3, 0.025])
    bounds = np.arange(-n_sub - 0.5, n_sub + 1.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(bounds, 256), cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=np.arange(-n_sub, n_sub + 1, 2))
    cb.ax.set_xticklabels([f"{abs(int(v))}" for v in np.arange(-n_sub, n_sub + 1, 2)])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    cb.set_label("participants with a significant effect (red: first > second / above both; blue: reverse)",
                 fontsize=9, color=INK)
    rows_prev = [prevalence(k, n_sub, args.alpha) for k in range(n_sub + 1)]
    fig.suptitle("NSD fMRI: within-participant tests and population prevalence"
                 + (f" (repeats ≤ {args.max_lag_days:g} days)" if args.max_lag_days is not None else ""),
                 fontsize=15, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.1,
             "Per participant and vertex: per-session mean beta difference (correct trials, raw betas; left + mirrored "
             "right hemisphere), one-sample t across sessions, family-wise error across vertices by sign-flipping "
             f"sessions ({args.n_perm} permutations, max statistic),\none-sided α = {args.alpha:g} per direction. "
             "Conjunction: consolidated > new and > recent (or < both), each significant. Colour: number of the "
             f"{n_sub} participants significant. Bayesian prevalence (Ince et al. 2021), MAP [96% HPDI], for k of "
             f"{n_sub} significant:\n" + ", ".join(f"{k}: {p[0]:.2f} [{p[1]:.2f}, {p[2]:.2f}]"
                                                for k, p in enumerate(rows_prev) if k >= 2),
             fontsize=8.5, color=INK_MUTED, va="top")
    path = args.out_dir / f"nsd_prevalence{window}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
