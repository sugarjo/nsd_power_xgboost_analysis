"""NSD fMRI contrasts computed as in Allen et al. (2022, Nat Neurosci), Fig. 4b.

Their method (figure legend): two-sample t-tests on NSD betas contrasting
'hits' > 'correct rejections', trials pooled across all NSD scan sessions,
per participant; results shown for participant 1 and as a simple average of
the t maps across participants, thresholded at |t| > 3, with a control in
which trial labels are shuffled.

Here the same test is run per HCP-MMP1 parcel (hemispheres averaged) on the
session-z-scored single-trial betas from extract_parcel_betas.py:
  seen_vs_new             hits (old image, answered old) > correct rejections
                          (new image, answered new)            -- the paper
  consolidated_vs_recent  consolidated hits > recent hits      -- same method
recent: last shown <= --hours ago; consolidated: > --hours ago.

Written to --out-dir:
  nsd_paper_<contrast>.csv        parcel x subject t, mean t, shuffled mean t
  nsd_paper_method.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shap analysis"))
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import DIVERGING, INK, INK_MUTED, draw_region_map, load_flatmap  # noqa: E402
from plot_recent_vs_consolidated import trial_table  # noqa: E402

CONTRASTS = {  # name: (title, positive trial type, negative trial type)
    "seen_vs_new": ("Hits > correct rejections (paper)", "seen", "new"),
    "consolidated_vs_recent": ("Consolidated hits > recent hits", "consolidated", "recent"),
}


def subject_trials(data_dir: Path, subject: int, hours: float) -> pd.DataFrame:
    """Correct trials x parcels (hemispheres averaged) with their trial type, all sessions."""
    beh = trial_table(data_dir / f"subj{subject:02d}_responses.tsv", hours)
    parts = []
    for session, trials in beh.groupby("SESSION"):
        lh, rh = (np.load(data_dir / f"subj{subject:02d}" / f"session{session:02d}_{h}.npy") for h in ("lh", "rh"))
        assert len(lh) == len(trials), (subject, session)
        values = pd.DataFrame(np.nanmean([lh, rh], 0), index=trials.index)
        parts.append(values.assign(type=trials["type"].to_numpy(), correct=trials["ISCORRECT"].to_numpy()))
    df = pd.concat(parts)
    return df[df.correct == 1].drop(columns="correct")


def two_sample_t(df: pd.DataFrame, pos: str, neg: str, labels: np.ndarray | None = None) -> np.ndarray:
    types = df["type"].to_numpy() if labels is None else labels
    if pos == "seen":                                   # hits = recent + consolidated
        types = np.where(np.isin(types, ["recent", "consolidated"]), "seen", types)
    x = df.drop(columns="type").to_numpy()
    return sps.ttest_ind(x[types == pos], x[types == neg], nan_policy="omit").statistic


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--n-labels", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    parcels = (args.data_dir / "parcels.txt").read_text().split()
    subjects = range(1, 9)
    tables = {name: {} for name in CONTRASTS}
    for s in subjects:
        df = subject_trials(args.data_dir, s, args.hours)
        n = df["type"].value_counts()
        print(f"subj{s:02d}: hits {n.get('recent', 0) + n.get('consolidated', 0)} (recent {n.get('recent', 0)}, "
              f"consolidated {n.get('consolidated', 0)}), correct rejections {n.get('new', 0)}")
        for name, (_, pos, neg) in CONTRASTS.items():
            sel = df if pos == "seen" else df[df["type"].isin([pos, neg])]
            tables[name][f"subj{s:02d}"] = two_sample_t(sel, pos, neg)
            tables[name][f"subj{s:02d}_shuffled"] = two_sample_t(sel, pos, neg, rng.permutation(sel["type"].to_numpy()))

    geo = load_flatmap(args.cache)
    fig, axes = plt.subplots(2, 3, figsize=(19, 9.6), facecolor="white",
                             gridspec_kw=dict(wspace=0.03, hspace=0.3, left=0.02, right=0.98, top=0.92, bottom=0.1))
    norm = TwoSlopeNorm(0, -10, 10)
    for row, (name, (label, _, _)) in enumerate(CONTRASTS.items()):
        t = pd.DataFrame(tables[name], index=parcels)
        t.index.name = "parcel"
        real = t[[f"subj{s:02d}" for s in subjects]]
        shuf = t[[f"subj{s:02d}_shuffled" for s in subjects]]
        t["mean_t"] = real.mean(axis=1)
        t["mean_t_shuffled"] = shuf.mean(axis=1)
        t["n_positive"] = (real > 0).sum(axis=1)
        t.to_csv(args.out_dir / f"nsd_paper_{name}.csv")
        for col, (values, what) in enumerate([(t["subj01"], "participant 1"),
                                              (t["mean_t"], "average of 8 participants"),
                                              (t["mean_t_shuffled"], "average, shuffled labels")]):
            shown = values[values.abs() > args.threshold]
            draw_region_map(axes[row, col], geo, shown, norm, n_labels=args.n_labels, inset_labels=False)
            axes[row, col].set_title(f"{chr(65 + 3 * row + col)}   {label}: {what} ({len(shown)})",
                                     fontsize=10.5, color=INK, loc="left")
        top = t.loc[t["mean_t"].abs() > args.threshold, ["mean_t", "n_positive"]].sort_values("mean_t")
        print(f"\n{label}: {len(top)} parcels with |mean t| > {args.threshold}")
        print(pd.concat([top.head(6), top.tail(6)]).drop_duplicates().round(2).to_string())

    cax = fig.add_axes([0.4, 0.085, 0.2, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax, orientation="horizontal",
                      extend="both")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
    cb.set_label("two-sample t", fontsize=8, color=INK)
    fig.suptitle("NSD fMRI, method of Allen et al. (2022) Fig. 4b: hits vs correct rejections, and consolidated vs recent",
                 fontsize=14, color=INK, x=0.02, ha="left")
    fig.text(0.02, 0.03,
             f"Per participant: two-sample t-test between trial types, correct trials pooled across all scan sessions "
             f"(betas_fithrf_GLMdenoise_RR, fsaverage, z-scored per vertex within session, averaged within HCP-MMP1 "
             f"parcels and over hemispheres).\nGroup: simple average of the participants' t values. Shown where "
             f"|t| > {args.threshold:g} (gray otherwise), as in the paper. Recent: last shown ≤ {args.hours:g} h before; "
             f"consolidated: > {args.hours:g} h before. Right column: trial labels shuffled within participant.",
             fontsize=8, color=INK_MUTED, va="top")
    path = args.out_dir / "nsd_paper_method.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
