"""NSD fMRI: recently seen vs consolidated images, per HCP-MMP1 parcel.

Uses the parcel-averaged, session-z-scored single-trial betas written by
extract_parcel_betas.py and the NSD behavioural files (responses.tsv).

Trials (correct responses only, unless --all-trials):
  new           first presentation of the image (ISOLD == 0)
  recent        seen before, last presentation <= --hours ago (in NSD always
                within the same scan session, <= ~1.5 h)
  consolidated  seen before, last presentation more than --hours ago (in NSD
                always in an earlier session, >= ~17 h, usually days)

Per subject and parcel (left and right hemisphere averaged) the mean z-scored
beta of each trial type is taken; contrasts per subject:
  consolidated - recent
  new - seen, seen = (recent + consolidated) / 2 (both weighted equally)
Subjects are then pooled: mean over subjects, two-sided one-sample t-test,
Benjamini-Hochberg FDR across parcels.

Written to --out-dir:
  nsd_<contrast>.csv                        parcel: mean, sem, t, p, q, n_subjects, n_positive
  nsd_<contrast>_subjects.csv               subject x parcel differences
  nsd_recent_vs_consolidated.png            flatmaps (group mean; FDR-significant t)
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
from plot_new_vs_repeated import DIVERGING, INK, INK_MUTED, draw_region_map, fdr_bh, load_flatmap  # noqa: E402

CONTRASTS = {
    "consolidated_vs_recent": ("Consolidated − recent", {"consolidated": 1, "recent": -1}),
    "new_vs_seen": ("New − seen", {"new": 1, "recent": -0.5, "consolidated": -0.5}),
}


def trial_table(beh: Path, hours: float, max_lag_days: float | None = None) -> pd.DataFrame:
    """Behaviour in presentation order with the trial type of every trial."""
    b = pd.read_csv(beh, sep="\t").sort_values(["SESSION", "RUN", "TRIAL"]).reset_index(drop=True)
    since = b.sort_values("TIME").groupby("73KID")["TIME"].diff() * 24          # hours, NaN on first showing
    b["since_last_h"] = since.reindex(b.index)
    b["type"] = np.where(b.ISOLD == 0, "new", np.where(b.since_last_h > hours, "consolidated", "recent"))
    if max_lag_days is not None:                         # repeats outside the lag window are left out
        b.loc[b.since_last_h > max_lag_days * 24, "type"] = "excluded"
    return b


def subject_means(data_dir: Path, subject: int, hours: float, all_trials: bool,
                  max_lag_days: float | None = None) -> pd.DataFrame | None:
    """Trial type x parcel mean (hemispheres averaged) for one subject, from the sessions available."""
    beh = trial_table(data_dir / f"subj{subject:02d}_responses.tsv", hours, max_lag_days)
    parts = []
    for session, trials in beh.groupby("SESSION"):
        files = [data_dir / f"subj{subject:02d}" / f"session{session:02d}_{h}.npy" for h in ("lh", "rh")]
        if not all(f.exists() for f in files):
            continue
        lh, rh = (np.load(f) for f in files)
        assert len(lh) == len(trials), (subject, session, len(lh), len(trials))
        values = pd.DataFrame(np.nanmean([lh, rh], 0), index=trials.index)
        parts.append(values.assign(type=trials["type"].to_numpy(), correct=trials["ISCORRECT"].to_numpy()))
    if not parts:
        return None
    df = pd.concat(parts)
    if not all_trials:
        df = df[df.correct == 1]
    counts = df["type"].value_counts()
    print(f"subj{subject:02d}: {len(parts)} sessions, trials " + ", ".join(f"{k} {counts.get(k, 0)}" for k in
                                                                         ("new", "recent", "consolidated")))
    return df.drop(columns="correct").groupby("type").mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--hours", type=float, default=3.0, help="recent/consolidated boundary")
    ap.add_argument("--all-trials", action="store_true", help="include incorrect responses")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--n-labels", type=int, default=6)
    ap.add_argument("--max-lag-days", type=float, help="leave out repeats last shown longer ago than this "
                    "(e.g. 13.18 = longest sEEG lag)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parcels = (args.data_dir / "parcels.txt").read_text().split()
    suffix = f"_maxlag{args.max_lag_days:g}d" if args.max_lag_days is not None else ""
    window = (f" Only repeats last shown ≤ {args.max_lag_days:g} days before (the sEEG lag range)."
              if args.max_lag_days is not None else "")
    means = {s: m for s in range(1, 9) if (m := subject_means(args.data_dir, s, args.hours, args.all_trials, args.max_lag_days)) is not None}
    n_subj = len(means)

    results = {}
    for name, (label, weights) in CONTRASTS.items():
        diff = pd.DataFrame({s: sum(w * m.loc[t] for t, w in weights.items()) for s, m in means.items()}).T
        diff.columns = parcels
        diff.index.name = "subject"
        t, p = sps.ttest_1samp(diff, 0, nan_policy="omit")
        res = pd.DataFrame({"mean": diff.mean(), "sem": diff.sem(), "t": t, "p": p,
                            "n_subjects": diff.notna().sum(), "n_positive": (diff > 0).sum()})
        res["q"] = fdr_bh(res["p"])
        res.index.name = "parcel"
        res.to_csv(args.out_dir / f"nsd_{name}{suffix}.csv")
        diff.to_csv(args.out_dir / f"nsd_{name}{suffix}_subjects.csv")
        results[name] = (label, res)
        top = res[res.q < args.fdr].sort_values("t")
        print(f"\n{label}: {len(top)} of {len(res)} parcels with q < {args.fdr}")
        print(pd.concat([top.head(6), top.tail(6)]).drop_duplicates()[["mean", "t", "q", "n_positive"]]
              .round(3).to_string())

    geo = load_flatmap(args.cache)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6), facecolor="white",
                             gridspec_kw=dict(wspace=0.04, hspace=0.32, left=0.02, right=0.98, top=0.92, bottom=0.1))
    for row, (name, (label, res)) in enumerate(results.items()):
        vmax = np.nanpercentile(np.abs(res["mean"]), 98)
        mean_norm = TwoSlopeNorm(0, -vmax, vmax)
        draw_region_map(axes[row, 0], geo, res["mean"], mean_norm, n_labels=args.n_labels, inset_labels=False)
        axes[row, 0].set_title(f"{chr(65 + 2 * row)}   {label}: group mean (z)", fontsize=11, color=INK, loc="left")

        sig = res.loc[res.q < args.fdr, "t"]
        tmax = np.nanpercentile(np.abs(res["t"]), 98) if res["t"].notna().any() else 1.0
        t_norm = TwoSlopeNorm(0, -tmax, tmax)
        draw_region_map(axes[row, 1], geo, sig, t_norm, n_labels=args.n_labels, inset_labels=False)
        axes[row, 1].set_title(f"{chr(66 + 2 * row)}   {label}: t, parcels with q < {args.fdr} ({len(sig)})",
                               fontsize=11, color=INK, loc="left")
        for col, (norm, unit) in enumerate([(mean_norm, "z-scored beta"), (t_norm, f"t ({n_subj - 1} df)")]):
            cax = axes[row, col].inset_axes([0.3, -0.06, 0.4, 0.03])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax, orientation="horizontal",
                              extend="both")
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
            cb.set_label(unit, fontsize=8, color=INK)

    trials = "all responses" if args.all_trials else "correct responses"
    fig.suptitle("NSD fMRI: recently seen vs consolidated images (and new vs seen)", fontsize=14, color=INK,
                 x=0.02, ha="left")
    fig.text(0.02, 0.035,
             f"Single-trial betas (betas_fithrf_GLMdenoise_RR, fsaverage), z-scored per vertex within session, averaged "
             f"within HCP-MMP1 parcels and over hemispheres. {trials.capitalize()} of {n_subj} subjects.\n"
             f"Recent: last shown ≤ {args.hours:g} h before (same session); consolidated: > {args.hours:g} h before "
             f"(earlier session); new vs seen weights recent and consolidated equally.{window}\n"
             f"Per subject the mean of each trial type is taken, then subjects are pooled: mean (left), two-sided "
             f"one-sample t-test, Benjamini-Hochberg FDR across 180 parcels (right; gray: q ≥ {args.fdr}). "
             f"Flatmap of the left hemisphere.",
             fontsize=8, color=INK_MUTED, va="top")
    path = args.out_dir / f"nsd_recent_vs_consolidated{suffix}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
