"""Recognition-memory behaviour, NSD fMRI vs sEEG, as in Allen et al. (2022) Fig. 4a.

2 x 2 figure, same axes in both rows:
  left   adjusted hit rate vs time since the image was last shown (log)
  right  adjusted hit rate of recent (last shown <= --hours ago) vs
         consolidated (> --hours ago) repeats
  top    NSD (8 participants; behaviour files from the NSD bucket)
  bottom sEEG (behaviour.pkl: subject, image ID, clock_time, response)

Adjusted hit rate (cvnlab/nsddatapaper nsd_recog/get_adj_hr.m): per session,
hit rate of old trials minus the session's false-alarm rate (new trials
answered 'old'). Trials without a response count in neither (NSD codes them
ISCORRECT = NaN; for sEEG ' ' and 'technical_error' responses are dropped).
Old/new is defined by whether the image was shown before. sEEG sessions are
split at breaks longer than --session-gap hours.

Left: bins of rounded log(seconds) as in the paper; dots = one session x bin;
line = mean over participants of their session-averaged values +- 1 s.e.m.,
for bins with data from at least --min-frac of the participants (the paper
uses all, 1.0). Right: dots = sessions, thin lines = participants, black =
mean +- s.e.m.; paired t-test across participants.

Written to --out-dir: behaviour_fmri_seeg.png, behaviour_fmri_seeg.csv
(dataset x participant: recent and consolidated adjusted hit rate).
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shap analysis"))
from plot_new_vs_repeated import AXIS, GRID, INK, INK_MUTED  # noqa: E402

DOT = "#2a78d6"
COLORS = {"recent": "#2a78d6", "consolidated": "#eb6834"}
LOG_TICKS = {"1 second": 1, "1 minute": 60, "1 hour": 3600, "1 day": 86400, "1 week": 604800,
             "1 month": 2592000, "1 year": 31536000}
YLIM = (-0.6, 1.0)


def nsd_trials(data_dir: Path) -> pd.DataFrame:
    """subject, session, lag_s (NaN on first showing), isold, said_old (NaN without response)."""
    parts = []
    for s in range(1, 9):
        b = pd.read_csv(data_dir / f"subj{s:02d}_responses.tsv", sep="\t").sort_values(["SESSION", "RUN", "TRIAL"])
        lag = b.sort_values("TIME").groupby("73KID")["TIME"].diff() * 86400
        said_old = np.where(b.ISOLD == 1, b.ISCORRECT, 1 - b.ISCORRECT)             # NaN stays NaN
        parts.append(pd.DataFrame({"subject": f"subj{s:02d}", "session": b.SESSION.to_numpy(),
                                   "lag_s": lag.reindex(b.index).to_numpy(), "isold": b.ISOLD.to_numpy(),
                                   "said_old": said_old}))
    return pd.concat(parts, ignore_index=True)


def seeg_trials(path: Path, session_gap_h: float) -> pd.DataFrame:
    b = pd.DataFrame(pickle.load(open(path, "rb"))).sort_values(["subject", "clock_time"]).reset_index(drop=True)
    b["session"] = (b.groupby("subject").clock_time.diff() > session_gap_h * 3600).groupby(b.subject).cumsum()
    b["lag_s"] = b.groupby(["subject", "image ID"]).clock_time.diff()
    b["isold"] = b.lag_s.notna().astype(int)
    b["said_old"] = b.response.map({"old": 1.0, "new": 0.0})                         # others -> NaN
    return b[["subject", "session", "lag_s", "isold", "said_old"]]


def adjusted(t: pd.DataFrame, bins: pd.Series) -> pd.DataFrame:
    """(subject, session, bin) adjusted hit rate: hit rate in bin - session false-alarm rate."""
    t = t[t.said_old.notna()]
    fa = t[t.isold == 0].groupby(["subject", "session"]).said_old.mean().rename("fa")
    old = t[t.isold == 1].assign(bin=bins)
    hr = old.groupby(["subject", "session", "bin"]).said_old.mean().rename("hr").reset_index()
    hr = hr.join(fa, on=["subject", "session"])
    return hr.assign(ahr=hr.hr - hr.fa).dropna(subset=["ahr"])


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(*YLIM)
    ax.axhline(0, color=INK_MUTED, ls=":", lw=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nsd-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--seeg", type=Path, required=True, help="behaviour.pkl (extracted from behaviour.7z)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--session-gap", type=float, default=1.0, help="sEEG session break, hours")
    ap.add_argument("--min-frac", type=float, default=1.0, help="participants needed for a bin's mean line")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    data = {"NSD fMRI": nsd_trials(args.nsd_dir), "sEEG": seeg_trials(args.seeg, args.session_gap)}
    xmin, xmax = np.log(1), np.log(LOG_TICKS["1 year"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.22, hspace=0.42))
    rows = []
    for r, (name, t) in enumerate(data.items()):
        n_subj = t.subject.nunique()
        old = t.isold == 1
        # left: log time
        ax = axes[r, 0]
        logbin = pd.Series(np.round(np.log(t.loc[old, "lag_s"])), index=t.index[old])
        a = adjusted(t, logbin)
        ax.scatter(a.bin + (rng.random(len(a)) - 0.5) * 0.7, a.ahr, s=5, color=DOT, alpha=0.35, lw=0)
        subj = a.groupby(["subject", "bin"]).ahr.mean().unstack()
        keep = subj.notna().sum() >= args.min_frac * n_subj
        m, e = subj.loc[:, keep].mean(), subj.loc[:, keep].sem()
        ax.plot(m.index, m, color=INK, lw=2)
        ax.fill_between(m.index, m - e, m + e, color=INK, alpha=0.25, lw=0)
        lb = np.log(args.hours * 3600)
        ax.axvline(lb, color=INK_MUTED, lw=1, ls="--")
        ax.text(lb - 0.15, -0.52, f"recent ≤ {args.hours:g} h", ha="right", fontsize=8, color=COLORS["recent"])
        ax.text(lb + 0.15, -0.52, f"consolidated > {args.hours:g} h", ha="left", fontsize=8,
                color=COLORS["consolidated"])
        ax.set_xlim(xmin - 0.6, xmax + 0.6)
        ax.set_xticks(np.log(list(LOG_TICKS.values())), list(LOG_TICKS), rotation=45, ha="right")
        ax.set_ylabel("Adjusted hit rate", fontsize=9, color=INK)
        ax.set_title(f"{'AC'[r]}   {name}: time since last presentation ({n_subj} participants)",
                     fontsize=11, color=INK, loc="left")
        style(ax)

        # right: recent vs consolidated
        ax = axes[r, 1]
        kind = pd.Series(np.where(t.loc[old, "lag_s"] > args.hours * 3600, "consolidated", "recent"),
                         index=t.index[old])
        a = adjusted(t, kind)
        per = a.groupby(["subject", "bin"]).ahr.mean().unstack()[["recent", "consolidated"]]
        for i, k in enumerate(per.columns):
            y = a.loc[a.bin == k, "ahr"]
            ax.scatter(i + (rng.random(len(y)) - 0.5) * 0.35, y, s=7, color=COLORS[k], alpha=0.4, lw=0)
        for _, row in per.iterrows():
            ax.plot([0, 1], row.to_numpy(), color=INK_MUTED, lw=0.8, alpha=0.8)
        mm, ee = per.mean(), per.sem()
        ax.errorbar([0, 1], mm, yerr=ee, color=INK, lw=2, capsize=4, marker="o", ms=6)
        both = per.dropna()
        tt, p = sps.ttest_rel(both["recent"], both["consolidated"])
        ax.text(0.5, -0.5, f"mean {mm['recent']:.2f} vs {mm['consolidated']:.2f}\n"
                f"paired t({len(both) - 1}) = {tt:.1f}, p = {p:.1g}", ha="center", fontsize=8, color=INK)
        ax.set_xticks([0, 1], [f"Recent\n(≤ {args.hours:g} h)", f"Consolidated\n(> {args.hours:g} h)"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("Adjusted hit rate", fontsize=9, color=INK)
        ax.set_title(f"{'BD'[r]}   {name}: recent vs consolidated", fontsize=11, color=INK, loc="left")
        style(ax)
        rows.append(per.assign(dataset=name))
        print(f"{name}: {n_subj} participants, {t.groupby('subject').session.nunique().sum()} sessions; "
              f"recent {mm['recent']:.3f}, consolidated {mm['consolidated']:.3f}, t = {tt:.2f}, p = {p:.2g}; "
              f"log bins in mean line: {int(keep.sum())} of {len(keep)}")

    fig.suptitle("Recognition memory over time: NSD fMRI (top) and sEEG (bottom)", fontsize=13, color=INK,
                 x=0.07, ha="left", y=0.96)
    fig.text(0.07, 0.0, "Adjusted hit rate = hit rate minus the session's false-alarm rate (method of Allen et al. "
             "2022, Fig. 4a); trials without a response excluded. Left: rounded log(seconds) bins; dots = one "
             "session per bin;\nblack = mean over participants ± 1 s.e.m. for bins with data from "
             f"{'all' if args.min_frac == 1 else f'≥ {args.min_frac:.0%} of the'} participants. Right: dots = "
             "sessions, thin lines = participants, black = mean ± 1 s.e.m. sEEG sessions split at breaks > "
             f"{args.session_gap:g} h.", fontsize=8, color=INK_MUTED, va="top")
    out = pd.concat(rows).reset_index().set_index(["dataset", "subject"])
    out.to_csv(args.out_dir / "behaviour_fmri_seeg.csv")
    path = args.out_dir / "behaviour_fmri_seeg.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
