"""Allen et al. (2022) Fig. 4a recomputed, plus recent vs consolidated.

Follows cvnlab/nsddatapaper nsd_recog/ (get_adj_hr.m, make_behav_figure.m):
  adjusted hit rate = hit rate of old trials in a bin - false-alarm rate of
  the same session (new trials answered 'old'), computed per session and bin;
  bins of time since the last presentation: 10-day bins (ceil(floor(days)/10)*10)
  and rounded log(seconds); dots = one session each (all participants), black
  line = mean over participants of their session-averaged values (bins only
  where every participant has data) +- 1 s.e.m.
Panel C splits the same adjusted hit rate at --hours: recent (<= 3 h) vs
consolidated (> 3 h); each dot one session, lines the participants' means.

Written to --out-dir: nsd_paper_behaviour.png, nsd_paper_behaviour.csv
(participant x {recent, consolidated} adjusted hit rate).
"""

import argparse
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
LOG_TICKS = {"1 trial": 1, "1 minute": 60, "1 hour": 3600, "1 day": 86400, "1 week": 604800,
             "1 month": 2592000, "1 year": 31536000}


def behaviour(path: Path) -> pd.DataFrame:
    b = pd.read_csv(path, sep="\t").sort_values(["SESSION", "RUN", "TRIAL"]).reset_index(drop=True)
    b["lag_days"] = b.sort_values("TIME").groupby("73KID")["TIME"].diff().reindex(b.index)
    fa = 1 - b[b.ISOLD == 0].groupby("SESSION")["ISCORRECT"].mean()     # session false-alarm rate
    b["session_fa"] = b["SESSION"].map(fa)
    return b


def adjusted(b: pd.DataFrame, bins: pd.Series) -> pd.DataFrame:
    """Session x bin adjusted hit rate (old trials only)."""
    old = b[b.ISOLD == 1].assign(bin=bins)
    g = old.groupby(["SESSION", "bin"])
    return (g["ISCORRECT"].mean() - g["session_fa"].first()).unstack()


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--hours", type=float, default=3.0)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    day_tabs, log_tabs, split = {}, {}, {}
    for s in range(1, 9):
        b = behaviour(args.data_dir / f"subj{s:02d}_responses.tsv")
        old = b.ISOLD == 1
        day_tabs[s] = adjusted(b, np.ceil(np.floor(b.loc[old, "lag_days"]) / 10) * 10)
        log_tabs[s] = adjusted(b, np.round(np.log(b.loc[old, "lag_days"] * 86400)))
        kind = np.where(b.loc[old, "lag_days"] * 24 > args.hours, "consolidated", "recent")
        split[s] = adjusted(b, pd.Series(kind, index=b.index[old]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1, 1, 0.55], wspace=0.28))
    for ax, tabs, title in [(axes[0], day_tabs, "A   Time between repetitions (10-day bins)"),
                            (axes[1], log_tabs, "B   Time between repetitions (log scale)")]:
        bins = sorted(set().union(*[t.columns for t in tabs.values()]))
        width = 0.7 * np.min(np.diff(bins))
        for t in tabs.values():
            for x in t.columns:
                y = t[x].dropna()
                ax.scatter(x + (rng.random(len(y)) - 0.5) * width, y, s=4, color=DOT, alpha=0.35, lw=0)
        grp = pd.DataFrame({s: t.mean() for s, t in tabs.items()}).T.reindex(columns=bins)
        grp = grp.loc[:, grp.notna().all()]                       # bins where every participant has data
        m, e = grp.mean(), grp.sem()
        ax.plot(m.index, m, color=INK, lw=2)
        ax.fill_between(m.index, m - e, m + e, color=INK, alpha=0.25, lw=0)
        ax.axhline(0, color=INK_MUTED, ls=":", lw=1)
        ax.set_ylim(-0.6, 1)
        ax.set_ylabel("Adjusted hit rate", fontsize=9, color=INK)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        style(ax)
    axes[0].set_xlim(-5, 320)
    axes[0].set_xlabel("Time (days)", fontsize=9, color=INK)
    axes[1].set_xticks(np.log(list(LOG_TICKS.values())), list(LOG_TICKS), rotation=45, ha="right")
    lb = np.log(args.hours * 3600)
    axes[1].axvline(lb, color=COLORS["consolidated"], lw=1.2, ls="--")
    axes[1].text(lb - 0.15, -0.5, f"recent  ≤ {args.hours:g} h", ha="right", fontsize=8, color=COLORS["recent"])
    axes[1].text(lb + 0.15, -0.5, f"> {args.hours:g} h  consolidated", ha="left", fontsize=8,
                 color=COLORS["consolidated"])

    ax = axes[2]
    per_subj = pd.DataFrame({s: t.mean() for s, t in split.items()}).T[["recent", "consolidated"]]
    per_subj.index = [f"subj{s:02d}" for s in per_subj.index]
    for i, k in enumerate(per_subj.columns):
        y = pd.concat([t[k] for t in split.values()]).dropna()
        ax.scatter(i + (rng.random(len(y)) - 0.5) * 0.35, y, s=6, color=COLORS[k], alpha=0.35, lw=0)
    for _, row in per_subj.iterrows():
        ax.plot([0, 1], row.to_numpy(), color=INK_MUTED, lw=0.8, alpha=0.8)
    m, e = per_subj.mean(), per_subj.sem()
    ax.errorbar([0, 1], m, yerr=e, color=INK, lw=2, capsize=4, marker="o", ms=6)
    t, p = sps.ttest_rel(per_subj["recent"], per_subj["consolidated"])
    ax.set_xticks([0, 1], [f"Recent\n(≤ {args.hours:g} h)", f"Consolidated\n(> {args.hours:g} h)"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.6, 1)
    ax.axhline(0, color=INK_MUTED, ls=":", lw=1)
    ax.set_ylabel("Adjusted hit rate", fontsize=9, color=INK)
    ax.set_title("C   Recent vs consolidated", fontsize=11, color=INK, loc="left")
    ax.text(0.5, -0.5, f"mean {m['recent']:.2f} vs {m['consolidated']:.2f}\npaired t({len(per_subj) - 1}) = {t:.1f}, "
            f"p = {p:.1g}", ha="center", fontsize=8, color=INK)
    style(ax)
    fig.suptitle("NSD recognition memory, method of Allen et al. (2022) Fig. 4a", fontsize=13, color=INK,
                 x=0.07, ha="left", y=1.02)
    fig.text(0.07, -0.13, "Adjusted hit rate = hit rate minus the session's false-alarm rate. Dots: one scan session "
             "(all 8 participants); black: mean over participants ± 1 s.e.m. (A, B: bins where all participants have "
             "data). C: thin lines are participants.", fontsize=8, color=INK_MUTED)
    per_subj.index.name = "subject"
    per_subj.to_csv(args.out_dir / "nsd_paper_behaviour.csv")
    print(per_subj.round(3).to_string(), f"\npaired t = {t:.2f}, p = {p:.2g}")
    path = args.out_dir / "nsd_paper_behaviour.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
