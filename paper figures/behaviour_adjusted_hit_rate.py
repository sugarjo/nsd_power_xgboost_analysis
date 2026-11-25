"""sEEG behaviour: adjusted hit rate for new, recent and consolidated images, one bar each.

Recognition responses are old/new. Per recording session (split at breaks
longer than --session-gap hours):
  recent        hit rate of repeats last shown <= --hours ago  minus the false-alarm rate
  consolidated  hit rate of repeats last shown  > --hours ago  minus the false-alarm rate
  new           correct-rejection rate (new images answered 'new') minus the
                miss rate (repeated images answered 'new'): the same correction
                with 'new' as the target response
False-alarm rate = new images answered 'old'. Trials without an old/new answer
(blank, technical_error) count in neither. 0 = chance in all three bars.
Session values are averaged within participant; bars = mean over participants
+- s.e.m., dots = participants, thin lines join each participant's bars.

--measure dprime: d' = z(hit rate) - z(false-alarm rate) per session instead,
with the log-linear correction (add 0.5 to every count, Hautus 1995) so rates
of 0 or 1 stay finite:
  recent / consolidated  z(hits of that repeat type) - z(false alarms)
  new                    z(correct rejections) - z(misses), which equals d' of
                         all repeats vs new images ("new vs seen")

Written to --out-dir: behaviour_adjusted_hit_rate.{svg,pdf,png,csv}
(behaviour_dprime.* with --measure dprime)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "nsd fmri"))
sys.path.insert(0, str(REPO / "shap analysis"))
from plot_behaviour_fmri_seeg import seeg_trials  # noqa: E402
from plot_new_vs_repeated import AXIS, GRID, INK, INK_MUTED  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
COLORS = {"new": "#7a7a7a", "recent": "#2a78d6", "consolidated": "#eb6834"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeg", type=Path, required=True, help="behaviour.pkl (extracted from behaviour.7z)")
    ap.add_argument("--hours", type=float, default=3.0, help="recent/consolidated boundary")
    ap.add_argument("--session-gap", type=float, default=1.0, help="session break, hours")
    ap.add_argument("--exclude", nargs="+", default=["CNP21"], help="participants left out (default CNP21)")
    ap.add_argument("--measure", choices=["adjusted", "dprime"], default="adjusted")
    ap.add_argument("--out-dir", type=Path, default=HERE / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t = seeg_trials(args.seeg, args.session_gap, args.exclude)
    t = t[t.said_old.notna()]
    t["type"] = np.where(t.isold == 0, "new", np.where(t.lag_s > args.hours * 3600, "consolidated", "recent"))
    key = ["subject", "session"]
    if args.measure == "dprime":
        from scipy.stats import norm
        def rate(sel):                      # log-linear corrected P('old')
            d = t[sel].groupby(key).said_old
            return (d.sum() + 0.5) / (d.count() + 1)
        z = lambda p: pd.Series(norm.ppf(p), index=p.index)
        fa, hit_all = rate(t.type == "new"), rate(t.isold == 1)
        sess = pd.DataFrame({
            "new": z(1 - fa) - z(1 - hit_all),          # = z(hit_all) - z(fa)
            "recent": z(rate(t.type == "recent")) - z(fa),
            "consolidated": z(rate(t.type == "consolidated")) - z(fa),
        })
    else:
        fa = t[t.type == "new"].groupby(key).said_old.mean()                    # new -> 'old'
        miss = 1 - t[t.isold == 1].groupby(key).said_old.mean()                 # old -> 'new'
        sess = pd.DataFrame({
            "new": (1 - fa) - miss,
            "recent": t[t.type == "recent"].groupby(key).said_old.mean() - fa,
            "consolidated": t[t.type == "consolidated"].groupby(key).said_old.mean() - fa,
        })
    subj = sess.groupby("subject").mean()
    counts = t.groupby(["subject", "type"]).size().unstack()
    name = "behaviour_dprime" if args.measure == "dprime" else "behaviour_adjusted_hit_rate"
    subj.join(counts.add_prefix("n_trials_")).to_csv(args.out_dir / f"{name}.csv")

    order = ["new", "recent", "consolidated"]
    mean, sem = subj[order].mean(), subj[order].sem()
    fig, ax = plt.subplots(figsize=(3.6, 3.4), facecolor="white")
    x = np.arange(3)
    ax.bar(x, mean, width=0.62, color=[COLORS[k] for k in order], alpha=0.85, zorder=2)
    ax.errorbar(x, mean, yerr=sem, fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=4)
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.12, 0.12, len(subj))
    for i, (_, row) in enumerate(subj[order].iterrows()):
        ax.plot(x + jit[i], row.to_numpy(), color=INK_MUTED, lw=0.5, alpha=0.5, zorder=3)
    for j, k in enumerate(order):
        ax.scatter(x[j] + jit, subj[k], s=12, color="white", edgecolor=INK, linewidths=0.7, zorder=5)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(x, ["New vs seen" if args.measure == "dprime" else "New", "Recent", "Consolidated"],
                  fontsize=9, color=INK)
    ax.set_ylabel("d′" if args.measure == "dprime" else "Adjusted hit rate", fontsize=9, color=INK)
    ax.tick_params(axis="y", labelsize=8, colors=INK_MUTED)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title(f"Recognition memory ({len(subj)} participants)", fontsize=10, color=INK, loc="left")
    stem = args.out_dir / name
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print("wrote", stem.with_suffix(".svg"))
    print(pd.DataFrame({"mean": mean, "sem": sem}).round(3).to_string())


if __name__ == "__main__":
    main()
