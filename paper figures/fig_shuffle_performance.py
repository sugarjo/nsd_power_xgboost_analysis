"""Shuffle control of the sEEG model: mean shuffled confusion matrix and null distributions.

Input: a pickle with a list of 3 x 3 test-set confusion matrices, one per
shuffled model (labels rolled within patient, model refitted with the best
hyperparameters; rows = true label, columns = predicted), and the real model's
summary (fig_model_performance.csv from fig_model_performance.py).

A  Mean over shuffles of each row as % of that true label's trials (± SD over
   shuffles below), on the same colour scale as the real confusion matrix.
B  Null distributions of accuracy and balanced accuracy, with the real model's
   value; p = (1 + number of shuffles >= real) / (1 + number of shuffles).

Written to --out-dir: fig_shuffle_performance.{svg,pdf,png,csv}
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shap analysis"))
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import AXIS, GRID, INK, INK_MUTED  # noqa: E402
from fig_model_performance import CMAP, NAMES  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shuffled", type=Path, required=True, help="pickle: list of 3x3 confusion matrices")
    ap.add_argument("--real", type=Path, default=HERE / "output_real" / "fig_model_performance.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "output_real")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cms = np.array(pickle.load(open(args.shuffled, "rb")), dtype=float)          # shuffles x 3 x 3
    pct = cms / cms.sum(2, keepdims=True) * 100
    acc = np.trace(cms, axis1=1, axis2=2) / cms.sum((1, 2))
    bal = (np.diagonal(cms, axis1=1, axis2=2) / cms.sum(2)).mean(1)
    real = pd.read_csv(args.real, index_col=0).loc["test"]
    n = len(cms)

    fig = plt.figure(figsize=(10.5, 3.9), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.05, 1.05], wspace=0.45, left=0.07, right=0.98, top=0.8,
                          bottom=0.17)
    ax = fig.add_subplot(gs[0])
    mean, sd = pct.mean(0), pct.std(0)
    ax.imshow(mean, cmap=CMAP, vmin=0, vmax=70)
    for i in range(3):
        for j in range(3):
            c = "white" if mean[i, j] > 45 else INK
            ax.text(j, i - 0.08, f"{mean[i, j]:.0f}%", ha="center", va="center", fontsize=10, color=c,
                    fontweight="bold" if i == j else "normal")
            ax.text(j, i + 0.22, f"± {sd[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color=c)
    ax.set_xticks(range(3), NAMES, fontsize=8, color=INK)
    ax.set_yticks(range(3), NAMES, fontsize=8, color=INK)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Predicted label", fontsize=8.5, color=INK)
    ax.set_ylabel("True (shuffled) label", fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"A   Shuffled models (mean of {n})", loc="left", fontsize=10, color=INK, pad=28)
    ax.text(0, -0.06, f"accuracy {acc.mean():.1%} · balanced {bal.mean():.1%} (means over shuffles)\n"
            "% of each true label's test trials; ± SD over shuffles", transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=INK_MUTED)

    rows = {}
    for k, (name, null, r) in enumerate((("Accuracy", acc, real.accuracy),
                                         ("Balanced accuracy", bal, real.balanced_accuracy))):
        ax = fig.add_subplot(gs[k + 1])
        p = (1 + np.sum(null >= r)) / (1 + n)
        lo = min(null.min(), 1 / 3) - 0.02
        bins = np.linspace(lo, max(null.max(), r) + 0.02, 45)
        ax.hist(null, bins=bins, color="#b9b9b9", edgecolor="white", lw=0.4)
        ax.axvline(1 / 3, color=INK_MUTED, ls=":", lw=1)
        ax.axvline(np.percentile(null, 95), color=INK_MUTED, ls="--", lw=0.9)
        ax.axvline(r, color="#d03030", lw=2)
        top = ax.get_ylim()[1]
        ax.text(r, top * 0.97, f" real\n {r:.1%}", color="#d03030", fontsize=8, ha="left", va="top")
        ax.text(1 / 3, top * 0.97, "chance ", color=INK_MUTED, fontsize=7, ha="right", va="top")
        ax.text(np.percentile(null, 95), top * 0.75, " 95th\n percentile", color=INK_MUTED, fontsize=6.5,
                ha="left", va="top")
        ax.set_xlabel(name, fontsize=8.5, color=INK)
        ax.set_ylabel("Shuffles" if k == 0 else "", fontsize=8.5, color=INK)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.tick_params(labelsize=7, colors=INK_MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_title(f"{'BC'[k]}   {name}: p = {p:.3f}" + (" (minimum)" if np.sum(null >= r) == 0 else ""),
                     loc="left", fontsize=10, color=INK)
        rows[name] = dict(real=r, null_mean=null.mean(), null_sd=null.std(), null_p95=np.percentile(null, 95),
                          null_max=null.max(), n_shuffles=n, n_ge_real=int(np.sum(null >= r)), p=p)

    stem = args.out_dir / "fig_shuffle_performance"
    pd.DataFrame(rows).T.to_csv(stem.with_suffix(".csv"))
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print("wrote", stem.with_suffix(".svg"))
    print(pd.DataFrame(rows).T.round(4).to_string())


if __name__ == "__main__":
    main()
