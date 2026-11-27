"""Shuffle control of the sEEG model: mean shuffled confusion matrix and null distributions.

Input: a pickle with a list of 3 x 3 test-set confusion matrices, one per
shuffled model (labels rolled within patient, model refitted with the best
hyperparameters; rows = true label, columns = predicted), and the real model's
summary (fig_model_performance.csv from fig_model_performance.py).

One confusion matrix: per cell the mean over shuffles of that cell as % of the
true label's trials, and the 95th percentile over shuffles; same colour scale as
the real confusion matrix. Below: accuracy and balanced accuracy of the shuffles
(mean, 95th percentile) against the real model;
p = (1 + number of shuffles >= real) / (1 + number of shuffles).

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

    fig, ax = plt.subplots(figsize=(4.4, 4.2), facecolor="white")
    mean, p95 = pct.mean(0), np.percentile(pct, 95, axis=0)
    ax.imshow(mean, cmap=CMAP, vmin=0, vmax=70)
    for i in range(3):
        for j in range(3):
            c = "white" if mean[i, j] > 45 else INK
            ax.text(j, i - 0.1, f"{mean[i, j]:.0f}%", ha="center", va="center", fontsize=10, color=c,
                    fontweight="bold" if i == j else "normal")
            ax.text(j, i + 0.22, f"95th: {p95[i, j]:.0f}%", ha="center", va="center", fontsize=6.5, color=c)
    ax.set_xticks(range(3), NAMES, fontsize=8, color=INK)
    ax.set_yticks(range(3), NAMES, fontsize=8, color=INK)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Predicted label", fontsize=8.5, color=INK)
    ax.set_ylabel("True (shuffled) label", fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Shuffled models (n = {n})", loc="left", fontsize=10, color=INK, pad=30)

    rows, lines = {}, []
    for name, null, r in (("Accuracy", acc, real.accuracy), ("Balanced accuracy", bal, real.balanced_accuracy)):
        p = (1 + np.sum(null >= r)) / (1 + n)
        rows[name] = dict(real=r, null_mean=null.mean(), null_sd=null.std(), null_p95=np.percentile(null, 95),
                          null_max=null.max(), n_shuffles=n, n_ge_real=int(np.sum(null >= r)), p=p)
        lines.append(f"{name.lower()}: shuffled mean {null.mean():.1%}, 95th percentile {np.percentile(null, 95):.1%}; "
                     f"real {r:.1%}, p {'≤' if np.sum(null >= r) == 0 else '='} {p:.3f}")
    ax.text(0, -0.05, "Cells: mean over shuffles of each true label's test trials (%), with the 95th percentile.\n"
            + "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK_MUTED)
    cell = pd.DataFrame([dict(true=NAMES[i], predicted=NAMES[j], mean_pct=mean[i, j], p95_pct=p95[i, j])
                         for i in range(3) for j in range(3)])
    cell.to_csv(args.out_dir / "fig_shuffle_performance_cells.csv", index=False)

    stem = args.out_dir / "fig_shuffle_performance"
    pd.DataFrame(rows).T.to_csv(stem.with_suffix(".csv"))
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print("wrote", stem.with_suffix(".svg"))
    print(pd.DataFrame(rows).T.round(4).to_string())


if __name__ == "__main__":
    main()
