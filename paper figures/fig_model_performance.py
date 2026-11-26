"""Decoding performance of the sEEG XGBoost model: confusion matrices and per-patient accuracy.

A  Correct test trials: true label (rows) vs predicted label (columns), each row
   as % of that true label's trials (= recall on the diagonal), counts below.
B  The same for error trials (true label = the image's actual category).
C  Balanced accuracy (mean recall over the three classes) per patient, test and
   error trials; chance = 1/3.
Prediction = the class with the highest predicted probability. AUC = macro
one-vs-one ROC AUC over the three classes (chance 0.5).

Written to --out-dir: fig_model_performance.{svg,pdf,png,csv}
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shap analysis"))
from plot_new_vs_repeated import AXIS, GRID, INK, INK_MUTED  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
NAMES = ["New", "Recent", "Consolidated"]
CMAP = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cfe0f5", "#6fa3e0", "#2a78d6", "#123f7a"])
SET_COL = {"test": "#2a78d6", "error": "#eb6834"}


def predictions(data, key):
    m = data["models"]
    model = m["xgboost"]
    names = list(model.feature_names_in_)
    idx = m[key]
    X = data["features"].loc[idx, names]
    y = np.abs(idx.get_level_values("image_category").to_numpy()).astype(int) - 1
    proba = model.predict_proba(X)
    return pd.DataFrame({"subject": idx.get_level_values("subject"), "y": y, "pred": proba.argmax(1)}), proba


def balanced(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    rec = np.diag(cm) / cm.sum(1).clip(min=1)
    return rec[cm.sum(1) > 0].mean()


def matrix_panel(ax, df, proba, title):
    cm = confusion_matrix(df.y, df.pred, labels=[0, 1, 2])
    pct = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(pct, cmap=CMAP, vmin=0, vmax=70)
    for i in range(3):
        for j in range(3):
            c = "white" if pct[i, j] > 45 else INK
            ax.text(j, i - 0.08, f"{pct[i, j]:.0f}%", ha="center", va="center", fontsize=10, color=c,
                    fontweight="bold" if i == j else "normal")
            ax.text(j, i + 0.22, f"n = {cm[i, j]}", ha="center", va="center", fontsize=6.5, color=c)
    ax.set_xticks(range(3), NAMES, fontsize=8, color=INK)
    ax.set_yticks(range(3), NAMES, fontsize=8, color=INK)
    ax.set_xlabel("Predicted label", fontsize=8.5, color=INK)
    ax.set_ylabel("True label", fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    acc = (df.y == df.pred).mean()
    auc = roc_auc_score(df.y, proba, multi_class="ovo", average="macro", labels=[0, 1, 2])
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.text(0, -0.06, f"accuracy {acc:.1%} · balanced {balanced(df.y, df.pred):.1%} · AUC {auc:.2f}\n"
            f"{len(df)} trials, {df.subject.nunique()} patients", transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color=INK_MUTED)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    return dict(accuracy=acc, balanced_accuracy=balanced(df.y, df.pred), auc=auc, n=len(df))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--out-dir", type=Path, default=HERE / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_pickle(args.pkl)
    test, p_test = predictions(data, "test_data")
    err, p_err = predictions(data, "error_data")

    fig = plt.figure(figsize=(12, 4.3), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.45, left=0.06, right=0.98, top=0.8, bottom=0.17)
    summary = {"test": matrix_panel(fig.add_subplot(gs[0]), test, p_test, "A   Correct trials (test set)"),
               "error": matrix_panel(fig.add_subplot(gs[1]), err, p_err, "B   Error trials")}

    ax = fig.add_subplot(gs[2])
    per = pd.DataFrame({k: df.groupby("subject").apply(lambda d: balanced(d.y, d.pred), include_groups=False)
                        for k, df in (("test", test), ("error", err))})
    order = per["test"].sort_values().index
    x = np.arange(len(order))
    for off, k in ((-0.2, "test"), (0.2, "error")):
        ax.bar(x + off, per.loc[order, k], width=0.38, color=SET_COL[k], alpha=0.85,
               label="Correct trials (test)" if k == "test" else "Error trials")
    ax.axhline(1 / 3, color=INK, ls="--", lw=0.9)
    ax.text(len(order) - 0.4, 1 / 3, " chance", ha="left", va="center", fontsize=7, color=INK)
    ax.set_xticks(x, order, rotation=60, ha="right", fontsize=6.5, color=INK_MUTED)
    ax.set_ylabel("Balanced accuracy", fontsize=8.5, color=INK)
    ax.set_ylim(0, max(0.7, per.max().max() + 0.05))
    ax.tick_params(axis="y", labelsize=7, colors=INK_MUTED)
    ax.tick_params(axis="x", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_title("C   Per patient", loc="left", fontsize=10, color=INK)

    cax = fig.add_axes([0.06, -0.05, 0.25, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 70)), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
    cb.set_label("% of the true label's trials (chance 33%)", fontsize=7.5, color=INK)

    stem = args.out_dir / "fig_model_performance"
    pd.DataFrame(summary).T.to_csv(stem.with_suffix(".csv"))
    per.to_csv(args.out_dir / "fig_model_performance_per_patient.csv")
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print("wrote", stem.with_suffix(".svg"))
    print(pd.DataFrame(summary).T.round(3).to_string())
    print(per.round(3).describe().loc[["min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
