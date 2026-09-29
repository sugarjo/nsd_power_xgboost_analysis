"""Plot old/new recognition response matrices per subject and for the population.

`image_category` in the feature index codes each trial as
   1,  2,  3  correct response to a new, recent, old image
   4,  5,  6  no response to a new, recent, old image (excluded here)
  -1, -2, -3  wrong response to a new, recent, old image
Participants answered whether an image was novel or seen before, so a wrong
answer to a new image means "seen" and a wrong answer to a recent or old image
means "new". Each trial therefore falls in a 3 (image) x 2 (response) matrix.

Each subject's matrix is row-normalized (each image type sums to 1), and the
population matrix is the mean of the normalized subject matrices. All panels
share the color scale 0-1; the numbers in the per-subject cells are trial counts.

Written to --out-dir:
  response_matrices_per_subject.png
  response_matrix_population.png
  response_counts_per_subject.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

IMAGES = ["New", "Recent", "Old"]
RESPONSES = ["New", "Seen"]
# image_category code -> (image type, response)
CODES = {
    1: ("New", "New"), -1: ("New", "Seen"),
    2: ("Recent", "Seen"), -2: ("Recent", "New"),
    3: ("Old", "Seen"), -3: ("Old", "New"),
}

# Sequential orange ramp, light -> dark (distinct from the blue used for
# electrode coverage).
CMAP = LinearSegmentedColormap.from_list("oranges", [
    "#fde8dd", "#f9c6ad", "#f39c75", "#eb6834", "#c94f1d", "#9a3a12", "#66250a"])
NORM = Normalize(0, 1)
INK, INK_MUTED = "#1a1a19", "#5f5e5a"


def response_counts(pkl: Path) -> pd.DataFrame:
    """Long table of trial counts per subject, image type and response."""
    idx = pd.read_pickle(pkl)["features"].index.to_frame(index=False)
    idx = idx[idx["image_category"].isin(list(CODES))]   # drops unanswered 4-6
    idx[["image", "response"]] = [CODES[int(c)] for c in idx["image_category"]]
    counts = idx.groupby(["subject", "image", "response"]).size()
    full = pd.MultiIndex.from_product([sorted(idx["subject"].unique()), IMAGES, RESPONSES],
                                      names=counts.index.names)
    return counts.reindex(full, fill_value=0).rename("trials")


def draw_matrix(ax, prop, annot, title, show_x=True, show_y=True):
    ax.imshow(prop, cmap=CMAP, norm=NORM, aspect="auto")
    for (i, j), p in np.ndenumerate(prop):
        ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=8,
                color="white" if p > 0.55 else INK)
    ax.set_xticks(range(len(RESPONSES)), RESPONSES if show_x else [])
    ax.set_yticks(range(len(IMAGES)), IMAGES if show_y else [])
    ax.tick_params(length=0, labelsize=8, colors=INK_MUTED)
    ax.set_title(title, fontsize=9, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)


def add_colorbar(fig, cax):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    cb.set_label("Proportion of the image type's answered trials", fontsize=9, color=INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "real_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    counts = response_counts(args.pkl)
    counts.unstack("response")[RESPONSES].reindex(IMAGES, level="image").to_csv(args.out_dir / "response_counts_per_subject.csv")
    subjects = counts.index.get_level_values("subject").unique()

    mats = {}
    for s in subjects:
        c = counts.loc[s].unstack("response").loc[IMAGES, RESPONSES].to_numpy()
        with np.errstate(invalid="ignore"):
            mats[s] = (c, c / c.sum(1, keepdims=True))
    props = np.stack([p for _, p in mats.values()])

    # Per subject.
    ncol = 4
    nrow = int(np.ceil(len(subjects) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.1 * nrow), facecolor="white",
                             gridspec_kw=dict(hspace=0.55, wspace=0.15))
    for k, (ax, s) in enumerate(zip(axes.ravel(), subjects)):
        c, p = mats[s]
        draw_matrix(ax, np.nan_to_num(p), c.astype(int).astype(str), f"{s} (n={c.sum()})",
                    show_x=k // ncol == nrow - 1 or k + ncol >= len(subjects), show_y=k % ncol == 0)
    for ax in axes.ravel()[len(subjects):]:
        ax.axis("off")
    fig.suptitle("Recognition responses per subject (rows: image, columns: response)",
                 fontsize=12, color=INK, x=0.1, ha="left", y=0.97)
    add_colorbar(fig, fig.add_axes([0.35, 0.035, 0.3, 0.012]))
    fig.text(0.1, -0.035, "Numbers are trial counts; color is the row-normalized proportion. "
             "Unanswered trials excluded.", fontsize=8, color=INK_MUTED)
    fig.savefig(args.out_dir / "response_matrices_per_subject.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Population: mean of the row-normalized subject matrices.
    mean, sd = np.nanmean(props, 0), np.nanstd(props, 0, ddof=1)
    annot = [[f"{m:.2f}\n± {d:.2f}" for m, d in zip(mr, dr)] for mr, dr in zip(mean, sd)]
    fig, ax = plt.subplots(figsize=(3.6, 3.9), facecolor="white")
    fig.subplots_adjust(bottom=0.3, top=0.85, left=0.2)
    draw_matrix(ax, mean, annot, "")
    ax.set_xlabel("Response", fontsize=9, color=INK)
    ax.set_ylabel("Image", fontsize=9, color=INK)
    fig.suptitle(f"Recognition responses, population (n={len(subjects)})",
                 fontsize=11, color=INK, x=0.03, ha="left")
    add_colorbar(fig, fig.add_axes([0.2, 0.1, 0.72, 0.03]))
    fig.text(0.03, -0.1, "Mean ± SD across subjects of the row-normalized\nsubject matrices. "
             "Unanswered trials excluded.", fontsize=7.5, color=INK_MUTED)
    fig.savefig(args.out_dir / "response_matrix_population.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures to", args.out_dir)


if __name__ == "__main__":
    main()
