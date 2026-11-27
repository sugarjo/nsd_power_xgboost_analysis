"""Schematic of the continuous recognition task (as in Allen et al., 2022, NSD).

A  One trial: image for 3 s, then 1 s without image; the participant answers
   whether the image has been shown before ('old') or not ('new'). The two
   analysis windows of the sEEG features (1st and 2nd second after onset) are marked.
B  A stream of trials: each image is shown up to three times. The first
   showing is new; a repeat is recent if the image was last shown <= 3 h before,
   consolidated if longer ago (up to 13.2 days in the sEEG data).
C  Trial classes for the model: correct new/recent/consolidated, and the
   corresponding error trials.

Images are drawn as abstract placeholders (no natural-scene images).

Written to --out-dir: fig_task_schematic.{svg,pdf,png}
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shap analysis"))
from plot_new_vs_repeated import AXIS, INK, INK_MUTED  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
COL = {"new": "#7a7a7a", "recent": "#2a78d6", "consolidated": "#eb6834"}
# placeholder "images": background colour + shape
IMAGES = {"A": ("#d9c7a3", "circle", "#6b8e5a"), "B": ("#a9c4d6", "triangle", "#3d5a80"),
          "C": ("#e3b5a4", "square", "#9c4f3d"), "D": ("#c9d6a3", "diamond", "#7a6a2e"),
          "E": ("#cdb8d9", "circle", "#5e4b7a")}


def draw_image(ax, x, y, size, key, label=True):
    bg, shape, fg = IMAGES[key]
    ax.add_patch(Rectangle((x, y), size, size, facecolor=bg, edgecolor=AXIS, lw=0.6))
    cx, cy, r = x + size / 2, y + size / 2, size * 0.27
    if shape == "circle":
        ax.add_patch(Circle((cx, cy), r, facecolor=fg, edgecolor="none"))
    elif shape == "triangle":
        ax.add_patch(Polygon([(cx - r, cy - r * 0.8), (cx + r, cy - r * 0.8), (cx, cy + r)], facecolor=fg))
    elif shape == "square":
        ax.add_patch(Rectangle((cx - r * 0.85, cy - r * 0.85), r * 1.7, r * 1.7, facecolor=fg))
    else:
        ax.add_patch(Polygon([(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)], facecolor=fg))
    ax.add_patch(Circle((cx, cy), size * 0.025, facecolor="#d03030", edgecolor="white", lw=0.4, zorder=5))
    if label:
        ax.text(x + size / 2, y - size * 0.12, f"image {key}", ha="center", va="top", fontsize=7, color=INK_MUTED)


def panel_trial(ax):
    ax.set_xlim(-0.6, 9.2); ax.set_ylim(-1.3, 2.9); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title("A   One trial", loc="left", fontsize=10, color=INK)
    # screens
    s = 1.6
    draw_image(ax, 0.2, 0.55, s, "A", label=False)
    ax.add_patch(Rectangle((2.6, 0.55), s, s, facecolor="#9a9a9a", edgecolor=AXIS, lw=0.6))
    ax.add_patch(Circle((2.6 + s / 2, 0.55 + s / 2), s * 0.025, facecolor="#d03030", edgecolor="white", lw=0.4))
    draw_image(ax, 5.0, 0.55, s, "B", label=False)
    ax.text(0.2 + s / 2, 2.3, "Image\n3 s", ha="center", va="bottom", fontsize=8, color=INK)
    ax.text(2.6 + s / 2, 2.3, "Gap\n1 s", ha="center", va="bottom", fontsize=8, color=INK)
    ax.text(5.0 + s / 2, 2.3, "Next trial", ha="center", va="bottom", fontsize=8, color=INK_MUTED)
    ax.text(7.0, 1.35, "“Have you seen\nthis image before?”\n\nold  /  new", ha="left", va="center",
            fontsize=8, color=INK)
    # time axis: 0..4 s mapped to x 0.2..4.2
    x0, k = 0.2, 1.0
    ax.annotate("", (4.6, 0.0), (0.0, 0.0), arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.8))
    for tsec in range(0, 5):
        ax.plot([x0 + k * tsec] * 2, [-0.06, 0.06], color=INK_MUTED, lw=0.8)
        ax.text(x0 + k * tsec, -0.15, f"{tsec}", ha="center", va="top", fontsize=7, color=INK_MUTED)
    ax.text(4.65, 0.0, "s", ha="left", va="center", fontsize=7, color=INK_MUTED)
    ax.add_patch(Rectangle((x0, 0.12), 3 * k, 0.16, facecolor="#cfcfcf", edgecolor="none"))
    ax.text(x0 + 1.5 * k, 0.33, "image on screen", ha="center", va="bottom", fontsize=6.5, color=INK_MUTED)
    for i, (lab, c) in enumerate((("1st second", "#bfa3d9"), ("2nd second", "#8f6bb8"))):
        ax.add_patch(Rectangle((x0 + i * k, -0.75), k, 0.22, facecolor=c, edgecolor="white", lw=0.8))
        ax.text(x0 + (i + 0.5) * k, -0.64, lab, ha="center", va="center", fontsize=6.5, color="white")
    ax.text(x0 + 2.15 * k, -0.64, "sEEG features\n(band power)", ha="left", va="center", fontsize=6.5,
            color=INK_MUTED)


def panel_stream(ax):
    ax.set_xlim(-0.3, 13.4); ax.set_ylim(-1.7, 2.6); ax.axis("off"); ax.set_aspect("equal")
    ax.set_title("B   Each image is shown up to three times", loc="left", fontsize=10, color=INK)
    seq = [("A", "new"), ("B", "new"), ("A", "recent"), ("C", "new"), ("B", "recent"),
           None, ("A", "consolidated"), ("D", "new"), ("C", "consolidated"), ("B", "consolidated")]
    s, x = 0.95, 0.0
    xs = {}
    for item in seq:
        if item is None:                         # gap: hours to days later
            ax.text(x + 0.35, 0.55 + s / 2, "···", ha="center", va="center", fontsize=14, color=INK_MUTED)
            ax.text(x + 0.35, 0.2, "> 3 h\nlater", ha="center", va="top", fontsize=6.5, color=INK_MUTED)
            x += 0.9
            continue
        key, kind = item
        draw_image(ax, x, 0.55, s, key, label=False)
        ax.add_patch(Rectangle((x, 0.55 - 0.16), s, 0.12, facecolor=COL[kind], edgecolor="none"))
        ax.text(x + s / 2, 0.3, key, ha="center", va="top", fontsize=7, color=INK_MUTED)
        xs.setdefault(key, []).append(x + s / 2)
        x += s + 0.25
    # arcs for image A
    a = xs["A"]
    for (x1, x2), lab in zip(zip(a[:-1], a[1:]), ("≤ 3 h: recent", "> 3 h: consolidated")):
        ax.annotate("", (x2, 1.62), (x1, 1.62), arrowprops=dict(arrowstyle="->", color=INK, lw=0.8,
                                                                connectionstyle="arc3,rad=-0.35"))
        ax.text((x1 + x2) / 2, 2.0 if x2 - x1 < 4 else 2.42, lab, ha="center", va="bottom", fontsize=7,
                color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    # legend
    for i, (kind, txt) in enumerate((("new", "New: first showing"),
                                     ("recent", "Recent: last shown ≤ 3 h before"),
                                     ("consolidated", "Consolidated: last shown > 3 h before (up to 13.2 days)"))):
        y = -0.45 - i * 0.4
        ax.add_patch(Rectangle((0.0, y - 0.08), 0.45, 0.16, facecolor=COL[kind], edgecolor="none"))
        ax.text(0.6, y, txt, ha="left", va="center", fontsize=7.5, color=INK)


def panel_classes(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
    ax.set_title("C   Trial classes", loc="left", fontsize=10, color=INK)
    cols = [("Image", 0.2), ("Correct answer", 3.0), ("Model label", 5.6), ("Wrong answer", 7.9)]
    for name, x in cols:
        ax.text(x, 3.6, name, fontsize=8, color=INK, fontweight="bold", va="center")
    rows = [("new", "New", "“new”", "new (1)", "error (−1)"),
            ("recent", "Recent", "“old”", "recent (2)", "error (−2)"),
            ("consolidated", "Consolidated", "“old”", "consolidated (3)", "error (−3)")]
    for i, (kind, img, ans, lab, err) in enumerate(rows):
        y = 2.8 - i * 0.95
        ax.add_patch(FancyBboxPatch((0.2, y - 0.25), 2.3, 0.5, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=COL[kind], edgecolor="none", alpha=0.85))
        ax.text(1.35, y, img, ha="center", va="center", fontsize=8, color="white")
        ax.text(3.0, y, ans, va="center", fontsize=8, color=INK)
        ax.text(5.6, y, lab, va="center", fontsize=8, color=INK)
        ax.text(7.9, y, err, va="center", fontsize=8, color=INK_MUTED)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=HERE / "output")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 6.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 0.85], hspace=0.25, wspace=0.08,
                          left=0.02, right=0.99, top=0.95, bottom=0.03)
    panel_trial(fig.add_subplot(gs[0, 0]))
    panel_classes(fig.add_subplot(gs[0, 1]))
    panel_stream(fig.add_subplot(gs[1, :]))
    stem = args.out_dir / "fig_task_schematic"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print("wrote", stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
