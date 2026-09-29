"""Which regions, frequency bands and seconds drive the new vs repeated decision.

Uses the signed mean-contrast SHAP values from signed_shap.py (recent and old
pooled as "repeated"; > 0 means the feature pushed the trial towards its
correct label, new or repeated). The stored SHAP values are used as they are.

Aggregation, for each of region / band / second:
  1. Features that are NaN on a trial are dropped (their SHAP is not counted).
  2. Per trial, the remaining signed SHAP values of the group are summed
     (SHAP is additive, so this is the group's net contribution to the trial).
  3. Per subject, the trial values are averaged separately over new and over
     repeated trials, and the two means are averaged, so both labels weigh
     equally whatever the trial counts. Trials where the group has no present
     feature are skipped.
  4. Subjects are pooled with equal weight: the mean over subjects, with a 95%
     bootstrap confidence interval over subjects.

Panel A: HCP-MMP1 flatmap of the region means, regions covered by at least
--min-subjects subjects only (others gray). Regions outside HCP-MMP1
(Amygdala, Hippocampus, Putamen) are shown as swatches under the map.
Panel B: net contribution of the seven frequency bands. Panel C: of the two
seconds. Dots in B and C are the individual subjects.

Written to --out-dir:
  new_vs_repeated_contributions.png
  new_vs_repeated_{region,band,second}.csv   mean, CI and number of subjects
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "electrode coverages"))
from plot_electrode_coverage import (GYRUS, INK, INK_MUTED, SULCUS, draw_flat,  # noqa: E402
                                     fetch_fsaverage, orient_flatmap, parcel_border,
                                     read_patch)
from signed_shap import NEW_CATEGORY, signed_shap  # noqa: E402

BANDS = ["Delta", "Theta", "Alpha", "Beta", "GammaL", "GammaH", "Ripple"]
# Diverging blue <-> red with a neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948", "#8f1f1f"])
BAR, DOT = "#2a78d6", "#52514e"
# Parcels with too few subjects: darker than any colour on the map's scale.
NO_DATA_GYRUS, NO_DATA_SULCUS = np.array(to_rgb("#b4b3ae")), np.array(to_rgb("#9c9b96"))
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def subject_means(signed: pd.DataFrame, missing: pd.DataFrame, level: str) -> pd.DataFrame:
    """Subject x group table of label-balanced mean net contributions (steps 1-3)."""
    values = signed.where(~missing.to_numpy())
    per_trial = values.T.groupby(level=level).sum(min_count=1).T     # NaN if group absent
    idx = signed.index
    label = np.where(idx.get_level_values("image_category") == NEW_CATEGORY, "new", "repeated")
    by_label = per_trial.groupby([idx.get_level_values("subject"), label]).mean()
    return by_label.groupby(level=0).mean()


def pool(subj: pd.DataFrame, n_boot: int, seed: int = 0) -> pd.DataFrame:
    """Mean over subjects with a 95% bootstrap CI over subjects (step 4)."""
    rng = np.random.default_rng(seed)
    rows = {}
    for col, v in subj.items():
        v = v.dropna().to_numpy()
        boot = rng.choice(v, (n_boot, len(v))).mean(1)
        rows[col] = dict(mean=v.mean(), ci_low=np.percentile(boot, 2.5),
                         ci_high=np.percentile(boot, 97.5), n_subjects=len(v))
    return pd.DataFrame(rows).T.astype({"n_subjects": int})


def parcel_pieces(mask: np.ndarray, faces: np.ndarray, min_share=0.05):
    """Centres of the connected pieces of a parcel on the flat patch.

    The flatmap cut can split a parcel (e.g. 31a); every piece holding at least
    min_share of its vertices gets its own label."""
    idx = np.flatnonzero(mask)
    lut = np.full(len(mask), -1)
    lut[idx] = np.arange(len(idx))
    f = lut[faces[mask[faces].all(1)]]
    edges = np.r_[f[:, [0, 1]], f[:, [1, 2]]]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(idx),) * 2)
    _, comp = connected_components(graph, directed=False)
    sizes = np.bincount(comp)
    return [idx[comp == c] for c in np.flatnonzero(sizes >= min_share * len(idx))]


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=AXIS, lw=0.8)


def bar_panel(ax, subj: pd.DataFrame, stats: pd.DataFrame, order, labels, title):
    x = np.arange(len(order))
    s = stats.loc[order]
    ax.bar(x, s["mean"], width=0.62, color=BAR, zorder=2)
    ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["ci_low"], s["ci_high"] - s["mean"]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=3, zorder=4)
    rng = np.random.default_rng(1)
    for i, g in enumerate(order):
        v = subj[g].dropna()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=DOT, alpha=0.55,
                   linewidths=0, zorder=3)
    style_axis(ax)
    ax.set_xticks(x, labels, fontsize=8, color=INK)
    ax.set_ylabel("Net signed SHAP (log-odds)", fontsize=8, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "real_data_bipolar_xgboost_3w_random_data.pkl")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--min-subjects", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out = signed_shap(pd.read_pickle(args.pkl))
    signed, missing = out["signed"], out["missing"]
    subj = {lvl: subject_means(signed, missing, lvl) for lvl in ("region", "band", "second")}
    stats = {lvl: pool(s, args.n_boot) for lvl, s in subj.items()}
    for lvl, s in stats.items():
        s.to_csv(args.out_dir / f"new_vs_repeated_{lvl}.csv")

    region = stats["region"]
    shown = region[region["n_subjects"] >= args.min_subjects]

    # fsaverage left hemisphere, HCP-MMP1 parcels, flat patch
    paths = fetch_fsaverage(args.cache)
    pial_v, faces = [d.data for d in nib.load(paths["lh.pial.surf.gii"]).darrays]
    sulc = nib.load(paths["lh.sulc.shape.gii"]).darrays[0].data
    labels, _, names = fs.read_annot(paths["lh.HCP-MMP1.annot"])
    parcel = np.array([n.decode()[2:-4] if n.decode() != "???" else "" for n in names])[labels]
    parcel[labels < 0] = ""
    border = parcel_border(labels, faces)
    vno, patch_xy = read_patch(paths["lh.cortex.patch.flat"])
    lut = np.full(len(pial_v), -1)
    lut[vno] = np.arange(len(vno))
    patch_faces = lut[faces]
    patch_faces = patch_faces[(patch_faces >= 0).all(1)]
    patch_xy = orient_flatmap(patch_xy, parcel[vno])

    in_atlas = set(parcel)
    cortical = shown[shown.index.isin(in_atlas)]
    outside = shown[~shown.index.isin(in_atlas)]
    # 98th percentile of |mean| so a single region does not wash out the rest;
    # the colour bar marks the clipped ends.
    vmax = np.percentile(np.abs(shown["mean"]), 98)
    norm = TwoSlopeNorm(0, -vmax, vmax)

    vals = pd.Series(parcel).map(cortical["mean"]).to_numpy(float)
    has = ~np.isnan(vals)
    rgb = np.where((sulc > 0)[:, None], NO_DATA_SULCUS, NO_DATA_GYRUS)
    rgb[parcel == ""] = np.where((sulc[parcel == ""] > 0)[:, None], SULCUS, GYRUS)  # medial wall
    rgb[has] = DIVERGING(norm(vals[has]))[:, :3] * np.where(sulc[has] > 0, 0.9, 1.0)[:, None]
    rgb[border & has] = rgb[border & has] * 0.6 + 0.4 * np.array(to_rgb(INK))

    fig = plt.figure(figsize=(12, 9.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], width_ratios=[2.2, 1],
                          hspace=0.28, wspace=0.22, left=0.07, right=0.98, top=0.93, bottom=0.08)

    ax = fig.add_subplot(gs[0, :])
    draw_flat(ax, patch_xy, patch_faces, rgb[vno])
    ax.set_title("A   Regions (flatmap, left hemisphere; hemispheres pooled)",
                 fontsize=10, color=INK, loc="left")
    top = cortical["mean"].abs().nlargest(6).index
    placed = []
    for r in top:
        for piece in parcel_pieces(parcel[vno] == r, patch_faces):
            pts = patch_xy[piece]
            xy = pts[np.argmin(((pts - pts.mean(0)) ** 2).sum(1))]   # a vertex inside the piece
            # label above the parcel, or below it when a label above would collide
            crowded = any(np.hypot(*(xy - q)) < 30 and q[1] >= xy[1] for q in placed)
            placed.append(xy)
            ax.annotate(f"{r} {cortical.at[r, 'mean']:+.3f}", xy, xytext=(0, -12 if crowded else 10),
                        textcoords="offset points", ha="center", va="top" if crowded else "baseline",
                        fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.5))

    cax = ax.inset_axes([0.6, -0.07, 0.28, 0.03])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax,
                      orientation="horizontal", extend="both")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
    cb.set_label("Net signed SHAP per region (log-odds); > 0 helps the correct new/repeated call",
                 fontsize=8, color=INK)

    # Legend on the left: no-data gray, then the regions outside HCP-MMP1.
    rows = [("swatch", NO_DATA_GYRUS, f"Fewer than {args.min_subjects} subjects")]
    if len(outside):
        rows.append(("header", None, "Outside HCP-MMP1"))
        rows += [("swatch", DIVERGING(norm(row["mean"]))[:3],
                  f"{r}  {row['mean']:+.3f}  ({int(row['n_subjects'])} subjects)")
                 for r, row in outside.iterrows()]
    leg = ax.inset_axes([0.0, -0.13, 0.3, 0.14])
    leg.set_xlim(0, 1); leg.set_ylim(-len(rows) + 0.5, 0.5); leg.axis("off")
    for i, (kind, color, text) in enumerate(rows):
        if kind == "header":
            leg.text(0, -i, text, fontsize=7, color=INK_MUTED, va="center")
            continue
        leg.add_patch(plt.Rectangle((0, -i - 0.35), 0.06, 0.7, facecolor=color, edgecolor=AXIS, lw=0.5))
        leg.text(0.09, -i, text, fontsize=7, color=INK, va="center")

    bar_panel(fig.add_subplot(gs[1, 0]), subj["band"], stats["band"], BANDS,
              [b.replace("GammaL", "Low γ").replace("GammaH", "High γ") for b in BANDS],
              "B   Frequency bands")
    bar_panel(fig.add_subplot(gs[1, 1]), subj["second"], stats["second"], [1, 2],
              ["1st second", "2nd second"], "C   Seconds")

    n_subj = len(subj["band"])
    fig.suptitle("What drives the model's new vs repeated (recent + old) decision",
                 fontsize=13, color=INK, x=0.07, ha="left")
    fig.text(0.07, 0.005,
             f"Mean-contrast SHAP, NaN features left out; per trial summed within a region/band/second, "
             f"averaged within subject (new and repeated trials weighted equally), then across {n_subj} subjects.\n"
             f"A: {len(shown)} regions with ≥{args.min_subjects} subjects "
             f"({len(region) - len(shown)} with fewer shown dark gray); top 6 by |value| labeled; "
             f"colour scale clipped at ±{vmax:.3f} (98th percentile of |value|). "
             f"B, C: bars = mean over subjects, whiskers = 95% bootstrap CI over subjects, dots = subjects.",
             fontsize=7.5, color=INK_MUTED)
    fig.savefig(args.out_dir / "new_vs_repeated_contributions.png", dpi=200, bbox_inches="tight")
    print("wrote", args.out_dir / "new_vs_repeated_contributions.png")
    print(stats["band"].round(4).to_string()); print(stats["second"].round(4).to_string())


if __name__ == "__main__":
    main()
