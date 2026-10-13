"""Which regions, frequency bands and seconds drive the new vs repeated decision.

Uses the mean-contrast SHAP values from signed_shap.py: recent and old are
pooled as "repeated", new_vs_rep = SHAP(new) - (SHAP(recent) + SHAP(old)) / 2.
The stored SHAP values are used as they are. Two measures (--measure):

  signed  new_vs_rep with its sign flipped on repeated trials, so > 0 means the
          feature pushed the trial towards its correct label (net help).
  abs     |new_vs_rep|: how strongly the feature moves the new vs repeated
          decision, in either direction (importance, as in a SHAP bar plot).

Aggregation, for each of region / band / second:
  1. Features that are NaN on a trial are dropped (their SHAP is not counted).
  2. Per trial, the remaining SHAP values of the group are summed (SHAP is
     additive, so this is the group's net contribution to the trial); for
     "abs" the absolute value of that sum is taken, so opposite pushes within
     a group cancel before the magnitude is measured.
  3. Per subject, the trial values are averaged separately over new and over
     repeated trials, and the two means are averaged, so both labels weigh
     equally whatever the trial counts. Trials where the group has no present
     feature are skipped.
  4. Subjects are pooled with equal weight: the mean over subjects, with a 95%
     bootstrap confidence interval over subjects.

Panel A: HCP-MMP1 flatmap of the region means, regions covered by at least
--min-subjects subjects only (others gray). Amygdala and hippocampus, which
are not HCP-MMP1 parcels, are drawn on the same colour scale in a ventral
glass-brain inset at the map's bottom left. A parcel split by the flatmap cut
is labeled once, on its largest piece.
Panel B: net contribution of each frequency band, split into its two seconds
(one bar per band x second, side by side). Bars are coloured with the map's
colormap on their own symmetric scale; dots are the individual subjects.

Written to --out-dir (suffix "_abs" for --measure abs):
  new_vs_repeated_contributions.png
  new_vs_repeated_{region,band,second,band_second}.csv   mean, CI and number of subjects
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pandas as pd
from scipy import stats as sps
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(HERE), str(REPO / "electrode coverages")]
from plot_electrode_coverage import (ASEG_LABELS, NOT_SHOWN, GYRUS, INK, INK_MUTED, SULCUS,  # noqa: E402
                                     aseg_meshes, draw_flat, draw_glass_brain,
                                     fetch_fsaverage, orient_flatmap, parcel_border,
                                     read_patch)
from signed_shap import NEW_CATEGORY, signed_shap  # noqa: E402

BANDS = ["Delta", "Theta", "Alpha", "Beta", "GammaL", "GammaH", "Ripple"]
BAND_LABELS = {"GammaL": "Low γ", "GammaH": "High γ"}
# Diverging blue <-> red with a neutral gray midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948", "#8f1f1f"])
# Sequential gray -> red for the absolute measure (no sign to show).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "gray_red", ["#f0efec", "#f3a9a8", "#e34948", "#8f1f1f", "#4d0f0f"])
BAR, DOT = "#2a78d6", "#52514e"
# Parcels with too few subjects: darker than any colour on the map's scale.
NO_DATA_GYRUS, NO_DATA_SULCUS = np.array(to_rgb("#b4b3ae")), np.array(to_rgb("#9c9b96"))
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def subject_means(signed: pd.DataFrame, missing: pd.DataFrame, level: str,
                  absolute: bool = False) -> pd.DataFrame:
    """Subject x group table of label-balanced mean contributions (steps 1-3)."""
    values = signed.where(~missing.to_numpy())
    per_trial = values.T.groupby(level=level).sum(min_count=1).T     # NaN if group absent
    if absolute:
        per_trial = per_trial.abs()
    idx = signed.index
    label = np.where(idx.get_level_values("image_category") == NEW_CATEGORY, "new", "repeated")
    by_label = per_trial.groupby([idx.get_level_values("subject"), label]).mean()
    return by_label.groupby(level=0).mean()


def pool(subj: pd.DataFrame, n_boot: int, seed: int = 0) -> pd.DataFrame:
    """Mean over subjects with 95% CIs over subjects (step 4).

    ci_low/ci_high: percentile bootstrap (resampling subjects).
    t_low/t_high:   mean +- t(0.975, n-1) * SD / sqrt(n); better calibrated
                    than the bootstrap for few subjects (NaN for n < 2).
    p_greater:      one-sided one-sample t-test p-value for mean > 0."""
    rng = np.random.default_rng(seed)
    rows = {}
    for col, v in subj.items():
        v = v.dropna().to_numpy()
        n = len(v)
        boot = rng.choice(v, (n_boot, n)).mean(1)
        sem = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        half = sps.t.ppf(0.975, n - 1) * sem if n > 1 else np.nan
        p_greater = sps.t.sf(v.mean() / sem, n - 1) if n > 1 and sem > 0 else np.nan
        rows[col] = dict(mean=v.mean(), ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                         t_low=v.mean() - half, t_high=v.mean() + half, p_greater=p_greater, n_subjects=n)
    return pd.DataFrame(rows).T.astype({"n_subjects": int})


def largest_piece(mask: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Vertices of the largest connected piece of a parcel on the flat patch.

    The flatmap cut can split a parcel (e.g. 31a); it is labeled only once."""
    idx = np.flatnonzero(mask)
    lut = np.full(len(mask), -1)
    lut[idx] = np.arange(len(idx))
    f = lut[faces[mask[faces].all(1)]]
    edges = np.r_[f[:, [0, 1]], f[:, [1, 2]]]
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(idx),) * 2)
    _, comp = connected_components(graph, directed=False)
    return idx[comp == np.bincount(comp).argmax()]


def load_flatmap(cache: Path) -> SimpleNamespace:
    """fsaverage left hemisphere with HCP-MMP1 parcels, oriented flat patch."""
    paths = fetch_fsaverage(cache)
    pial_v, faces = [d.data for d in nib.load(paths["lh.pial.surf.gii"]).darrays]
    sulc = nib.load(paths["lh.sulc.shape.gii"]).darrays[0].data
    labels, _, names = fs.read_annot(paths["lh.HCP-MMP1.annot"])
    parcel = np.array([n.decode()[2:-4] if n.decode() != "???" else "" for n in names])[labels]
    parcel[labels < 0] = ""
    vno, patch_xy = read_patch(paths["lh.cortex.patch.flat"])
    lut = np.full(len(pial_v), -1)
    lut[vno] = np.arange(len(vno))
    patch_faces = lut[faces]
    patch_faces = patch_faces[(patch_faces >= 0).all(1)]
    return SimpleNamespace(paths=paths, pial_v=pial_v, faces=faces, sulc=sulc, parcel=parcel,
                           border=parcel_border(labels, faces), vno=vno, patch_faces=patch_faces,
                           patch_xy=orient_flatmap(patch_xy, parcel[vno]), meshes={})


def fdr_bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values (q-values); NaN stays NaN."""
    ok = p.dropna().sort_values()
    m = len(ok)
    q = (ok * m / np.arange(1, m + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    return q.reindex(p.index)


def draw_region_map(ax, geo, values: pd.Series, norm, n_labels=0, inset_labels=True, cmap=DIVERGING,
                    regions: dict | None = None, highlight=()):
    """Colour HCP-MMP1 parcels by value; regions outside the atlas go in an inset.

    values: region -> value, only for the regions to show (others dark gray).
    regions: optional parcel -> merged region name; parcels of one merged
    region share its value, and borders are drawn only between regions.
    The n_labels largest |values| on the cortex are named; a parcel split by
    the flatmap cut is named once, on its largest piece."""
    parcel, sulc, vno = geo.parcel, geo.sulc, geo.vno
    if regions:
        region_of = np.array([regions.get(p, p) if p else "" for p in parcel])
        tri = region_of[geo.faces]
        edge = (tri[:, 0] != tri[:, 1]) | (tri[:, 1] != tri[:, 2])
        border = np.zeros(len(parcel), bool)
        border[geo.faces[edge].ravel()] = True
    else:
        region_of, border = parcel, geo.border
    in_atlas = values.index.isin(set(region_of))
    cortical, outside = values[in_atlas], values[~in_atlas]

    vals = pd.Series(region_of).map(cortical).to_numpy(float)
    has = ~np.isnan(vals)
    rgb = np.where((sulc > 0)[:, None], NO_DATA_SULCUS, NO_DATA_GYRUS)
    rgb[parcel == ""] = np.where((sulc[parcel == ""] > 0)[:, None], SULCUS, GYRUS)  # medial wall
    rgb[has] = cmap(norm(vals[has]))[:, :3] * np.where(sulc[has] > 0, 0.9, 1.0)[:, None]
    rgb[border & has] = rgb[border & has] * 0.6 + 0.4 * np.array(to_rgb(INK))
    if len(highlight):
        # outline: the region's border vertices plus one ring inside, in solid ink
        inside = np.isin(region_of, list(highlight))
        ring = np.zeros(len(parcel), bool)
        ring[geo.faces[(inside & border)[geo.faces].any(1)].ravel()] = True
        rgb[inside & ring] = np.array(to_rgb(INK))
    draw_flat(ax, geo.patch_xy, geo.patch_faces, rgb[vno])

    placed = []
    for r in cortical.abs().nlargest(n_labels).index:
        pts = geo.patch_xy[largest_piece(region_of[vno] == r, geo.patch_faces)]
        xy = pts[np.argmin(((pts - pts.mean(0)) ** 2).sum(1))]   # a vertex inside the piece
        # label above the parcel, or below it when a label above would collide
        crowded = any(np.hypot(*(xy - q)) < 30 for q in placed)
        placed.append(xy)
        ax.annotate(r, xy, xytext=(0, -12 if crowded else 10),
                    textcoords="offset points", ha="center", va="top" if crowded else "baseline",
                    fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.5),
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    # Amygdala / hippocampus in a ventral glass brain, in the map's empty bottom-left corner.
    subcortical = [r for r in outside.index if r in ASEG_LABELS and r not in NOT_SHOWN]
    if subcortical:
        missing = [r for r in subcortical if r not in geo.meshes]
        geo.meshes.update(aseg_meshes(geo.paths["aseg.mgz"], missing) if missing else {})
        ghost = np.where((sulc > 0)[:, None], SULCUS, GYRUS) * 0.25 + 0.75
        colors = {r: np.array(cmap(norm(outside[r]))[:3]) for r in subcortical}
        sub = ax.inset_axes([-0.04, 0.0, 0.34, 0.38])
        draw_glass_brain(sub, geo.pial_v, geo.faces, ghost, {r: geo.meshes[r] for r in subcortical},
                         colors, "Ventral", {r: r.replace("Hippocampus-", "HC\n") + (" *" if r in highlight else "")
                                             for r in subcortical} if inset_labels else None)
        if inset_labels:
            sub.set_title("Outside HCP-MMP1 (ventral view)", fontsize=7, color=INK_MUTED, pad=2)


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0, color=AXIS, lw=0.8)


def band_second_panel(ax, subj: pd.DataFrame, stats: pd.DataFrame, norm, cmap=DIVERGING,
                      ylabel="Net signed SHAP (log-odds)"):
    """One bar per band x second: the two seconds of a band side by side.

    Bars carry the map's colormap (value), so the second is told by position
    and the "1 s" / "2 s" tick under each bar, with the band name below."""
    width, gap = 0.36, 0.04
    x, ticks = [], []
    for i, band in enumerate(BANDS):
        for sec, off in ((1, -(width + gap) / 2), (2, (width + gap) / 2)):
            x.append(i + off)
            ticks.append((band, sec))
    x = np.array(x)
    s = stats.loc[ticks]
    ax.bar(x, s["mean"], width=width, color=cmap(norm(s["mean"].to_numpy(float))), zorder=2)
    ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["ci_low"], s["ci_high"] - s["mean"]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=2.5, zorder=4)
    rng = np.random.default_rng(1)
    for xi, key in zip(x, ticks):
        v = subj[key].dropna()
        ax.scatter(xi + rng.uniform(-0.11, 0.11, len(v)), v, s=8, color=DOT, alpha=0.55,
                   linewidths=0, zorder=3)
    style_axis(ax)
    ax.set_xlim(-0.6, len(BANDS) - 0.4)
    ax.set_xticks(x, [f"{sec} s" for _, sec in ticks], fontsize=7, color=INK_MUTED)
    for i, band in enumerate(BANDS):
        ax.text(i, -0.1, BAND_LABELS.get(band, band), transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=8, color=INK)
    ax.set_title("B   Frequency bands, split by second", fontsize=10, color=INK, loc="left")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--min-subjects", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--measure", choices=["signed", "abs"], default="signed",
                    help="signed: net push towards the correct label; abs: |new vs repeated| importance")
    ap.add_argument("--merged", type=Path, nargs="?", const=REPO / "electrode coverages" / "figures" / "merged_regions.csv",
                    help="parcel -> merged region table (merge_low_coverage.py); SHAP is summed within merged regions")
    ap.add_argument("--fdr", type=float, default=0.05, help="FDR level for outlining regions > 0 (signed measure)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    absolute = args.measure == "abs"
    suffix = ("_abs" if absolute else "") + ("_merged" if args.merged else "")

    out = signed_shap(pd.read_pickle(args.pkl))
    values, missing = (out["new_vs_rep"] if absolute else out["signed"]), out["missing"]
    regions = None
    if args.merged:
        # rename each parcel to its merged region; SHAP is additive, so summing a
        # merged region's features per trial gives its contribution
        regions = pd.read_csv(args.merged, dtype=str).set_index("parcel")["region"].to_dict()
        cols = values.columns
        merged_cols = pd.MultiIndex.from_arrays(
            [[regions.get(r, r) for r in cols.get_level_values("region")],
             cols.get_level_values("band"), cols.get_level_values("second")], names=cols.names)
        values = values.set_axis(merged_cols, axis=1)
        missing = missing.set_axis(merged_cols, axis=1)
    levels = {"region": "region", "band": "band", "second": "second", "band_second": ["band", "second"]}
    subj = {name: subject_means(values, missing, lvl, absolute) for name, lvl in levels.items()}
    stats = {name: pool(s, args.n_boot) for name, s in subj.items()}
    stats["band_second"].index.names = ["band", "second"]
    for name, s in stats.items():
        s.to_csv(args.out_dir / f"new_vs_repeated{suffix}_{name}.csv")

    region = stats["region"]
    shown = region[region["n_subjects"] >= args.min_subjects]

    geo = load_flatmap(args.cache)
    # 98th percentile of |mean| so a single region does not wash out the rest;
    # the colour bar marks the clipped ends.
    vmax = np.percentile(np.abs(shown["mean"]), 98)
    cmap = SEQUENTIAL if absolute else DIVERGING
    norm = Normalize(0, vmax) if absolute else TwoSlopeNorm(0, -vmax, vmax)

    fig = plt.figure(figsize=(12, 9.2), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 1], hspace=0.28,
                          left=0.07, right=0.98, top=0.93, bottom=0.1)

    ax = fig.add_subplot(gs[0])
    # signed measure: outline regions with mean > 0 after FDR correction over the
    # regions shown (one-sided t-test over subjects, Benjamini-Hochberg q < 0.05)
    shown = shown.assign(q_greater=fdr_bh(shown["p_greater"]))
    shown.to_csv(args.out_dir / f"new_vs_repeated{suffix}_region_tested.csv")
    marked = [] if absolute else list(shown.index[shown["q_greater"] < args.fdr])
    draw_region_map(ax, geo, shown["mean"], norm, n_labels=6, cmap=cmap, regions=regions, highlight=marked)
    ax.set_title("A   Regions" + (", low-coverage parcels merged" if regions else "")
                 + " (flatmap, left hemisphere; hemispheres pooled)",
                 fontsize=10, color=INK, loc="left")

    cax = ax.inset_axes([0.6, -0.07, 0.28, 0.03])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", extend="max" if absolute else "both")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
    cb.set_label("Mean |SHAP new − repeated| per region (log-odds)" if absolute else
                 "Net signed SHAP per region (log-odds); > 0 helps the correct new/repeated call",
                 fontsize=8, color=INK)

    # Swatch for parcels below the subject threshold, left of the colour bar.
    leg = ax.inset_axes([0.22, -0.07, 0.25, 0.03])
    leg.set_xlim(0, 1); leg.set_ylim(0, 1); leg.axis("off")
    leg.add_patch(plt.Rectangle((0, 0), 0.08, 1, facecolor=NO_DATA_GYRUS, edgecolor=AXIS, lw=0.5))
    leg.text(0.11, 0.5, f"Fewer than {args.min_subjects} subjects", fontsize=7, color=INK, va="center")
    if not absolute:
        leg.add_patch(plt.Rectangle((0, -1.6), 0.08, 1, facecolor="white", edgecolor=INK, lw=1.5, clip_on=False))
        leg.text(0.11, -1.1, f"> 0, FDR q < {args.fdr:g} (one-sided)", fontsize=7, color=INK, va="center")

    # Bars use the map's colormap on their own symmetric scale (the map's clipped
    # scale would saturate them).
    bar_max = np.abs(stats["band_second"]["mean"]).max()
    band_second_panel(fig.add_subplot(gs[1]), subj["band_second"], stats["band_second"],
                      Normalize(0, bar_max) if absolute else TwoSlopeNorm(0, -bar_max, bar_max), cmap,
                      "Mean |SHAP new − repeated| (log-odds)" if absolute else "Net signed SHAP (log-odds)")

    n_subj = len(subj["band"])
    fig.suptitle("How strongly each region, band and second drives the new vs repeated (recent + old) decision"
                 if absolute else "What drives the model's new vs repeated (recent + old) decision",
                 fontsize=13, color=INK, x=0.07, ha="left")
    what = ("|SHAP(new) − (SHAP(recent) + SHAP(old))/2|, NaN features left out; per trial summed within a "
            "region/band/second, then made absolute, " if absolute else
            "Mean-contrast SHAP, NaN features left out; per trial summed within a region/band/second, ")
    fig.text(0.07, 0.005 - 0.022 * (bool(regions) + (not absolute)),
             what +
             f"averaged within subject (new and repeated trials weighted equally), then across {n_subj} subjects.\n" +
             (f"A: parcels with <{args.min_subjects} subjects merged with neighbours ({args.merged.name}), "
              f"SHAP summed within each merged region;\n   " if regions else "A: ") +
             f"{len(shown)} regions with ≥{args.min_subjects} subjects "
             f"({len(region) - len(shown)} with fewer shown dark gray); top 6 by |value| named; "
             + ("" if absolute else f"\n   {len(marked)} outlined (or * in the inset): mean > 0, one-sided one-sample t-test over subjects, "
                                   f"Benjamini-Hochberg FDR q < {args.fdr:g} across the {len(shown)} regions shown; ") +
             f"colour scale clipped at {'' if absolute else '±'}{vmax:.3f} (98th percentile of |value|).\n"
             f"B: per band and second; map colormap on its own scale ({'0–' if absolute else '±'}{bar_max:.3f}); bars = mean over subjects, whiskers = 95% bootstrap CI over subjects, dots = subjects.",
             fontsize=7.5, color=INK_MUTED)
    path = args.out_dir / f"new_vs_repeated_contributions{suffix}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)
    print(stats["band_second"].round(4).to_string())


if __name__ == "__main__":
    main()
