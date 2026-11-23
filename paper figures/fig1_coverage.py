"""Figure 1: electrode coverage, per HCP-MMP1 parcel and after merging low-coverage parcels.

  A  pial surface (lateral, medial, ventral), parcels coloured by the number of
     patients with a bipolar channel in them; hippocampus (anterior/posterior)
     and amygdala in a ventral glass-brain view
  B  the same coverage on the flattened cortex
  C  merged regions (merge_low_coverage.py table: >= 4 patients, merged within
     Regions ROIs, remaining regions joined across groups), coloured by the
     region's number of patients; borders between regions, merged regions numbered
     (key below)

Coverage comes from the model data (features present per patient and parcel).
Hemispheres are pooled and shown on the left fsaverage hemisphere.

Vector output (SVG and PDF; text stays editable). The cortical meshes are
embedded as high-resolution images (--dpi) inside the vector file, because a
fully vector mesh would be several hundred thousand polygons; colour bar,
region borders, numbers and text are vector. A PNG preview is written too.

Written to --out-dir: fig1_coverage.svg, .pdf, .png, fig1_merged_regions_key.csv
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
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "electrode coverages"), str(REPO / "shap analysis")]
from plot_electrode_coverage import (ASEG_LABELS, BLUES, GYRUS, INK, INK_MUTED, NOT_SHOWN, SULCUS,  # noqa: E402
                                     aseg_meshes, draw_flat, draw_glass_brain, draw_surface, fetch_fsaverage,
                                     load_coverage, orient_flatmap, parcel_border, read_patch, vertex_colors)
from plot_new_vs_repeated import largest_piece  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7})


def rasterize(ax):
    for c in ax.collections:
        c.set_rasterized(True)


def region_borders(patch_xy, patch_faces, region_of_vertex):
    """Line segments on the flat patch between triangles of different regions (shared edges)."""
    tri_region = region_of_vertex[patch_faces]
    # a triangle belongs to a region if all three vertices do; mixed triangles mark the border
    mixed = ~((tri_region[:, 0] == tri_region[:, 1]) & (tri_region[:, 1] == tri_region[:, 2]))
    segs = []
    for f in patch_faces[mixed]:
        r = region_of_vertex[f]
        mids = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if r[a] != r[b]:
                mids.append((patch_xy[f[a]] + patch_xy[f[b]]) / 2)
        if len(mids) == 2:
            segs.append(mids)
        elif len(mids) == 3:                      # three regions meet: connect edge midpoints to the centre
            c = patch_xy[f].mean(0)
            segs += [[m, c] for m in mids]
    return np.array(segs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--merged", type=Path, default=REPO / "electrode coverages" / "figures" /
                    "merged_regions_min4_staged.csv")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--out-dir", type=Path, default=HERE / "output")
    ap.add_argument("--vmax", type=int, help="top of the colour scale (default: the largest coverage of any region)")
    ap.add_argument("--dpi", type=int, default=600, help="resolution of the embedded brain images")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = fetch_fsaverage(args.cache)
    pial_v, faces = [d.data for d in nib.load(paths["lh.pial.surf.gii"]).darrays]
    sulc = nib.load(paths["lh.sulc.shape.gii"]).darrays[0].data
    labels, _, names = fs.read_annot(paths["lh.HCP-MMP1.annot"])
    parcel = np.array([n.decode()[2:-4] if n.decode() != "???" else "" for n in names])[labels]
    parcel[labels < 0] = ""
    border = parcel_border(labels, faces)

    cov = load_coverage(args.pkl)                        # patient x region, bool
    n_pat = cov.sum()
    vmax = args.vmax or int(n_pat.max())
    cmap = ListedColormap(BLUES[:1] + list(np.array(BLUES)[np.linspace(1, len(BLUES) - 1, vmax - 1)
                                                         .round().astype(int)]))
    norm = BoundaryNorm(np.arange(0.5, vmax + 1), cmap.N)
    values = pd.Series(parcel).map(n_pat).fillna(0).to_numpy()
    rgb = vertex_colors(values, sulc, border, cmap, norm)

    table = pd.read_csv(args.merged)
    region_of = table.set_index("parcel")["region"].to_dict()
    n_region = table.groupby("region")["n_subjects_region"].first()
    reg_vertex = np.array([region_of.get(p, "") if p else "" for p in parcel])
    merged_values = pd.Series(reg_vertex).map(n_region).fillna(0).to_numpy()
    no_border = np.zeros(len(parcel), bool)
    rgb_merged = vertex_colors(merged_values, sulc, no_border, cmap, norm)

    vno, patch_xy = read_patch(paths["lh.cortex.patch.flat"])
    lut = np.full(len(pial_v), -1)
    lut[vno] = np.arange(len(vno))
    patch_faces = lut[faces]
    patch_faces = patch_faces[(patch_faces >= 0).all(1)]
    patch_xy = orient_flatmap(patch_xy, parcel[vno])

    subcortical = [r for r in cov.columns if r in ASEG_LABELS and r not in NOT_SHOWN]
    meshes = aseg_meshes(paths["aseg.mgz"], subcortical)
    ghost = np.where((sulc > 0)[:, None], SULCUS, GYRUS) * 0.25 + 0.75
    sub_colors = {r: np.array(cmap(norm(n_pat[r]))[:3]) if n_pat[r] > 0 else SULCUS for r in subcortical}

    # ---- layout: 183 mm (double column) wide
    fig = plt.figure(figsize=(7.2, 6.9), facecolor="white")
    gs = fig.add_gridspec(3, 4, height_ratios=[0.72, 1.55, 0.72], hspace=0.04, wspace=0.03,
                          left=0.01, right=0.99, top=0.97, bottom=0.01)
    for col, view in enumerate(["Lateral", "Medial", "Ventral"]):
        ax = fig.add_subplot(gs[0, col])
        draw_surface(ax, pial_v, faces, rgb, view)
        rasterize(ax)
        ax.set_title(view, fontsize=7, color=INK, pad=1)
    ax = fig.add_subplot(gs[0, 3])
    names_short = {"Amygdala": "Amygdala", "Hippocampus-anterior": "aHC", "Hippocampus-posterior": "pHC"}
    draw_glass_brain(ax, pial_v, faces, ghost, meshes, sub_colors, "Ventral",
                     {r: f"{names_short.get(r, r)} ({int(n_pat[r])})" for r in subcortical})
    for t in ax.texts:
        t.set_fontsize(6)
    rasterize(ax)
    ax.set_title("Hippocampus, amygdala", fontsize=7, color=INK, pad=1)
    fig.text(0.01, 0.975, "A", fontsize=10, fontweight="bold", color=INK, va="top")

    axb = fig.add_subplot(gs[1, :2])
    draw_flat(axb, patch_xy, patch_faces, rgb[vno])
    rasterize(axb)
    axb.set_title("HCP-MMP1 parcels", fontsize=7, color=INK, pad=1)
    axc = fig.add_subplot(gs[1, 2:])
    draw_flat(axc, patch_xy, patch_faces, rgb_merged[vno])
    rasterize(axc)
    segs = region_borders(patch_xy, patch_faces, reg_vertex[vno])
    axc.add_collection(LineCollection(segs, colors=INK, linewidths=0.35))
    axc.set_title(f"Merged regions (≥ 4 patients), numbered (key below)", fontsize=7, color=INK, pad=1)
    pos_b, pos_c = axb.get_position(), axc.get_position()
    fig.text(0.01, pos_b.y1 + 0.012, "B", fontsize=10, fontweight="bold", color=INK, va="bottom")
    fig.text(pos_c.x0, pos_c.y1 + 0.012, "C", fontsize=10, fontweight="bold", color=INK, va="bottom")

    # number every region on panel C, in reading order (top to bottom in bands, left to right)
    on_map = [r for r in table.region.unique() if (reg_vertex[vno] == r).any()]
    centre = {r: np.median(patch_xy[reg_vertex[vno] == r], 0) for r in on_map}
    span = np.ptp(patch_xy[:, 1])
    order = sorted(on_map, key=lambda r: (-np.floor((centre[r][1] - patch_xy[:, 1].max()) / (span / 7)),
                                          centre[r][0]))
    key = []
    for i, r in enumerate(order, 1):
        mask = reg_vertex[vno] == r
        if not mask.any():
            continue
        pts = patch_xy[largest_piece(mask, patch_faces)]
        xy = pts[np.argmin(((pts - np.median(pts, 0)) ** 2).sum(1))]
        axc.text(*xy, str(i), fontsize=3.8, ha="center", va="center", color=INK,
                 bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        key.append((i, r, int(n_region[r])))
    pd.DataFrame(key, columns=["number", "region", "patients"]).to_csv(
        args.out_dir / "fig1_merged_regions_key.csv", index=False)

    # colour bar and key
    cax = fig.add_axes([0.04, 0.215, 0.3, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks(np.arange(1, vmax + 1, 1 if vmax <= 12 else 2))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, colors=INK_MUTED, length=0)
    cb.set_label("Patients with a bipolar channel", fontsize=7, color=INK)
    fig.text(0.04, 0.16,
             f"{len(cov)} patients; {int((n_pat.drop([r for r in cov.columns if r not in set(parcel)]) > 0).sum())} "
             "of 180 HCP-MMP1 parcels covered.\nGray: no channel. Hemispheres pooled,\nleft fsaverage hemisphere."
             + (f"\nNot shown: " + ", ".join(f"{r} ({int(n_pat[r])})" for r in cov.columns if r in NOT_SHOWN)
                if any(r in NOT_SHOWN for r in cov.columns) else ""),
             fontsize=6, color=INK_MUTED, va="top")
    ncol = 4
    per_col = int(np.ceil(len(key) / ncol))
    renderer = fig.canvas.get_renderer()
    x = 0.355
    for c in range(ncol):                         # each column starts after the widest line of the previous one
        chunk = key[c * per_col:(c + 1) * per_col]
        t = fig.text(x, 0.235, "\n".join(f"{i:>2}  {r} ({n})" for i, r, n in chunk),
                     fontsize=4.5, color=INK, va="top", family="DejaVu Sans", linespacing=1.25)
        x += t.get_window_extent(renderer).width / fig.bbox.width + 0.02

    for ext in ("svg", "pdf"):
        fig.savefig(args.out_dir / f"fig1_coverage.{ext}", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out_dir / "fig1_coverage.png", dpi=200, bbox_inches="tight")
    print("wrote", args.out_dir / "fig1_coverage.svg", "(.pdf, .png)")


if __name__ == "__main__":
    main()
