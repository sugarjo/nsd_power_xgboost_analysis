"""Plot electrode (bipolar channel) coverage on fsaverage.

The feature table stores one set of band-power features per HCP-MMP1 parcel
(plus Amygdala, Hippocampus and Putamen), and a parcel is NaN for subjects
without a bipolar channel in it. Coverage is therefore defined per parcel as
the number of subjects that have data in that parcel. Hemispheres are pooled in
the feature names, so everything is drawn on the left hemisphere.

Regions that are not in HCP-MMP1 (Amygdala, Hippocampus, Putamen) are drawn as
meshes extracted from the fsaverage aseg segmentation, inside a faded pial
surface. All group figures share one color scale running from 1 to the number
of subjects; the per-subject figures only show present/absent.

Figures written to --out-dir:
  coverage_surfaces.png                pial + inflated; lateral, medial, ventral
  coverage_flatmap.png                 flattened cortex
  coverage_subcortical.png             non-HCP-MMP1 regions; lateral, medial, ventral
  coverage_flatmap_per_subject.png     one flatmap per subject
  coverage_subcortical_per_subject.png non-HCP-MMP1 regions per subject
  coverage_subject_by_region.csv       subject x region coverage table
"""

import argparse
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.freesurfer as fs
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

FSAVERAGE_FILES = {
    "lh.inflated": "https://raw.githubusercontent.com/mne-tools/mne-testing-data/master/subjects/fsaverage/surf/lh.inflated",
    "lh.cortex.patch.flat": "https://raw.githubusercontent.com/mne-tools/mne-testing-data/master/subjects/fsaverage/surf/lh.cortex.patch.flat",
    "lh.pial.surf.gii": "https://templateflow.s3.amazonaws.com/tpl-fsaverage/tpl-fsaverage_hemi-L_den-164k_pial.surf.gii",
    "lh.sulc.shape.gii": "https://templateflow.s3.amazonaws.com/tpl-fsaverage/tpl-fsaverage_hemi-L_den-164k_sulc.shape.gii",
    "lh.HCP-MMP1.annot": "https://raw.githubusercontent.com/poldrack/GOBS/master/extract/HCP-MMP1/lh.HCP-MMP1.annot",
    "aseg.mgz": "https://raw.githubusercontent.com/mne-tools/mne-testing-data/master/subjects/fsaverage/mri/aseg.mgz",
}

# Region names in the feature table that are not HCP-MMP1 parcels, mapped to
# left-hemisphere labels in FreeSurfer's aseg.
ASEG_LABELS = {
    "Thalamus": 10, "Caudate": 11, "Putamen": 12, "Pallidum": 13,
    "Hippocampus": 17, "Amygdala": 18, "Accumbens": 26, "VentralDC": 28,
}

# Sequential blue ramp, light -> dark.
BLUES = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
GYRUS, SULCUS = np.array(to_rgb("#dcdcda")), np.array(to_rgb("#a9a9a6"))
INK, INK_MUTED = "#1a1a19", "#5f5e5a"

# Camera direction (from camera towards the brain) and screen "up" per view,
# for a left hemisphere in fsaverage RAS coordinates.
VIEWS = {
    "Lateral": (np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
    "Medial": (np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])),
    "Ventral": (np.array([0, 0, 1.0]), np.array([-1.0, 0, 0])),  # anterior left
}


def fetch_fsaverage(cache: Path) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in FSAVERAGE_FILES.items():
        path = cache / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def read_patch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a FreeSurfer binary patch; returns vertex indices and 2-D coords."""
    with open(path, "rb") as f:
        if np.fromfile(f, ">i4", 1)[0] != -1:
            raise ValueError(f"{path} is not a new-format FreeSurfer patch")
        n = np.fromfile(f, ">i4", 1)[0]
        rec = np.fromfile(f, np.dtype([("vno", ">i4"), ("xyz", ">f4", 3)]), n)
    return np.abs(rec["vno"]) - 1, rec["xyz"][:, :2].astype(float)


def load_coverage(pkl: Path) -> pd.DataFrame:
    """Subject x region boolean table: does the subject have data in the region?"""
    X = pd.read_pickle(pkl)["features"]
    X = X[[c for c in X.columns if " | " in c]]
    present = X.notna().groupby(level="subject").any()
    region = present.columns.str.split(" | ", regex=False).str[0]
    return present.T.groupby(region, sort=False).any().T


def orient_flatmap(xy: np.ndarray, parcel: np.ndarray) -> np.ndarray:
    """Rotate the patch so frontal pole is left, V1 right and the medial wall up."""
    c = {r: np.median(xy[parcel == r], 0) for r in ("10pp", "V1", "MI", "p24")}
    axis = c["V1"] - c["10pp"]
    theta = -np.arctan2(axis[1], axis[0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy = (xy - xy.mean(0)) @ rot.T
    if ((c["p24"] - c["MI"]) @ rot.T)[1] < 0:  # cingulate should sit above insula
        xy[:, 1] *= -1
    return xy


def parcel_border(labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = labels[faces]
    edge = (tri[:, 0] != tri[:, 1]) | (tri[:, 1] != tri[:, 2])
    border = np.zeros(len(labels), bool)
    border[faces[edge].ravel()] = True
    return border


def vertex_colors(values: np.ndarray, sulc: np.ndarray, border: np.ndarray,
                  cmap, norm) -> np.ndarray:
    """Covered parcels in the ramp, the rest as curvature-shaded gray."""
    rgb = np.where((sulc > 0)[:, None], SULCUS, GYRUS)
    covered = values > 0
    rgb[covered] = cmap(norm(values[covered]))[:, :3] * np.where(sulc[covered] > 0, 0.88, 1.0)[:, None]
    rgb[border & covered] = rgb[border & covered] * 0.55 + 0.45
    return rgb


def draw_surface(ax, verts, faces, rgb, view, set_limits=True):
    forward, up = VIEWS[view]
    right = np.cross(forward, up)
    xy = np.c_[verts @ right, verts @ up]
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    facing = normals @ -forward
    keep = facing > 0                       # back-face culling
    order = np.argsort(-(tri[keep].mean(1) @ forward))  # far to near
    idx = np.flatnonzero(keep)[order]
    shade = 0.45 + 0.55 * facing[idx] ** 0.8
    colors = np.clip(rgb[faces[idx]].mean(1) * shade[:, None], 0, 1)
    ax.add_collection(PolyCollection(xy[faces[idx]], facecolors=colors,
                                     edgecolors="none", antialiased=False))
    if set_limits:
        ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
        ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
        ax.set_aspect("equal")
        ax.axis("off")
    return xy


def aseg_meshes(aseg_path: Path, structures) -> dict:
    """Smoothed surface meshes (fsaverage surface RAS) of aseg structures."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    img = nib.load(aseg_path)
    seg = np.asarray(img.dataobj)
    tkr = img.header.get_vox2ras_tkr()
    meshes = {}
    for name in structures:
        mask = gaussian_filter((seg == ASEG_LABELS[name]).astype(float), 1.0)
        verts, faces, _, _ = marching_cubes(mask, 0.5)
        verts = nib.affines.apply_affine(tkr, verts)
        if np.linalg.det(tkr[:3, :3]) < 0:   # keep normals pointing outwards
            faces = faces[:, ::-1]
        meshes[name] = (verts, faces)
    return meshes


def draw_glass_brain(ax, pial_v, faces, ghost_rgb, meshes, colors, view, labels=None):
    """Faded pial surface with structure meshes drawn on top of it."""
    xy = draw_surface(ax, pial_v, faces, ghost_rgb, view)
    forward, up = VIEWS[view]
    right = np.cross(forward, up)
    # far structures first so nearer ones overlap them
    for name in sorted(meshes, key=lambda n: -(meshes[n][0].mean(0) @ forward)):
        verts, f = meshes[name]
        draw_surface(ax, verts, f, np.tile(colors[name], (len(verts), 1)), view, set_limits=False)
    if labels:
        # a row of labels below the brain, ordered left to right, with leader lines
        ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
        y_text = ymin - 0.1 * (ymax - ymin)
        names = sorted(meshes, key=lambda n: meshes[n][0].mean(0) @ right)
        x_text = np.linspace(*np.percentile(xy[:, 0], [8, 92]), len(names))
        for name, xt in zip(names, x_text):
            c = meshes[name][0].mean(0)
            ax.annotate(labels[name], (c @ right, c @ up), xytext=(xt, y_text), ha="center", va="top",
                        fontsize=8, color=INK, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.6, shrinkA=2, shrinkB=0))
        ax.set_ylim(y_text - 0.22 * (ymax - ymin), ymax)
    return xy


def draw_flat(ax, patch_xy, patch_faces, rgb_patch):
    ax.add_collection(PolyCollection(patch_xy[patch_faces],
                                     facecolors=rgb_patch[patch_faces].mean(1),
                                     edgecolors="none", antialiased=False))
    ax.set_xlim(patch_xy[:, 0].min(), patch_xy[:, 0].max())
    ax.set_ylim(patch_xy[:, 1].min(), patch_xy[:, 1].max())
    ax.set_aspect("equal")
    ax.axis("off")


def add_colorbar(fig, cax, cmap, norm, vmax):
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal")
    cb.set_ticks(np.arange(1, vmax + 1))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    cb.set_label("Subjects with a bipolar channel in the region", fontsize=9, color=INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "real_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = fetch_fsaverage(args.cache)
    pial_v, faces = [d.data for d in nib.load(paths["lh.pial.surf.gii"]).darrays]
    infl_v, infl_f = fs.read_geometry(paths["lh.inflated"])
    assert len(infl_v) == len(pial_v) and np.array_equal(np.sort(infl_f, 1)[:50], np.sort(faces, 1)[:50])
    sulc = nib.load(paths["lh.sulc.shape.gii"]).darrays[0].data
    labels, _, names = fs.read_annot(paths["lh.HCP-MMP1.annot"])
    parcel = np.array([n.decode()[2:-4] if n.decode() != "???" else "" for n in names])[labels]
    parcel[labels < 0] = ""
    border = parcel_border(labels, faces)

    cov = load_coverage(args.pkl)
    cov.astype(int).to_csv(args.out_dir / "coverage_subject_by_region.csv")
    n_subj = cov.sum()
    extra = [r for r in cov.columns if r not in set(parcel)]   # not in HCP-MMP1
    subcortical = [r for r in extra if r in ASEG_LABELS]
    if set(extra) - set(subcortical):
        print("regions not found in HCP-MMP1 or aseg:", sorted(set(extra) - set(subcortical)))

    # One fixed scale for every group figure: 1 .. number of subjects.
    vmax = len(cov)
    cmap = ListedColormap(BLUES[:1] + list(np.array(BLUES)[np.linspace(1, len(BLUES) - 1, vmax - 1).round().astype(int)]))
    norm = BoundaryNorm(np.arange(0.5, vmax + 1), cmap.N)
    values = pd.Series(parcel).map(n_subj).fillna(0).to_numpy()
    rgb = vertex_colors(values, sulc, border, cmap, norm)

    sub_txt = "Not in HCP-MMP1 (see coverage_subcortical.png): " + ", ".join(
        f"{r} {int(n_subj[r])}" for r in extra)
    footer = (f"{len(cov)} subjects, {int((n_subj.drop(extra) > 0).sum())} of 180 "
              f"HCP-MMP1 parcels covered. Hemispheres pooled, shown on the left fsaverage hemisphere.")

    # Surfaces: pial and inflated, three views each.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), facecolor="white",
                             gridspec_kw=dict(hspace=0.05, wspace=0.02, width_ratios=[1, 1, 1.25]))
    for row, (surf_name, verts) in enumerate([("Pial", pial_v), ("Inflated", infl_v)]):
        for col, view in enumerate(VIEWS):
            ax = axes[row, col]
            draw_surface(ax, verts, faces, rgb, view)
            ax.set_title(f"{surf_name}, {view.lower()}", fontsize=10, color=INK)
    fig.suptitle("Electrode coverage per HCP-MMP1 parcel (fsaverage)", fontsize=13, color=INK, x=0.125, ha="left")
    add_colorbar(fig, fig.add_axes([0.35, 0.09, 0.3, 0.018]), cmap, norm, vmax)
    fig.text(0.125, 0.0, footer + "\n" + sub_txt, fontsize=8, color=INK_MUTED)
    fig.savefig(args.out_dir / "coverage_surfaces.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Flatmap.
    vno, patch_xy = read_patch(paths["lh.cortex.patch.flat"])
    lut = np.full(len(pial_v), -1)
    lut[vno] = np.arange(len(vno))
    patch_faces = lut[faces]
    patch_faces = patch_faces[(patch_faces >= 0).all(1)]
    patch_xy = orient_flatmap(patch_xy, parcel[vno])

    fig = plt.figure(figsize=(10, 7.4), facecolor="white")
    ax = fig.add_axes([0.02, 0.17, 0.96, 0.75])
    draw_flat(ax, patch_xy, patch_faces, rgb[vno])
    fig.suptitle("Electrode coverage per HCP-MMP1 parcel, flatmap", fontsize=13, color=INK, x=0.02, ha="left")
    add_colorbar(fig, fig.add_axes([0.35, 0.1, 0.3, 0.02]), cmap, norm, vmax)
    fig.text(0.02, 0.0, footer + "\n" + sub_txt,
             fontsize=8, color=INK_MUTED)
    fig.savefig(args.out_dir / "coverage_flatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # One flatmap per subject.
    ncol = 4
    nrow = int(np.ceil(len(cov) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow), facecolor="white")
    subj_cmap = ListedColormap([BLUES[8]])
    subj_norm = BoundaryNorm([0.5, 1.5], 1)
    for ax, (subj, row) in zip(axes.ravel(), cov.iterrows()):
        vals = pd.Series(parcel[vno]).map(row.astype(int)).fillna(0).to_numpy()
        draw_flat(ax, patch_xy, patch_faces, vertex_colors(vals, sulc[vno], border[vno], subj_cmap, subj_norm))
        ax.set_title(f"{subj}: {int(row.drop(extra).sum())} parcels", fontsize=8, color=INK)
    for ax in axes.ravel()[len(cov):]:
        ax.axis("off")
    fig.suptitle("Parcels with a bipolar channel, per subject", fontsize=13, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out_dir / "coverage_flatmap_per_subject.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Regions outside HCP-MMP1, drawn inside a faded pial surface.
    meshes = aseg_meshes(paths["aseg.mgz"], subcortical)
    ghost = np.where((sulc > 0)[:, None], SULCUS, GYRUS) * 0.25 + 0.75
    absent = SULCUS
    group_colors = {r: np.array(cmap(norm(n_subj[r]))[:3]) if n_subj[r] > 0 else absent for r in subcortical}
    labels = {r: f"{r}\n{int(n_subj[r])} subjects" for r in subcortical}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), facecolor="white",
                             gridspec_kw=dict(wspace=0.02, width_ratios=[1, 1, 1.25]))
    for ax, view in zip(axes, VIEWS):
        draw_glass_brain(ax, pial_v, faces, ghost, meshes, group_colors, view, labels)
        ax.set_title(view, fontsize=10, color=INK)
    fig.suptitle("Electrode coverage outside HCP-MMP1 (fsaverage aseg)", fontsize=13, color=INK, x=0.125, ha="left")
    add_colorbar(fig, fig.add_axes([0.35, 0.06, 0.3, 0.03]), cmap, norm, vmax)
    fig.text(0.125, -0.06, f"{len(cov)} subjects. Same color scale as the cortical figures. "
             "Hemispheres pooled, shown in the left hemisphere; structures drawn through a faded pial surface.",
             fontsize=8, color=INK_MUTED)
    fig.savefig(args.out_dir / "coverage_subcortical.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 1.75 * nrow), facecolor="white")
    present = np.array(to_rgb(BLUES[8]))
    for ax, (subj, row) in zip(axes.ravel(), cov.iterrows()):
        colors = {r: present if row[r] else absent for r in subcortical}
        draw_glass_brain(ax, pial_v, faces, ghost, meshes, colors, "Ventral")
        have = [r for r in subcortical if row[r]]
        ax.set_title(f"{subj}: {', '.join(have) if have else 'none'}", fontsize=8, color=INK)
    for ax in axes.ravel()[len(cov):]:
        ax.axis("off")
    fig.suptitle("Regions outside HCP-MMP1 with a bipolar channel, per subject (ventral view)",
                 fontsize=13, color=INK, x=0.02, ha="left")
    fig.text(0.02, -0.02, "Blue: present, gray: absent. Ventral view, anterior to the left: putamen is the "
             "anterior-lateral shape, amygdala the middle one and hippocampus the long posterior one "
             "(labelled in coverage_subcortical.png).", fontsize=8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out_dir / "coverage_subcortical_per_subject.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures to", args.out_dir)


if __name__ == "__main__":
    main()
