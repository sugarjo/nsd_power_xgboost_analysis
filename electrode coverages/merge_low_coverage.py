"""Merge neighbouring HCP-MMP1 parcels until every region has enough subjects.

A parcel covered by fewer than --min-subjects subjects is merged with a
neighbouring parcel (one that shares a border on the fsaverage surface). A
merged region counts every subject that has data in any of its parcels.
Merges are made one at a time, greedily, to keep as many separate regions as
possible:

  1. take the region still below the threshold with the fewest subjects;
  2. merge it with the neighbour that brings it to the threshold, preferring a
     neighbour that is itself below the threshold (one merge fixes both), and
     then the smallest resulting area;
  3. if no single neighbour is enough, merge with the neighbour that adds the
     most subjects (preferring one below the threshold) and repeat.

Parcels without any coverage are never merged (they add no subjects), and
regions outside HCP-MMP1 (amygdala, hippocampus, putamen) are left as they
are. Greedy merging is not guaranteed to be the global minimum, but each step
only merges where it is needed.

Written to --out-dir (suffix "_min<N>" when --min-subjects is not 3):
  merged_regions.csv            parcel -> merged region, with subject counts
  coverage_flatmap_merged.png   flatmap of the merged regions' coverage
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
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from plot_electrode_coverage import (BLUES, GYRUS, INK, INK_MUTED, SULCUS, add_colorbar,  # noqa: E402
                                     draw_flat, fetch_fsaverage, load_coverage, orient_flatmap,
                                     read_patch)


def parcel_adjacency(parcel: np.ndarray, faces: np.ndarray) -> dict:
    """Parcel -> set of parcels sharing an edge with it on the surface."""
    edges = np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    a, b = parcel[edges[:, 0]], parcel[edges[:, 1]]
    keep = (a != b) & (a != "") & (b != "")
    adj = {}
    for p, q in set(zip(a[keep], b[keep])):
        adj.setdefault(p, set()).add(q)
        adj.setdefault(q, set()).add(p)
    return adj


def merge_parcels(subjects: dict, adj: dict, area: dict, min_subjects: int):
    """Greedy merging; returns (parcel -> group id, group id -> member parcels, merge log)."""
    group = {p: p for p in subjects}                      # parcel -> group id
    members = {p: [p] for p in subjects}
    subj = {p: set(s) for p, s in subjects.items()}
    size = {p: area[p] for p in subjects}
    log = []

    def neighbours(g):
        return {group[q] for p in members[g] for q in adj.get(p, ()) if q in group} - {g}

    while True:
        low = [g for g in members if len(subj[g]) < min_subjects and neighbours(g)]
        if not low:
            break
        g = min(low, key=lambda g: (len(subj[g]), size[g]))
        def score(h):
            union = len(subj[g] | subj[h])
            return (union >= min_subjects, len(subj[h]) < min_subjects,
                    union if union < min_subjects else 0, -(size[g] + size[h]))
        h = max(neighbours(g), key=score)
        log.append((" + ".join(members[g]), " + ".join(members[h]), len(subj[g] | subj[h])))
        for p in members[h]:
            group[p] = g
        members[g] += members.pop(h)
        subj[g] |= subj.pop(h)
        size[g] += size.pop(h)
    return group, members, subj, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--min-subjects", type=int, default=3)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = fetch_fsaverage(args.cache)
    pial_v, faces = [d.data for d in nib.load(paths["lh.pial.surf.gii"]).darrays]
    sulc = nib.load(paths["lh.sulc.shape.gii"]).darrays[0].data
    labels, _, names = fs.read_annot(paths["lh.HCP-MMP1.annot"])
    parcel = np.array([n.decode()[2:-4] if n.decode() != "???" else "" for n in names])[labels]
    parcel[labels < 0] = ""

    cov = load_coverage(args.pkl)                          # subject x region, bool
    in_atlas = [r for r in cov.columns if r in set(parcel)]
    subjects = {r: set(cov.index[cov[r]]) for r in in_atlas if cov[r].any()}
    area = pd.Series(parcel).value_counts().to_dict()     # vertices per parcel
    adj = parcel_adjacency(parcel, faces)

    group, members, subj, log = merge_parcels(subjects, adj, area, args.min_subjects)
    name = {g: "+".join(m) for g, m in members.items()}

    rows = [dict(parcel=p, region=name[group[p]], n_subjects_parcel=len(subjects[p]),
                 n_subjects_region=len(subj[group[p]]), n_parcels_in_region=len(members[group[p]]))
            for p in subjects]
    rows += [dict(parcel=r, region=r, n_subjects_parcel=int(cov[r].sum()), n_subjects_region=int(cov[r].sum()),
                  n_parcels_in_region=1) for r in cov.columns if r not in set(parcel)]
    table = pd.DataFrame(rows).sort_values(["region", "parcel"])
    suffix = "" if args.min_subjects == 3 else f"_min{args.min_subjects}"
    table.to_csv(args.out_dir / f"merged_regions{suffix}.csv", index=False)

    n_low_before = sum(len(s) < args.min_subjects for s in subjects.values())
    still_low = [name[g] for g in members if len(subj[g]) < args.min_subjects]
    print(f"{len(subjects)} covered HCP-MMP1 parcels, {n_low_before} below {args.min_subjects} subjects")
    print(f"{len(log)} merges -> {len(members)} regions; still below {args.min_subjects}: {still_low or 'none'}")
    for a, b, n in log:
        print(f"  {a}  +  {b}  -> {n} subjects")

    # Flatmap: colour by the merged region's coverage, borders only between merged regions.
    vno, patch_xy = read_patch(paths["lh.cortex.patch.flat"])
    lut = np.full(len(pial_v), -1)
    lut[vno] = np.arange(len(vno))
    patch_faces = lut[faces]
    patch_faces = patch_faces[(patch_faces >= 0).all(1)]
    patch_xy = orient_flatmap(patch_xy, parcel[vno])

    region_of = np.array([name[group[p]] if p in group else "" for p in parcel])
    n_region = np.array([len(subj[group[p]]) if p in group else 0 for p in parcel])
    tri = region_of[faces]
    edge = (tri[:, 0] != tri[:, 1]) | (tri[:, 1] != tri[:, 2])
    border = np.zeros(len(parcel), bool)
    border[faces[edge].ravel()] = True

    vmax = len(cov)
    cmap = ListedColormap(BLUES[:1] + list(np.array(BLUES)[np.linspace(1, len(BLUES) - 1, vmax - 1).round().astype(int)]))
    norm = BoundaryNorm(np.arange(0.5, vmax + 1), cmap.N)
    rgb = np.where((sulc > 0)[:, None], SULCUS, GYRUS)
    has = n_region > 0
    rgb[has] = cmap(norm(n_region[has]))[:, :3] * np.where(sulc[has] > 0, 0.88, 1.0)[:, None]
    rgb[border & has] = rgb[border & has] * 0.4 + 0.6 * np.array(to_rgb(INK)) * 0.5

    merged = {g for g, m in members.items() if len(m) > 1}
    # Label each merged region at its largest parcel: by name, or, when there are
    # too many to read, by a number listed next to the map (numbered left to right).
    anchor = {}
    for g in merged:
        p = max(members[g], key=lambda p: area[p])
        anchor[g] = np.median(patch_xy[parcel[vno] == p], 0)
    numbered = len(merged) > 30
    order = sorted(merged, key=lambda g: (anchor[g][0], -anchor[g][1]))
    fig = plt.figure(figsize=(14 if numbered else 10, 7.6), facecolor="white")
    ax = fig.add_axes([0.02, 0.17, 0.66 if numbered else 0.96, 0.75])
    draw_flat(ax, patch_xy, patch_faces, rgb[vno])
    for i, g in enumerate(order, 1):
        ax.annotate(str(i) if numbered else "+".join(members[g]), anchor[g], fontsize=6 if numbered else 5.5,
                    color=INK, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    if numbered:
        key = fig.add_axes([0.69, 0.06, 0.3, 0.88])
        key.axis("off")
        per_col = int(np.ceil(len(order) / 2))
        for i, g in enumerate(order):
            key.text(0.5 * (i // per_col), 1 - (i % per_col) / per_col, f"{i + 1:>2}  " + "+".join(members[g]),
                     fontsize=5.5, color=INK, va="top", transform=key.transAxes)
    fig.suptitle(f"Electrode coverage after merging parcels with fewer than {args.min_subjects} subjects",
                 fontsize=13, color=INK, x=0.02, ha="left")
    add_colorbar(fig, fig.add_axes([0.25 if numbered else 0.35, 0.1, 0.3 if not numbered else 0.22, 0.02]), cmap, norm, vmax)
    fig.text(0.02, 0.0,
             f"{len(subjects)} covered parcels, {n_low_before} below {args.min_subjects} subjects; "
             f"{len(log)} merges give {len(members)} regions ({len(merged)} of them merged, {'numbered' if numbered else 'labelled'}). "
             f"Dark lines: borders between regions after merging.\n"
             f"A merged region counts every subject with data in any of its parcels. Gray: no coverage. "
             f"Hemispheres pooled, shown on the left fsaverage hemisphere.",
             fontsize=8, color=INK_MUTED)
    fig.savefig(args.out_dir / f"coverage_flatmap_merged{suffix}.png", dpi=200, bbox_inches="tight")
    print("wrote", args.out_dir / f"merged_regions{suffix}.csv", f"and coverage_flatmap_merged{suffix}.png")


if __name__ == "__main__":
    main()
