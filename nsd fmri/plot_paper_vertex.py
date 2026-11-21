"""Allen et al. (2022) Fig. 4b recomputed exactly, plus consolidated vs recent hits.

Follows cvnlab/nsddatapaper mainfigures/SCIENCE.MEMORY/SCIENCEMEMORYnotes.m:
  - betas_fithrf_GLMdenoise_RR on fsaverage, raw percent signal change (no
    normalisation), every vertex of both hemispheres;
  - per participant, t = (m1 - m2) / sqrt(se1^2 + se2^2) (Welch), for single
    sessions (5, 10, 15, 20), for trials pooled across all sessions and for
    the pooled trials with shuffled labels; NaN t set to 0;
  - group map = simple mean of the participants' t maps;
  - shown on the flattened left hemisphere, thresholded at |t| > 3, colour
    range +-8 for single sessions and +-40 for pooled.
Contrasts: hits > correct rejections (the paper) and consolidated hits >
recent hits (recent: last shown <= 3 h before, consolidated: > 3 h, the
boundary set in extract_vertex_stats.py). The shuffled control is pooled over
labels shuffled within each session (the paper shuffles across the pooled
trials; with this many trials both give |t| near 0).

Inputs from extract_vertex_stats.py. Written to --out-dir:
  nsd_paper_vertex.png
  nsd_paper_vertex_<contrast>_parcels.csv  per parcel (left + right averaged):
      mean over vertices of each participant's pooled t, and of the group mean
  nsd_paper_psc_<contrast>.csv             parcel-level version: Welch t on the
      raw parcel-mean betas, per participant and averaged
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgb

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "shap analysis"))
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import (DIVERGING, GYRUS, INK, INK_MUTED, NO_DATA_GYRUS, NO_DATA_SULCUS,  # noqa: E402
                                  SULCUS, draw_flat, load_flatmap)
from extract_vertex_stats import GROUPS, trial_types  # noqa: E402
from extract_parcel_betas import parcel_index  # noqa: E402

SUBJECTS = range(1, 9)
TITLES = {"pooled": "Pooled (paper)", "demeaned": "Pooled, session means removed",
          "shuffled": "Pooled, shuffled", "shuffled_demeaned": "Shuffled, session means removed"}
SESSIONS = [5, 10, 15, 20]
CONTRASTS = {  # name: (title, positive groups, negative groups, shuffled positive, shuffled negative)
    "hits_vs_cr": ("Hits > correct rejections", ["recent", "consolidated"], ["cr"], ["perm_a"], ["perm_b"]),
    "consolidated_vs_recent": ("Consolidated hits > recent hits", ["consolidated"], ["recent"],
                               ["perm_d"], ["perm_c"]),
}
G = {g: i for i, g in enumerate(GROUPS)}


def welch_t(n, s, ss, pos, neg):
    """Welch t from summed n / sum / sumsq (sessions already summed); NaN -> 0 as in the paper."""
    def stats(groups):
        k = sum(n[G[g]] for g in groups)
        tot = sum(s[G[g]] for g in groups)
        sq = sum(ss[G[g]] for g in groups)
        mean = tot / k
        var = (sq - tot ** 2 / k) / (k - 1)
        return mean, var / k
    with np.errstate(invalid="ignore", divide="ignore"):
        (m1, v1), (m2, v2) = stats(pos), stats(neg)
        t = (m1 - m2) / np.sqrt(v1 + v2)
    return np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def demean_sessions(n, s, ss):
    """Remove each session's mean (over its correct trials) from the per-session sums.

    With mu the session mean, sum' = sum - n*mu and sumsq' = sumsq - 2*mu*sum + n*mu^2.
    Removes slow signal drifts between sessions, which otherwise leak into
    pooled contrasts whose trial mix changes over sessions (hits are rare
    in early sessions)."""
    real = [G[g] for g in ("cr", "recent", "consolidated")]
    mu = s[:, real].sum(1) / n[:, real].sum(1)[:, None]             # sessions x vertices
    s2 = s - n[:, :, None] * mu[:, None]
    ss2 = ss - 2 * mu[:, None] * s + n[:, :, None] * mu[:, None] ** 2
    return s2, ss2


def load_subject(data_dir: Path, subject: int, hemi: str, window: str = ""):
    """Per-session n (sessions x groups), sum and sumsq (sessions x groups x vertices), float64."""
    files = sorted((data_dir / f"subj{subject:02d}").glob(f"session*_{hemi}_stats{window}.npz"))
    sessions = [int(f.name[7:9]) for f in files]
    d = [np.load(f) for f in files]
    return (sessions, np.stack([x["n"] for x in d]).astype(float),
            np.stack([x["sum"] for x in d]).astype(float), np.stack([x["sumsq"] for x in d]).astype(float))


def parcel_psc_t(data_dir: Path, subject: int, hours: float, max_lag_days: float | None = None) -> dict:
    """Welch t per parcel on the raw parcel-mean betas (hemispheres averaged), pooled sessions."""
    beh = trial_types(data_dir / f"subj{subject:02d}_responses.tsv", hours, max_lag_days)
    parts = []
    for session, trials in beh.groupby("SESSION"):
        lh, rh = (np.load(data_dir / f"subj{subject:02d}" / f"session{session:02d}_{h}_psc.npy") for h in ("lh", "rh"))
        parts.append(pd.DataFrame(np.mean([lh, rh], 0)).assign(group=trials["group"].to_numpy()))
    df = pd.concat(parts)
    x = df.drop(columns="group")
    out = {}
    for name, (_, pos, neg, _, _) in CONTRASTS.items():
        a, b = x[df.group.isin(pos)], x[df.group.isin(neg)]
        out[name] = ((a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))).to_numpy()
    return out


def draw_t(ax, geo, t_lh, rng, threshold, border):
    rgb = np.where((geo.sulc > 0)[:, None], NO_DATA_SULCUS, NO_DATA_GYRUS)
    medial = geo.parcel == ""
    rgb[medial] = np.where((geo.sulc[medial] > 0)[:, None], SULCUS, GYRUS)
    show = np.abs(t_lh) > threshold
    rgb[show] = DIVERGING(Normalize(-rng, rng)(t_lh[show]))[:, :3]
    rgb[border] = rgb[border] * 0.88 + 0.12 * np.array(to_rgb(INK))
    draw_flat(ax, geo.patch_xy, geo.patch_faces, rgb[geo.vno])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO / "data" / "nsd")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--cache", type=Path, default=REPO / "data" / "fsaverage")
    ap.add_argument("--hours", type=float, default=3.0, help="must match extract_vertex_stats.py")
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--max-lag-days", type=float, help="use the stats from extract_vertex_stats.py "
                    "--max-lag-days (repeats last shown longer ago left out)")
    args = ap.parse_args()
    window = f"_maxlag{args.max_lag_days:g}d" if args.max_lag_days is not None else ""

    # t maps: contrast -> column ("session05" ... "pooled", "shuffled") -> hemi -> subjects x vertices
    columns = [f"session{s:02d}" for s in SESSIONS] + ["pooled", "demeaned", "shuffled", "shuffled_demeaned"]
    tmaps = {c: {col: {"lh": [], "rh": []} for col in columns} for c in CONTRASTS}
    for subject in SUBJECTS:
        for hemi in ("lh", "rh"):
            sessions, n, s, ss = load_subject(args.data_dir, subject, hemi, window)
            sd, ssd = demean_sessions(n, s, ss)
            print(f"subj{subject:02d} {hemi}: {len(sessions)} sessions, correct trials "
                  + ", ".join(f"{g} {int(n[:, G[g]].sum())}" for g in GROUPS[:3]), flush=True)
            for name, (_, pos, neg, ppos, pneg) in CONTRASTS.items():
                for col in columns:
                    if col == "shuffled_demeaned":
                        t = welch_t(n.sum(0), sd.sum(0), ssd.sum(0), ppos, pneg)
                    elif col == "demeaned":
                        t = welch_t(n.sum(0), sd.sum(0), ssd.sum(0), pos, neg)
                    elif col == "shuffled":
                        t = welch_t(n.sum(0), s.sum(0), ss.sum(0), ppos, pneg)
                    elif col == "pooled":
                        t = welch_t(n.sum(0), s.sum(0), ss.sum(0), pos, neg)
                    else:
                        i = sessions.index(int(col[7:]))
                        t = welch_t(n[i], s[i], ss[i], pos, neg)
                    tmaps[name][col][hemi].append(t)

    # parcel summaries
    parcels = (args.data_dir / "parcels.txt").read_text().split()
    columns_idx = {h: parcel_index(args.data_dir, h)[0] for h in ("lh", "rh")}
    psc = {s: parcel_psc_t(args.data_dir, s, args.hours, args.max_lag_days) for s in SUBJECTS}
    for name in CONTRASTS:
        for variant, suffix in [("pooled", ""), ("demeaned", "_demeaned")]:
            table = {}
            for k, subject in enumerate(SUBJECTS):
                per_hemi = [pd.Series(tmaps[name][variant][h][k]).groupby(columns_idx[h]).mean() for h in ("lh", "rh")]
                table[f"subj{subject:02d}"] = (per_hemi[0] + per_hemi[1]).drop(-1, errors="ignore").to_numpy() / 2
            t = pd.DataFrame(table, index=parcels)
            t.insert(0, "mean_t", t.mean(axis=1))
            t.insert(1, "n_positive", (t.iloc[:, 1:] > 0).sum(axis=1))
            t.index.name = "parcel"
            t.to_csv(args.out_dir / f"nsd_paper_vertex_{name}{suffix}{window}_parcels.csv")
            print(f"\n{CONTRASTS[name][0]}, {TITLES[variant]} (mean over vertices of t, group mean):")
            top = t.sort_values("mean_t")[["mean_t", "n_positive"]]
            print(pd.concat([top.head(6), top.tail(6)]).round(2).to_string())
        p = pd.DataFrame({f"subj{s:02d}": psc[s][name] for s in SUBJECTS}, index=parcels)
        p.insert(0, "mean_t", p.mean(axis=1))
        p.insert(1, "n_positive", (p.iloc[:, 1:] > 0).sum(axis=1))
        p.index.name = "parcel"
        p.to_csv(args.out_dir / f"nsd_paper_psc_{name}{window}.csv")

    # every participant's t maps (both hemispheres), for later use; the figure shows the group mean only
    tpath = args.data_dir / f"nsd_paper_vertex{window}_tmaps.npz"
    np.savez_compressed(tpath, subjects=np.array([f"subj{s:02d}" for s in SUBJECTS]),
                        **{f"{name}__{col}__{h}": np.array(tmaps[name][col][h], dtype=np.float32)
                           for name in CONTRASTS for col in columns for h in ("lh", "rh")})
    print("wrote", tpath)

    # figure: per contrast, the group mean
    geo = load_flatmap(args.cache)
    border = geo.border
    rows = [(name, "mean of 8 participants") for name in CONTRASTS]
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(3.3 * len(columns), 3.4 * len(rows) + 2.2),
                             facecolor="white", gridspec_kw=dict(wspace=0.02, hspace=0.25, left=0.07, right=0.99,
                                                                 top=0.86, bottom=0.24))
    for r, (name, who) in enumerate(rows):
        for c, col in enumerate(columns):
            stack = np.array(tmaps[name][col]["lh"])
            t_lh = stack.mean(0)
            rng = 8 if col.startswith("session") else 40
            draw_t(axes[r, c], geo, t_lh, rng, args.threshold, border)
            if True:
                axes[r, c].set_title(col.replace("session", "Session ") if col.startswith("session")
                                     else TITLES[col],
                                     fontsize=10, color=INK)
        axes[r, 0].text(-0.06, 0.5, f"{CONTRASTS[name][0]}\n{who}", transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=9, color=INK)
    n_single = len(SESSIONS)
    for cols, rng in [(range(n_single), 8), (range(n_single, len(columns)), 40)]:
        left, right = axes[-1, cols[0]].get_position(), axes[-1, cols[-1]].get_position()
        mid = (left.x0 + right.x1) / 2
        cax = fig.add_axes([mid - 0.07, left.y0 - 0.06, 0.14, 0.018])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-rng, rng), cmap=DIVERGING), cax=cax,
                          orientation="horizontal", extend="both")
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, colors=INK_MUTED, length=0)
        cb.set_label("Welch t", fontsize=8, color=INK)
    fig.suptitle("NSD fMRI recognition memory, method of Allen et al. (2022) Fig. 4b", fontsize=14, color=INK,
                 x=0.02, ha="left")
    fig.text(0.02, 0.07,
             f"betas_fithrf_GLMdenoise_RR on fsaverage in percent signal change (no normalisation), correct trials. "
             f"Per participant and vertex: Welch t between trial types, for single sessions and pooled across "
             f"sessions; group = simple mean of the 8 t maps.\nShown where |t| > {args.threshold:g} (gray otherwise); "
             f"flattened left hemisphere with HCP-MMP1 borders. Hits = recent + consolidated; recent: last shown "
             f"≤ {args.hours:g} h before, consolidated: > {args.hours:g} h. Shuffled: labels shuffled within session (keeps each session's trial mix, so it shows what session-to-session drift alone produces).\n"
             f"Session means removed: each session's mean over its correct trials subtracted per vertex before pooling."
             + (f" Only repeats last shown ≤ {args.max_lag_days:g} days before (the sEEG lag range)."
                if args.max_lag_days is not None else ""),
             fontsize=8, color=INK_MUTED, va="top")
    path = args.out_dir / f"nsd_paper_vertex{window}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
