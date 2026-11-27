"""Trial-level distribution of signed SHAP per band and second for a set of parcels combined.

Per test-set trial: the signed new-vs-seen SHAP (signed_shap.py; > 0 helps the
correct new/seen call) summed over the features of the listed HCP-MMP1 parcels
(missing features left out; a trial counts only if at least one of the parcels
has data). One split violin per band and second: new trials (left) and seen
trials (recent + old, right). Black dots: each patient's balanced mean (new,
recent, old weighted 1/2, 1/4, 1/4, as in the summary figure); line: their mean.

Example:
  python3 plot_shap_distribution.py --parcels POS1 POS2 v23ab 31pd 31pv 31a d23ab

Written to --out-dir: shap_distribution_<parcels>.{svg,pdf,png,csv}
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import AXIS, BAND_LABELS, BANDS, GRID, INK, INK_MUTED  # noqa: E402
from signed_shap import CONTRASTS, signed_shap  # noqa: E402

COLORS = {"new": "#2a78d6", "seen": "#eb6834"}
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--parcels", nargs="+", required=True)
    ap.add_argument("--trials", choices=["test", "error"], default="test")
    ap.add_argument("--clip", type=float, default=99.0, help="y range: this percentile of |value| over all trials")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    out = signed_shap(pd.read_pickle(args.pkl), "new_vs_repeated", args.trials)
    values = out["signed"].where(~out["missing"].to_numpy())
    sel = values.columns.get_level_values("region").isin(args.parcels)
    unknown = set(args.parcels) - set(values.columns.get_level_values("region"))
    if unknown:
        raise SystemExit(f"unknown parcels: {sorted(unknown)}")
    v = values.loc[:, sel].T.groupby(level=["band", "second"]).sum(min_count=1).T      # trials x (band, second)
    v = v.dropna(how="all")
    cat = v.index.get_level_values("image_category").astype(int)
    subj = v.index.get_level_values("subject")
    kind = np.where(cat == 1, "new", "seen")

    # balanced patient means (new 1/2, recent 1/4, old 1/4)
    w = CONTRASTS["new_vs_repeated"]["weights"]
    names = {1: "new", 2: "recent", 3: "old"}
    per = v.groupby([subj, cat]).mean()
    patient = sum(w[names[c]] * per.xs(c, level=1) for c in (1, 2, 3))
    patient = patient.dropna(how="all")

    long = v.stack(["band", "second"], future_stack=True).rename("value").reset_index()
    long.to_csv(args.out_dir / f"shap_distribution_{'_'.join(re.sub(r'[^A-Za-z0-9]+', '', p) for p in args.parcels)}.csv",
                index=False)

    ticks = [(b, s) for b in BANDS for s in (1, 2)]
    lim = np.nanpercentile(np.abs(v.to_numpy(float)), args.clip)
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor="white")
    width, gap = 0.36, 0.04
    rng = np.random.default_rng(0)
    for i, band in enumerate(BANDS):
        for sec, off in ((1, -(width + gap) / 2), (2, (width + gap) / 2)):
            x = i + off
            for k, side in (("new", -1), ("seen", 1)):
                data = v.loc[kind == k, (band, sec)].dropna().to_numpy(float)
                data = data[np.abs(data) <= lim * 1.5]
                if len(data) < 5:
                    continue
                parts = ax.violinplot(data, positions=[x], widths=width, showextrema=False,
                                      points=200, bw_method=0.3)
                for body in parts["bodies"]:
                    verts = body.get_paths()[0].vertices
                    verts[:, 0] = np.where(side < 0, np.minimum(verts[:, 0], x), np.maximum(verts[:, 0], x))
                    body.set_facecolor(COLORS[k])
                    body.set_edgecolor("none")
                    body.set_alpha(0.55)
                med = np.median(data)
                ax.plot([x + side * 0.02, x + side * width / 2 * 0.9], [med, med], color=COLORS[k], lw=1)
            pm = patient[(band, sec)].dropna()
            ax.scatter(x + rng.uniform(-0.05, 0.05, len(pm)), pm, s=6, color=INK, zorder=4, linewidths=0)
            ax.plot([x - 0.12, x + 0.12], [pm.mean()] * 2, color=INK, lw=1.4, zorder=5)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-0.6, len(BANDS) - 0.4)
    xs = [i + o for i in range(len(BANDS)) for o in (-(width + gap) / 2, (width + gap) / 2)]
    ax.set_xticks(xs, [f"{s} s" for _, s in ticks], fontsize=6, color=INK_MUTED)
    for i, band in enumerate(BANDS):
        ax.text(i, -0.14, BAND_LABELS.get(band, band), transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=7.5, color=INK)
    ax.set_ylabel("Signed SHAP per trial (log-odds)", fontsize=7.5, color=INK)
    ax.tick_params(axis="y", labelsize=6.5, colors=INK_MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS["new"], alpha=0.55),
               plt.Rectangle((0, 0), 1, 1, color=COLORS["seen"], alpha=0.55),
               plt.Line2D([], [], marker="o", ls="", color=INK, ms=2.5),
               plt.Line2D([], [], color=INK, lw=1.4)]
    ax.legend(handles, [f"New trials (n = {(kind == 'new').sum()})", f"Seen trials (n = {(kind == 'seen').sum()})",
                        "Patient (balanced mean)", "Mean over patients"],
              fontsize=6, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    title = " + ".join(args.parcels)
    fig.suptitle(f"{title}: signed SHAP per trial" + (" (error trials)" if args.trials == "error" else ""),
                 fontsize=9, color=INK, x=0.01, ha="left", y=1.06)
    fig.text(0.01, -0.12,
             f"Test-set {'error' if args.trials == 'error' else 'correct'} trials of the {subj.nunique()} patients with a "
             "channel in any of these parcels. Signed new-vs-seen SHAP (> 0 helps the correct call), summed over the "
             "parcels' features, missing features left out.\nSplit violins: new (left) and seen (recent + old, right) "
             f"trials, short lines = medians; y range ±{lim:.3f} ({args.clip:g}th percentile of |value|). "
             "Dots: patients' balanced means (new, recent, old 1/2, 1/4, 1/4).",
             fontsize=6, color=INK_MUTED, va="top")
    stem = args.out_dir / f"shap_distribution_{'_'.join(re.sub(r'[^A-Za-z0-9]+', '', p) for p in args.parcels)}"
    for ext in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    print("wrote", stem.with_suffix(".svg"), f"({len(v)} trials, {subj.nunique()} patients)")
    summary = pd.DataFrame({"patient_mean": patient.mean(), "median_new": v[kind == "new"].median(),
                            "median_seen": v[kind == "seen"].median()})
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
