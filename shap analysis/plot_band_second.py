"""Panel B of the main figure on its own: signed SHAP by band and second, all regions, correct and error trials.

Same computation as plot_new_vs_repeated.py panel B (per trial summed over all
of the subject's features in a band/second, missing features left out;
averaged within subject with the contrast's balanced trial-type weights, then
across subjects). One figure per trial set; both share one y range (the
extremes of either, plus a margin) and the colour scale --color-max (e.g. the
main figure's flatmap scale).

Written to --out-dir: <contrast>_band_second_[errors_]all.{svg,pdf,png}
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from plot_new_vs_repeated import DIVERGING, INK, INK_MUTED, band_second_panel, subject_means  # noqa: E402
from signed_shap import CONTRASTS, signed_shap  # noqa: E402

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--contrast", choices=list(CONTRASTS), default="new_vs_repeated")
    ap.add_argument("--color-max", type=float, required=True, help="colour scale (+-), e.g. the main flatmap's")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    data = pd.read_pickle(args.pkl)
    C = CONTRASTS[args.contrast]
    subj = {}
    for trials in ("test", "error"):
        out = signed_shap(data, args.contrast, trials)
        subj[trials] = subject_means(out["signed"], out["missing"], ["band", "second"], weights=C["weights"])
    lo = min(min(np.nanmin(s.to_numpy(float)), 0) for s in subj.values())
    hi = max(np.nanmax(s.to_numpy(float)) for s in subj.values())
    pad = 0.06 * (hi - lo)
    norm = TwoSlopeNorm(0, -args.color_max, args.color_max)

    for trials, s in subj.items():
        fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="white")
        band_second_panel(ax, s, pd.DataFrame({"mean": s.mean()}), norm, DIVERGING, "Net signed SHAP (log-odds)",
                          None, whiskers=False)
        ax.set_ylim(lo - pad, hi + pad)
        kind = "error trials" if trials == "error" else "correct trials"
        ax.set_title(f"All regions: {C['title']}, {kind}  ({len(s)} subjects)", fontsize=11, color=INK, loc="left")
        fig.text(0.07, -0.06,
                 f"{C['formula']} (sign flipped on {C['flipped']} trials; error trials: sign set by the true category); "
                 "\nper trial summed over all of the subject's features in each band/second, missing features left out; "
                 f"averaged within subject\n({C['balance']}), then across subjects. Bars = mean, dots = subjects. "
                 f"Colour scale ±{args.color_max:.3f} (main figure's flatmap); same y axis for correct and error trials.",
                 fontsize=7.5, color=INK_MUTED, va="top")
        stem = args.out_dir / f"{C['prefix']}_band_second_{'errors_' if trials == 'error' else ''}all"
        for ext in ("svg", "pdf", "png"):
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("wrote", stem.with_suffix(".svg"), "| bar range", s.mean().min().round(4), s.mean().max().round(4))
    print(f"y range {lo - pad:.3f} .. {hi + pad:.3f}")


if __name__ == "__main__":
    main()
