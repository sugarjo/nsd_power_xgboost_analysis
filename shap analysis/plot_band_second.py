"""Panel B of the main figure on its own: signed SHAP by band and second, all regions, correct and error trials.

Same computation as plot_new_vs_repeated.py panel B (per trial summed over all
of the subject's features in a band/second, missing features left out;
averaged within subject with the contrast's balanced trial-type weights, then
across subjects). One figure per trial set; both share one y range (the
extremes of either, plus a margin) and the colour scale --color-max (e.g. the
main figure's flatmap scale).

--drop-recent-errors (new vs repeated only): recent error trials are left out
(too few), so error trials are weighted new 1/2, consolidated 1/2; correct
trials keep 1/2, 1/4, 1/4. The y range is still shared with the correct trials.

Written to --out-dir: <contrast>_band_second_[errors_]all.{svg,pdf,png};
with --drop-recent-errors: <contrast>_band_second_errors_norecent_all and
<contrast>_band_second_all_sharedy (correct trials on the shared y range)
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
    ap.add_argument("--trials", nargs="+", choices=["test", "error"], default=["test", "error"],
                    help="trial sets to plot (the y range is shared between the sets given)")
    ap.add_argument("--axes-height", type=float, default=3.2,
                    help="height of the plot area in inches (3.2 = default; 1.6 halves it)")
    ap.add_argument("--suffix", default="", help="appended to the output file names")
    ap.add_argument("--drop-recent-errors", action="store_true",
                    help="leave recent trials out of the error set (new and consolidated weighted 1/2, 1/2)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    data = pd.read_pickle(args.pkl)
    C = CONTRASTS[args.contrast]
    if args.drop_recent_errors and args.contrast != "new_vs_repeated":
        raise SystemExit("--drop-recent-errors is for --contrast new_vs_repeated")
    error_weights = {"new": 0.5, "old": 0.5} if args.drop_recent_errors else C["weights"]
    subj = {}
    for trials in args.trials:
        out = signed_shap(data, args.contrast, trials)
        w = error_weights if trials == "error" else C["weights"]
        subj[trials] = subject_means(out["signed"], out["missing"], ["band", "second"], weights=w)
        print(f"{trials}: {len(subj[trials].dropna(how='all'))} subjects; weights {w}")
    lo = min(min(np.nanmin(s.to_numpy(float)), 0) for s in subj.values())
    hi = max(np.nanmax(s.to_numpy(float)) for s in subj.values())
    pad = 0.06 * (hi - lo)
    norm = TwoSlopeNorm(0, -args.color_max, args.color_max)

    for trials, s in subj.items():
        H = args.axes_height + 1.1                     # inches: 0.5 above the axes, 0.6 below for the labels
        fig, ax = plt.subplots(figsize=(9, H), facecolor="white")
        fig.subplots_adjust(top=1 - 0.5 / H, bottom=0.6 / H)
        band_second_panel(ax, s, pd.DataFrame({"mean": s.mean()}), norm, DIVERGING, "Net signed SHAP (log-odds)",
                          None, whiskers=False)
        ax.set_ylim(lo - pad, hi + pad)
        kind = "error trials" if trials == "error" else "correct trials"
        if trials == "error" and args.drop_recent_errors:
            kind += " (recent left out)"
        ax.set_title(f"All regions: {C['title']}, {kind}  ({len(s)} subjects)", fontsize=11, color=INK, loc="left")
        fig.text(0.07, -0.25 / H,
                 f"{C['formula']} (sign flipped on {C['flipped']} trials; error trials: sign set by the true category); "
                 "\nper trial summed over all of the subject's features in each band/second, missing features left out; "
                 f"averaged within subject\n("
                 + ("new and consolidated weighted 1/2, 1/2; recent error trials left out (too few)"
                    if trials == "error" and args.drop_recent_errors else C["balance"])
                 + "), then across subjects. Bars = mean, dots = subjects. "
                 f"Colour scale ±{args.color_max:.3f} (main figure's flatmap)"
                 + ("; same y axis for correct and error trials." if len(args.trials) > 1 else "."),
                 fontsize=7.5, color=INK_MUTED, va="top")
        stem = args.out_dir / f"{C['prefix']}_band_second_{'errors_' if trials == 'error' else ''}all"
        if args.drop_recent_errors:
            stem = args.out_dir / (f"{C['prefix']}_band_second_errors_norecent_all" if trials == "error"
                                   else f"{C['prefix']}_band_second_all_sharedy")
        stem = stem.with_name(stem.name + args.suffix)
        for ext in ("svg", "pdf", "png"):
            fig.savefig(stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("wrote", stem.with_suffix(".svg"), "| bar range", s.mean().min().round(4), s.mean().max().round(4))
    print(f"y range {lo - pad:.3f} .. {hi + pad:.3f}")


if __name__ == "__main__":
    main()
