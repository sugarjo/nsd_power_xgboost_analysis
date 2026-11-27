"""Many feature-distribution bar plots from one load of the data file.

Edit FIGURES below (or pass --specs a JSON file with the same list) and run.
Each entry is [area, band, seconds]:

  area     a merged-region name from the merge table ("31pd+31pv+31a+d23ab"),
           a single parcel ("TE2p"), parcels joined with "+" ("FFC+VVC+PH+PHA3"),
           or a list of any of these (["PHA3", "FFC+VVC+PH", "TE2p"]);
           merged-region names are expanded to their parcels
  band     Delta, Theta, Alpha, Beta, GammaL, GammaH or Ripple
  seconds  1, 2 or [1, 2] (one panel per second)

Each figure is made as in plot_feature_distribution.py (per trial the band
power averaged over the area's parcels the patient has; violins = trials,
bars = mean of patient means +- s.e.m., dots = patients), and written to
--out-dir as feature_distribution_<band>_<area>[_s<seconds>].{svg,pdf,png,csv}.

Examples:
  python plot_feature_distribution_batch.py --pkl real_data_bipolar_xgboost_3w.pkl
  python plot_feature_distribution_batch.py --pkl real.pkl --specs my_figures.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from plot_feature_distribution import load_features, make_plot  # noqa: E402

# ---- edit this list -----------------------------------------------------------
FIGURES = [
    ["31pd+31pv+31a+d23ab", "GammaH", 1],
    ["POS1+v23ab+POS2", "GammaH", 1],
    [["PHA3", "FFC+VVC+PH", "TE2p"], "Delta", 2],
    ["8C+PEF+p9-46v+IFSp", "GammaH", [1, 2]],
]
# --------------------------------------------------------------------------------


def expand(area, merged: dict, parcels: set) -> tuple[list, str]:
    """area spec -> (list of parcels, label)."""
    items = area if isinstance(area, (list, tuple)) else [area]
    out = []
    for item in items:
        if item in merged:                              # merged-region name
            out += merged[item]
        elif item in parcels:                           # single parcel (may contain '-')
            out.append(item)
        else:                                           # parcels / regions joined with '+'
            for part in item.split("+"):
                if part in merged:
                    out += merged[part]
                elif part in parcels:
                    out.append(part)
                else:
                    raise ValueError(f"unknown area {part!r} in {item!r}")
    label = " + ".join(items)
    return list(dict.fromkeys(out)), label


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--specs", type=Path, help="JSON file with a list of [area, band, seconds] (default: FIGURES)")
    ap.add_argument("--merged", type=Path, default=REPO / "electrode coverages" / "figures" /
                    "merged_regions_min4_staged.csv")
    ap.add_argument("--test-only", action="store_true", help="only the model's test trials")
    ap.add_argument("--unit", default="dB")
    ap.add_argument("--clip", type=float, default=99.5)
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    specs = json.loads(args.specs.read_text()) if args.specs else FIGURES
    table = pd.read_csv(args.merged, dtype=str)
    merged = table.groupby("region").parcel.apply(list).to_dict()
    print(f"loading {args.pkl} ...")
    X = load_features(args.pkl, args.test_only)                # loaded once
    available = set(X.columns.get_level_values("region"))

    failed = []
    for i, spec in enumerate(specs, 1):
        try:
            area, band, seconds = spec
            parcels, label = expand(area, merged, available | set(table.parcel))
            have = [p for p in parcels if p in available]
            if not have:
                raise ValueError(f"no features for {label} (parcels {parcels})")
            print(f"[{i}/{len(specs)}] {label} | {band} | seconds {seconds}")
            make_plot(X, have, band, seconds, args.out_dir, args.test_only, args.unit, args.clip, label=label)
        except Exception as e:                           # keep going with the other figures
            print(f"[{i}/{len(specs)}] skipped {spec}: {e}")
            failed.append(spec)
    print(f"done: {len(specs) - len(failed)} figures written to {args.out_dir}"
          + (f"; {len(failed)} skipped" if failed else ""))


if __name__ == "__main__":
    main()
