"""Export the SHAP values as one long table: one row per trial x feature.

Each row is one feature (parcel x band x second) of one trial, with
  set                     test (correct responses), unanswered or error
  pic, repetition, image_category, subject   the trial's index in the pickle
  trial_type              new / recent / old (the image shown, whatever the answer)
  subject_stratification  from the features table
  parcel                  HCP-MMP1 parcel (or amygdala / hippocampus), not merged
  <label column>          the parcel's group from --labels (default "Regions ROIs")
  band, second
  shap_new, shap_recent, shap_old         SHAP value per class (log-odds)
  new_vs_repeated         shap_new - (shap_recent + shap_old)/2
  consolidated_vs_recent  shap_old - shap_recent

Features that were NaN for the trial (no channel in the parcel, or that second
missing) are left out, as in the figures; --keep-missing keeps them and adds a
"missing" column. The SHAP values are taken as they are in the pickle (no
recomputation, no sign flip).

Parcels are matched to the --labels table on its "Label" column; a parcel
like Hippocampus-anterior that is not in the table falls back to the name
before the "-" (Hippocampus). Unmatched parcels are listed and get an empty
group.

Example:
  python export_shap_long.py --pkl real.pkl --out shap_values_long.csv.gz
  python export_shap_long.py --pkl real.pkl --sets test unanswered error
  python export_shap_long.py --pkl real.pkl --per-subject    # one file per subject
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from signed_shap import feature_columns

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SETS = {"test": "test", "unanswered": "unanswered", "error": "error"}
TRIAL_TYPE = {1: "new", 2: "recent", 3: "old"}         # |image_category| mod 3 -> image shown


def parcel_groups(parcels, labels: Path, column: str) -> dict:
    """Parcel -> group from the labels table (see module docstring)."""
    table = pd.read_excel(labels) if labels.suffix in (".xlsx", ".xls") else pd.read_csv(labels)
    if column not in table.columns:
        raise SystemExit(f"--label-column {column!r} not in {labels.name}; columns: {list(table.columns)}")
    lookup = table.dropna(subset=["Label"]).set_index(table["Label"].astype(str).str.strip())[column].to_dict()
    groups, unmatched = {}, []
    for p in parcels:
        g = lookup.get(p, lookup.get(p.split("-")[0]))
        if g is None or pd.isna(g):
            unmatched.append(p)
        groups[p] = g
    if unmatched:
        print(f"no {column!r} for {len(unmatched)} parcels: {unmatched}")
    return groups


def long_table(data: dict, which: str, groups: dict, column: str, keep_missing: bool) -> pd.DataFrame:
    models = data["models"]
    shap = models[f"{which}_shap_values"]
    index = models[f"{which}_data"]
    names = list(models["xgboost"].feature_names_in_)
    assert shap.shape[:2] == (len(index), len(names)), which
    assert names == list(data["features"].columns[:-1])

    X = data["features"].loc[index]
    missing = X[names].isna().to_numpy()
    rows, cols = np.nonzero(np.ones_like(missing) if keep_missing else ~missing)

    feat = feature_columns(names)
    trials = index.to_frame(index=False)
    category = trials["image_category"].to_numpy()
    trial_type = pd.Series(np.abs(category).astype(int)).map(lambda c: TRIAL_TYPE[(c - 1) % 3 + 1]).to_numpy()
    parcel = feat.get_level_values("region").to_numpy()
    s = shap[rows, cols]                                   # (n_rows, 3)

    out = {"set": which}
    for name in trials.columns:
        out[name] = trials[name].to_numpy()[rows]
    out["trial_type"] = trial_type[rows]
    out["subject_stratification"] = X["subject_stratification"].to_numpy()[rows]
    out["parcel"] = parcel[cols]
    out[column] = pd.Series(parcel).map(groups).to_numpy()[cols]
    out["band"] = feat.get_level_values("band").to_numpy()[cols]
    out["second"] = feat.get_level_values("second").to_numpy()[cols]
    if keep_missing:
        out["missing"] = missing[rows, cols]
    out["shap_new"], out["shap_recent"], out["shap_old"] = s[:, 0], s[:, 1], s[:, 2]
    out["new_vs_repeated"] = s[:, 0] - (s[:, 1] + s[:, 2]) / 2
    out["consolidated_vs_recent"] = s[:, 2] - s[:, 1]
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--labels", type=Path, default=REPO / "electrode coverages" / "HPC_MMP1_Labels.xlsx")
    ap.add_argument("--label-column", default="Regions ROIs")
    ap.add_argument("--sets", nargs="+", choices=list(SETS), default=["test"],
                    help="which SHAP sets to export (default: test = correct responses)")
    ap.add_argument("--keep-missing", action="store_true", help="also export features that were NaN")
    ap.add_argument("--out", type=Path, default=HERE / "shap_values_long.csv.gz",
                    help="output file; .gz compresses")
    ap.add_argument("--per-subject", action="store_true",
                    help="write one file per subject into a folder named after --out "
                         "(<stem>/<stem>_<subject>.csv.gz) instead of one file")
    args = ap.parse_args()

    data = pd.read_pickle(args.pkl)
    names = list(data["models"]["xgboost"].feature_names_in_)
    groups = parcel_groups(feature_columns(names).get_level_values("region").unique(), args.labels, args.label_column)

    table = pd.concat([long_table(data, s, groups, args.label_column, args.keep_missing) for s in args.sets],
                      ignore_index=True)
    if args.per_subject:
        stem = args.out.name.split(".")[0]
        folder = args.out.with_name(stem)
        folder.mkdir(parents=True, exist_ok=True)
        for subject, part in table.groupby("subject"):
            part.to_csv(folder / f"{stem}_{subject}{args.out.name[len(stem):]}", index=False, float_format="%.6g")
        args.out = folder
    else:
        table.to_csv(args.out, index=False, float_format="%.6g")
    print(f"wrote {args.out}: {len(table)} rows, {table['subject'].nunique()} subjects, "
          f"{table['parcel'].nunique()} parcels, sets {args.sets}")


if __name__ == "__main__":
    main()
