"""Collapse the 3-class SHAP values to one signed new-vs-repetition value.

data["models"]["shap_values"] has shape (trials, features, classes) with the
classes new (0), recent (1) and old (2); its rows follow
data["models"]["test_data"] and its features follow data["features"] without
the last column (subject_stratification). The values are in the model's
log-odds (margin) units.

For every trial and feature:
  1. repetition = SHAP(recent) + SHAP(old)
  2. new_vs_rep = SHAP(new) - repetition     (> 0 pushes towards "new")
  3. signed     = new_vs_rep on new trials, -new_vs_rep on recent/old trials,
                  so a positive value is a contribution towards the correct
                  label (new vs repeated).

Writes shap_new_vs_repetition.pkl next to this script (git-ignored), a dict:
  "signed"     DataFrame trials x features, index = test_data, columns =
               MultiIndex (region, band, second)
  "new_vs_rep" same, before the sign flip
  "missing"    bool DataFrame, True where the feature was NaN for that trial
               (no channel in the region, or that second missing)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

NEW, RECENT, OLD = 0, 1, 2          # class order of the XGBClassifier
NEW_CATEGORY = 1                    # image_category of a (correct) new trial


def feature_columns(names) -> pd.MultiIndex:
    """'Amygdala | Alpha1' -> ('Amygdala', 'Alpha', 1)."""
    region, band = zip(*(n.split(" | ") for n in names))
    return pd.MultiIndex.from_arrays(
        [list(region), [b[:-1] for b in band], [int(b[-1]) for b in band]],
        names=["region", "band", "second"])


def signed_shap(data: dict) -> dict:
    models = data["models"]
    shap = models["shap_values"]
    test = models["test_data"]
    names = list(models["xgboost"].feature_names_in_)
    assert shap.shape[:2] == (len(test), len(names))
    assert names == list(data["features"].columns[:-1])

    repetition = shap[..., RECENT] + shap[..., OLD]
    new_vs_rep = shap[..., NEW] - repetition
    is_new = test.get_level_values("image_category").to_numpy() == NEW_CATEGORY
    signed = np.where(is_new[:, None], new_vs_rep, -new_vs_rep)

    cols = feature_columns(names)
    frame = lambda a: pd.DataFrame(a.astype(np.float32), index=test, columns=cols)
    missing = data["features"].loc[test, names].isna().to_numpy()
    return {"signed": frame(signed), "new_vs_rep": frame(new_vs_rep),
            "missing": pd.DataFrame(missing, index=test, columns=cols)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "real_data_bipolar_xgboost_3w.pkl")
    ap.add_argument("--out", type=Path, default=HERE / "shap_new_vs_repetition.pkl")
    args = ap.parse_args()

    out = signed_shap(pd.read_pickle(args.pkl))
    pd.to_pickle(out, args.out)

    s, miss = out["signed"], out["missing"]
    total = s.sum(axis=1)
    cat = s.index.get_level_values("image_category")
    print(f"wrote {args.out}  ({s.shape[0]} trials x {s.shape[1]} features)")
    print("trials whose summed signed SHAP favors the correct label:")
    print(total.gt(0).groupby(np.where(cat == NEW_CATEGORY, "new", "repetition")).mean().round(3).to_string())
    print(f"share of |signed SHAP| carried by missing features: "
          f"{s.abs().to_numpy()[miss.to_numpy()].sum() / s.abs().to_numpy().sum():.3f}")


if __name__ == "__main__":
    main()
