"""Collapse the 3-class SHAP values to one signed new-vs-repetition value.

data["models"]["test_shap_values"] (older files: "shap_values") has shape
(trials, features, classes) with the
classes new (0), recent (1) and old (2); its rows follow
data["models"]["test_data"] and its features follow data["features"] without
the last column (subject_stratification). The values are in the model's
log-odds (margin) units.

For every trial and feature:
  1. repetition = (SHAP(recent) + SHAP(old)) / 2
  2. new_vs_rep = SHAP(new) - repetition     (> 0 pushes towards "new")
  3. signed     = new_vs_rep on new trials, -new_vs_rep on recent/old trials,
                  so a positive value is a contribution towards the correct
                  label (new vs repeated).

The softmax probabilities only depend on differences between the class
margins, so a single class's SHAP value has no meaning on its own. The mean
contrast in step 2 is the average of the two pairwise log-odds contrasts
(new vs recent and new vs old), i.e. log(p_new / sqrt(p_recent * p_old)),
and it stays exactly additive. Summing recent and old instead would count
"repetition" twice and is not invariant to a shift of all margins.

Writes shap_new_vs_repetition.pkl next to this script (git-ignored), a dict:
  "signed"     DataFrame trials x features, index = test_data, columns =
               MultiIndex (region, band, second)
  "new_vs_rep" same, before the sign flip (also as "contrast")
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

# The contrasts the analysis can be run for. "weights" are the within-subject
# weights of the trial types (see subject_means in plot_new_vs_repeated.py).
CONTRASTS = {
    "new_vs_repeated": dict(
        prefix="new_vs_repeated", title="new vs repeated (recent + old)", short="new − repeated",
        call="new/repeated", flipped="repeated", weights={"new": 0.5, "recent": 0.25, "old": 0.25},
        balance="new, recent and old weighted 1/2, 1/4, 1/4",
        formula="SHAP(new) − (SHAP(recent) + SHAP(old))/2"),
    "recent_vs_consolidated": dict(
        prefix="recent_vs_consolidated", title="recent vs consolidated (old)", short="consolidated − recent",
        call="recent/consolidated", flipped="recent", weights={"recent": 0.5, "old": 0.5},
        balance="recent and consolidated weighted 1/2, 1/2; new trials left out",
        formula="SHAP(consolidated) − SHAP(recent)"),
}


def feature_columns(names) -> pd.MultiIndex:
    """'Amygdala | Alpha1' -> ('Amygdala', 'Alpha', 1)."""
    region, band = zip(*(n.split(" | ") for n in names))
    return pd.MultiIndex.from_arrays(
        [list(region), [b[:-1] for b in band], [int(b[-1]) for b in band]],
        names=["region", "band", "second"])


def signed_shap(data: dict, contrast: str = "new_vs_repeated") -> dict:
    """Per trial x feature contrast values; see the module docstring.

    contrast "new_vs_repeated":        SHAP(new) - (SHAP(recent) + SHAP(old))/2
                                        (> 0 pushes towards new), all trials.
    contrast "recent_vs_consolidated": SHAP(old) - SHAP(recent)
                                        (> 0 pushes towards consolidated/old),
                                        recent and old trials only.
    "signed" flips the sign on trials of the negative label, so > 0 is always a
    push towards the trial's correct label."""
    models = data["models"]
    shap = models["test_shap_values"] if "test_shap_values" in models else models["shap_values"]
    test = models["test_data"]
    names = list(models["xgboost"].feature_names_in_)
    assert shap.shape[:2] == (len(test), len(names))
    assert names == list(data["features"].columns[:-1])
    category = test.get_level_values("image_category").to_numpy()

    if contrast == "new_vs_repeated":
        value = shap[..., NEW] - (shap[..., RECENT] + shap[..., OLD]) / 2
        positive = category == NEW_CATEGORY
    elif contrast == "recent_vs_consolidated":
        keep = np.isin(category, [2, 3])                  # recent and old (consolidated) trials
        shap, test, category = shap[keep], test[keep], category[keep]
        value = shap[..., OLD] - shap[..., RECENT]
        positive = category == 3
    else:
        raise ValueError(f"unknown contrast {contrast!r}")
    signed = np.where(positive[:, None], value, -value)

    cols = feature_columns(names)
    frame = lambda a: pd.DataFrame(a.astype(np.float32), index=test, columns=cols)
    missing = data["features"].loc[test, names].isna().to_numpy()
    return {"signed": frame(signed), "contrast": frame(value), "new_vs_rep": frame(value),
            "missing": pd.DataFrame(missing, index=test, columns=cols)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", type=Path, default=REPO / "random_data_bipolar_xgboost_3w.pkl")
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
