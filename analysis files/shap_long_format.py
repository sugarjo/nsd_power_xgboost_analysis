# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:25:50 2026

@author: jorgels

Trial x feature SHAP values in long format, signed as in the plots
(shap analysis/signed_shap.py):

  shap_new_seen    SHAP(new) - (SHAP(recent) + SHAP(old)) / 2, sign flipped on
                   recent and old trials, so > 0 always pushes towards the trial's
                   correct new/seen call (all trials)
  shap_recent_old  SHAP(old) - SHAP(recent), sign flipped on recent trials, so > 0
                   pushes towards the correct recent/old call (NaN on new trials)

image_category in the pickle is 1 = new, 2 = recent, 3 = old (errors -1/-2/-3).
SHAP class order is 0 = new, 1 = recent, 2 = old. Missing (NaN) features are left
out, as in the plots.

To reproduce a plotted bar: per trial sum the values of the features in the
region/band/second, average per subject and trial type, combine the trial types
with the balanced weights (new vs seen: new 1/2, recent 1/4, old 1/4; recent vs
old: 1/2, 1/2), then average over subjects.

Usage: python shap_long_format.py [pickle] [merge table] [labels xlsx | none] [output csv] [test|error]
"""
import sys

import numpy as np
import pandas as pd

args = sys.argv[1:]
file_path = args[0] if len(args) > 0 else r"\\hypatia.uio.no\lh-med-imb-cnp\nsd\overall_RF\all_trials\real_data_bipolar_xgboost_3w.pkl"
merged_path = args[1] if len(args) > 1 else r"C:\Users\jorgels\OneDrive - Universitetet i Oslo\Dokumenter\GitHub\nsd_power_xgboost_analysis\electrode coverages\figures\merged_regions_min4_staged.csv"
labels_path = args[2] if len(args) > 2 else r"\\hypatia.uio.no\lh-med-imb-cnp\anatomical_analysis\HPC_MMP1_Labels.xlsx"
out_path = args[3] if len(args) > 3 else "shap_long_format.csv"
trials = args[4] if len(args) > 4 else "test"          # "test" (correct) or "error"

data = pd.read_pickle(file_path)
merged_areas = pd.read_csv(merged_path, dtype=str)

if trials == "error":
    shap_values, index = data["models"]["error_shap_values"], data["models"]["error_data"]
else:
    shap_values, index = data["models"]["test_shap_values"], data["models"]["test_data"]

features = list(data["features"].columns[:-1])          # last column: subject_stratification
X = data["features"].loc[index]
present = X[features].notna().to_numpy()               # trials x features
category = np.abs(index.get_level_values("image_category").to_numpy()).astype(int)   # 1 new, 2 recent, 3 old

new, recent, old = shap_values[..., 0], shap_values[..., 1], shap_values[..., 2]
n_s = new - (recent + old) / 2
n_s = np.where((category == 1)[:, None], n_s, -n_s)
r_o = old - recent
r_o = np.where((category == 3)[:, None], r_o, np.where((category == 2)[:, None], -r_o, np.nan))

# feature name -> parcel, band, second
parcel = np.array([f.strip("'").split(" | ")[0] for f in features])
band_sec = [f.strip("'").split(" | ")[1] for f in features]
freq = np.array([b[:-1] for b in band_sec])
sec = np.array([int(b[-1]) for b in band_sec])
region_of = merged_areas.set_index("parcel")["region"].to_dict()
region = np.array([region_of.get(p, p) for p in parcel])

if labels_path.lower() != "none":
    labels = pd.read_excel(labels_path)
    roi_of = dict(zip(labels["Label"].astype(str), labels["Regions ROIs"]))
    roi = np.array([roi_of.get("Hippocampus" if "Hippocampus" in p else p, np.nan) for p in parcel], dtype=object)
else:
    roi = np.full(len(parcel), np.nan, dtype=object)

t, k = np.nonzero(present)                             # one row per trial x non-missing feature
long_format = pd.DataFrame({
    "subject": index.get_level_values("subject").to_numpy()[t],
    "category": category[t],
    "error": trials == "error",
    "session_id": X["subject_stratification"].to_numpy()[t],
    "trial": t,
    "pic": index.get_level_values("pic").to_numpy()[t],
    "repetition": index.get_level_values("repetition").to_numpy()[t],
    "parcel": parcel[k],
    "glasser_area": region[k],
    "regions ROI": roi[k],
    "time": sec[k],
    "frequency": freq[k],
    "shap_new_seen": n_s[t, k],
    "shap_recent_old": r_o[t, k],
})
long_format.to_csv(out_path, index=False)
print(len(long_format), "rows ->", out_path)
