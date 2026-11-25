# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 21:25:50 2026

@author: jorgels
"""
import pickle
import pandas as pd
import numpy as np


file_path = r"\\hypatia.uio.no\lh-med-imb-cnp\nsd\overall_RF\all_trials\real_data_bipolar_xgboost_3w.pkl"

with open(file_path, "rb") as f:
    data = pickle.load(f)

labels = pd.read_excel(r"\\hypatia.uio.no\lh-med-imb-cnp\anatomical_analysis\HPC_MMP1_Labels.xlsx")

merged_areas = pd.read_csv(r"C:\Users\jorgels\OneDrive - Universitetet i Oslo\Dokumenter\GitHub\nsd_power_xgboost_analysis\electrode coverages\figures\merged_regions_min4_staged.csv")


long_format = []

shap_values = data["models"]["test_shap_values"]

features = data["features"].keys()[:-1]

for ind, (trial_id, repetition, label, subject)  in enumerate(data["models"]["test_data"]):
    print(ind, 'of', len(data["models"]["test_data"]))
    for k, feature in enumerate(features):
        
        if np.isnan(data["features"].loc[(trial_id, repetition, label, subject)][feature]):
            continue
        
        glasser, freq_t = feature.strip("'").split(" | ")
        
        if glasser  == 'Putamen':
            continue
        
        selected = merged_areas.parcel == glasser
        merged_region = merged_areas.loc[selected, "region"].values[0]
        
        
        sec = int(freq_t[-1])
        freq = freq_t[:-1]
        
        if 'Hippocampus' in glasser:        
            ana_ROI_sel = labels['Label'] == 'Hippocampus'
        else:
            ana_ROI_sel = labels['Label'].astype('string') == glasser
        
        if not sum(ana_ROI_sel) == 1:
            print(glasser)
        
        ana_ROI = labels['Regions ROIs'].iloc[ana_ROI_sel].iloc[0]
        
        if label == 0:
            n_s = shap_values[ind, k, 0] - (shap_values[ind, k, 1] + shap_values[ind, k, 2])/2
            r_o = np.nan
        elif label == 1:
            n_s = -shap_values[ind, k, 0] + (shap_values[ind, k, 1] + shap_values[ind, k, 2])/2
            r_o = shap_values[ind, k, 1] - shap_values[ind, k, 2]
        else:
            n_s = -shap_values[ind, k, 0] + (shap_values[ind, k, 1] + shap_values[ind, k, 2])/2
            r_o = shap_values[ind, k, 2] - shap_values[ind, k, 1]
        
        long_format.append({
            "subject": subject,
            "category": label,
            "session_id":  data["features"].loc[(trial_id, repetition, label, subject)].subject_stratification,
            "trial": ind,
            "glasser_area": merged_region,
            "regions ROI": ana_ROI,
            "time": sec,
            "frequency": freq,
            "shap_new_seen": n_s,
            "shap_recent_old": r_o,
            })

pd.DataFrame(long_format).to_csv("shap_long_format.csv", index=False)
len(long_format)