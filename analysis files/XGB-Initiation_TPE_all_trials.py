import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import read_data
import feature_setup
import shap
import pandas as pd
import pickle
import multiprocessing
from joblib import Parallel, delayed
from pathlib import Path

import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, atpe
from hyperopt.early_stop import no_progress_loss
from hyperopt.fmin import generate_trials_to_calculate

from neural_decoding.utils import anatomy
from neural_decoding.utils import get_ephys_meta

from matplotlib import pyplot as plt



# CNP04 should be first to set up the image structure- #CNP11 not included because of previous surgery. CNP13 excluded because of few old trials. CNP21 has a lot of premature button presses
subjects = ['CNP04', 'CNP21', 'CNP05', 'CNP08', 'CNP09', 'CNP10', 'CNP12',
            'CNP14', 'CNP18', 'CNP19', 'CNP20', 'CNP22', 'CNP26', 'CNP27', 'CNP28', 'CNP29']

# bands = ['Delta1', 'Delta2',
#          'Theta1', 'Theta2',
#          'Alpha1', 'Alpha2',
#          'Beta1', 'Beta2',
#          'GammaL1', 'GammaL2', 
#          'GammaH1', 'GammaH2',
#          'Ripple1', 'Ripple2']

#later output is shaped: trials x features, where the features are ordered: bands, seconds and channels
bands = ['Delta1' , 'Theta1' , 'Alpha1' , 'Beta1' , 'GammaL1' , 'GammaH1' , 'Ripple1', 'Delta2' , 'Theta2' , 'Alpha2' , 'Beta2' , 'GammaL2' , 'GammaH2', 'Ripple2']
  

frequency_cutoffs = [(1, 4),
                     (4, 8),
                     (8, 16),
                     (16, 32),
                     (32, 64),
                     (64, 128),
                     (128, 256)]

continue_hyperparam_search = False
override_old = True
bipolar = True
hyperparams_iterations = 50#50


button_limit = 2.25
z_score_remove = 10
random_shuffles = 1000
fill_missing_data = 'nan'
bipolar = True

test_accuracies = []
train_accuracies = []
num_features = []

cpu_count = multiprocessing.cpu_count()





def objective(space):

    # print(space)
    # return

    pars = {
        'max_depth': int(space['max_depth']),
        'gamma': space['gamma'],
        'reg_lambda': space['reg_lambda'],
        'reg_alpha': space['reg_alpha'],
        'min_child_weight': space['min_child_weight'],
        'learning_rate': space['learning_rate'],
        'subsample': space['subsample'],
        'colsample_bytree': space['colsample_bytree'],
        'colsample_bylevel': space['colsample_bylevel'],
        'colsample_bynode': space['colsample_bynode'],
        'early_stopping_rounds': int(space['early_stopping_rounds']),
        'n_estimators': int(space['n_estimators']),
        'max_delta_step': space['max_delta_step'],
        'grow_policy': grow_policy[space['grow_policy']],
        'max_leaves': int(space['max_leaves']),
        'max_bin': int(space['max_bin'])
    }

    # print(pars["max_leaves"])

    fit_X, eval_X, fit_y, eval_y, fit_subject_categories, eval_subject_categories = train_test_split(
        subsample_X, subsample_y, subsample_subject_categories, stratify=subsample_y, test_size=space['eval_fraction'], random_state=0)

    # get class weights for balanced training
    # np.array([1,  best_hyperparams['sample_weights_1'],  best_hyperparams['sample_weights_2']]) #
    class_weights = compute_class_weight(
        'balanced', classes=unique_subject_categories, y=fit_subject_categories)

    fit_sample_weight = []
    for y in fit_subject_categories:
        which_class = y == unique_subject_categories
        fit_sample_weight.append(class_weights[which_class])

    eval_sample_weight = []
    for y in eval_subject_categories:
        which_class = y == unique_subject_categories
        eval_sample_weight.append(class_weights[which_class])

    # val_sample_weight=[]
    # for y in validation_subject_categories:
    #     val_sample_weight.append(class_weights[y])

    # sum_weights = sum(overall_gm_feature_weights)
    # image_weight = space['image_weight']*sum_weights

    # feature_weights = overall_gm_feature_weights + [image_weight]

    model = xgb.XGBClassifier(
        **pars, num_class=len(unique_y), objective="multi:softprob", eval_metric="auc")

    model.fit(fit_X, fit_y, verbose=False, sample_weight=fit_sample_weight, sample_weight_eval_set=(fit_sample_weight, eval_sample_weight),
              eval_set=[(fit_X, fit_y), (eval_X, eval_y)])

    pred = model.predict_proba(val_X)

    # make sure that only one class is used for sampling
    roc = roc_auc_score(val_y, pred, average='macro', multi_class='ovo')

    return {'loss': -roc, 'status': STATUS_OK}


def shuffle_models(ind, shift):

    # roll the y labels by a random number
    y_copy = image_category.copy()
    shuffled_y = np.roll(y_copy, shift, axis=0)

    #shuffled_y = np.array(shuffled_y).astype(int)
    shuffled_y = roll_within_subject(image_category, shift).astype(int)

    # roll the y labels by a random number
    stratify_copy = stratify_sessions.copy()
    #shuffled_stratify = np.roll(stratify_copy, shift, axis=0)
    shuffled_stratify = roll_within_subject(stratify_sessions, shift)

    subject_categories_copy = subject_categories.copy()
    # shuffled_subject_categories = np.roll(
    #     subject_categories_copy, shift, axis=0)
    shuffled_subject_categories = roll_within_subject(subject_categories, shift)

    s_train_X, s_test_X, s_train_y, s_test_y, s_train_subject_categories, s_test_subject_categories = train_test_split(
        correct_df, shuffled_y, shuffled_subject_categories, stratify=shuffled_stratify, test_size=test_size, random_state=ind)
    # s_train_X, s_test_X, s_train_y, s_test_y, s_train_subject_categories, s_test_subject_categories, s_train_original_order, _ = train_test_split(correct_df, shuffled_y, shuffled_subject_categories, original_order, stratify=shuffled_stratify, test_size=test_size, random_state=ind)
    # #reorganize to the original order
    # s_train_order = np.argsort(s_train_original_order)
    # s_train_X = s_train_X.iloc[s_train_order]
    # s_train_y = [s_train_y[i] for i in s_train_order]
    # s_train_subject_categories = [s_train_subject_categories[i] for i in s_train_order]

    s_train_y = np.array(s_train_y).astype(int)-1
    s_test_y = np.array(s_test_y).astype(int)-1

    # get som data for evaluation
    s_fit_X, s_eval_X, s_fit_y, s_eval_y, s_fit_subject_categories, s_eval_subject_categories = train_test_split(
        s_train_X, s_train_y, s_train_subject_categories, test_size=eval_fraction, stratify=s_train_y, random_state=ind)

    # get class weights
    s_unique_subject_categories = np.unique(s_fit_subject_categories)
    s_class_weights = compute_class_weight(
        'balanced', classes=s_unique_subject_categories, y=s_fit_subject_categories)

    s_fit_sample_weight = []
    for y in s_fit_subject_categories:
        which_class = y == s_unique_subject_categories
        s_fit_sample_weight.append(s_class_weights[which_class])

    s_eval_sample_weight = []
    for y in s_eval_subject_categories:
        which_class = y == s_unique_subject_categories
        s_eval_sample_weight.append(s_class_weights[which_class])

    shuffled_model = xgb.XGBClassifier(
        **best_hyperparams, num_class=3, objective="multi:softprob", eval_metric="auc")

    shuffled_model.fit(s_fit_X, s_fit_y, verbose=False, sample_weight=s_fit_sample_weight, sample_weight_eval_set=(s_fit_sample_weight, s_eval_sample_weight),
                       eval_set=[(s_fit_X, s_fit_y), (s_eval_X, s_eval_y)])

    s_test_pred = shuffled_model.predict(s_test_X)

    shuffled_confusion_matrix = confusion_matrix(s_test_y, s_test_pred)

    # calculate shap values
    # shuffled_explainer = shap.TreeExplainer(shuffled_model, data=shuffled_train_X)
    # shuffled_shaps = np.array(shuffled_explainer.shap_values(shuffled_test_X, check_additivity=False))
    shuffled_shaps = []

    # compress_shaps(shuffled_shaps, shuffled_test_X, shuffled_test_y)
    compressed_shap_values = shuffled_shaps

    return (shuffled_confusion_matrix, compressed_shap_values, shift, shuffled_model)


def compress_shaps(shaps, x_data, y_labels):

    # compress shap values from trial x feature x category to featur x category

    # initate array
    compressed_shaps = np.zeros((len(np.unique(y_labels)), x_data.shape[1]))

    for current_class in np.unique(y_labels).astype(int):
        # trials with label of interest
        selected = y_labels == current_class
        # #mean shap value of features for these trials. add by two since we will subtract values from the two other classes.
        positive_shap_values = np.nanmean(
            shaps[current_class, selected, :], axis=0)
        # #add these to feature importance_metric
        compressed_shaps[current_class, :] += positive_shap_values
        # #add these to feature importance_metric
        for other_class in np.unique(y_labels).astype(int):
            # skip if same
            if current_class == other_class:
                continue
            # if feature is important for predicting the wrong class, the feature should have negative impact on feat importance
            negative_shap_values = np.nanmean(
                shaps[other_class, selected, :], axis=0)

            compressed_shaps[current_class, :] -= negative_shap_values

    return compressed_shaps

def roll_within_subject(values, shift):
    #Roll values by `shift` positions, separately within each subject.
    values = np.asarray(values)
    rolled = values.copy()
    for s in np.unique(trial_subjects):
        idx = np.flatnonzero(trial_subjects == s)
        rolled[idx] = np.roll(values[idx], shift % len(idx))
    return rolled


xgb_models = []
test_accuracies = []
train_accuracies = []
hyperparams = []

overall_data = []
# overall_gm_feature_weights = []

hypatia_path = os.environ["CNP_HYPATIA"]


# import sys
# from datetime import datetime
# created_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# log_file_name = os.path.join(hypatia_path, "nsd", "logfiles", f"{created_at}.txt")

# log_file = open(log_file_name, "a", buffering=1)
# sys.stdout = log_file
# sys.stderr = log_file



savedir = os.path.join(hypatia_path, "nsd", "overall_RF", "all_trials")

# open the regions file
# Path to the Excel file
region_path = os.path.join(hypatia_path, "anatomical_analysis", "HPC_MMP1_Labels.xlsx")

# Read the Excel file
anatomy_df = pd.read_excel(region_path)

glasser_areas = anatomy_df.Label.values.tolist()
glasser_areas = [item for item in glasser_areas if item != '/']
glasser_areas.append("Putamen")
glasser_areas.append("Hippocampus-anterior")
glasser_areas.append("Hippocampus-posterior")

# collect behaviour
# first row is correct, 2nd is error. columns are categories
behaviour = np.zeros((2, 3))

filename = 'hyperparameter_search_3w.pkl'
model_filename = 'xgboost_3w.pkl'

if bipolar:
    hyperparam_filename = 'bipolar_' + filename
    model_filename = 'bipolar_' + model_filename
    figure_name = 'bipolar_confusion_matrix.png'
else:
    hyperparam_filename = 'referential_' + filename
    model_filename = 'referential_' + model_filename
    figure_name = 'referential_confusion_matrix.png'

if not override_old:
    with open(os.path.join(savedir, 'real_data_' + model_filename), 'rb') as f:
        saved_model = pickle.load(f)
    
    overall_df = saved_model["features"]
    
else:

    for subject_ind, subject in enumerate(subjects):
        print('\n\n')
        print(subject)
    
        # with open(os.path.join(savedir, 'hyperparameter_search.pkl'), 'rb') as f:
        #     old_trials = pickle.load(f)
    
        # old_hyperparams = old_trials.best_trial['misc']['vals']
        # #reformat the lists
        # test_hyperparams = {}
        # for field, val in old_hyperparams.items():
        #     test_hyperparams[field] = val[0]
    
        # hyperparams.append(test_hyperparams.copy())
    
        if bipolar:
            input_dir = os.path.join(os.environ["CNP_HYPATIA"], "data_seeg/", subject, "analysis_natural_scenes_task", "bipolar_trials")
        else:
            input_dir = os.path.join(os.environ["CNP_HYPATIA"], "data_seeg/", subject, "analysis_natural_scenes_task", "referential_trials")

        trial_dir = os.path.join(os.environ["CNP_HYPATIA"], "data_seeg/", subject, "analysis_natural_scenes_task", "Man_Trial_Inclusion_Indices.pkl")    

        # 'ephys' (extract raw ffts from timeseries) or 'session_corrected_fft', 'block_corrected_fft'. Run bas_corr_fft to store the corrected ffts before running this script
        extract_field = 'session_corrected_fft'

    
        data, fft_freq, channels, [pic, presentation_time, responses, repetitions, time_last_presentation] = read_data.load_data(
            input_dir, extract_field, button_limit, trial_dir)

        print(str(int(np.shape(data)[0])) + ' trials')

        print(str(len(channels)) + ' channels')
        print(str(((data.size - np.sum(np.isfinite(data))) +
              np.sum(data == 0))/data.size) + ' zeros, inf or nans in FFTs')
    
        yreal = feature_setup.data_labelling(pic, presentation_time, responses, time_last_presentation)
        for c in range(1, 4):
            selected = yreal[:, 0] == c
            behaviour[0, c-1] += sum(selected)
    
            selected = yreal[:, 0] == -c
            behaviour[1, c-1] += sum(selected)
    
        # remove nans, zeroes and infs
        data, clean_channels = read_data.clean_dataset(
            data, channels, method=fill_missing_data)
        print(str(len(clean_channels)) + ' channels without missing data.')
    
        # get data in grey and white matter and remove channels outside brain and in lesioned areas
        # weights are set so that each lobe counts the same.
        cortex_data, features, wm_data, wm_features, _ = feature_setup.remove_ch(
            subject, channels, data, bipolar, True, bands)

    
        print(str(int(len(features)/2/7)) + ' cortex channels')
        print(str(int(len(features))) + ' cortex features')
    
        print(str(int(len(wm_features)/2/7)) + ' white matter channels')
        print(str(int(len(wm_features))) + ' white matter features')
    
        # get the band means
        #output is 4D: trials, channels, seconds, bands in ascending order
        xdata = feature_setup.compute_band_means(
            cortex_data, frequency_cutoffs, fft_freq, extract_field)
    
        # get the band means
        wm_xdata = feature_setup.compute_band_means(
            wm_data, frequency_cutoffs, fft_freq, extract_field)
    
        # remove outliers and flattens the matrix
        xdata, yreal, wm_xdata, noicy_trials = feature_setup.trial_removal(
            xdata, yreal, wm_xdata, z_score_remove)
    
        print(str(int(np.shape(xdata)[0])) + ' trials after automatic cleaning')

        #merge electrodes across each glasser area
        area_data_df = feature_setup.area_median(xdata, subject, features, bipolar, glasser_areas)
    
        # non_error_trials = (yreal[:, 0] > 0)
    
        # correct_trials = (yreal[:, 0]
    
        # #remove error / no response encoding of yreal
        # #selected = yreal[:, 0] < 0
        # #yreal[selected, 0]  = np.abs(yreal[selected, 0])
        # selected = yreal[:, 0] > 3
        # yreal[selected, 0]  = yreal[selected, 0] - 3
    
        # Create a temporary DataFrame for these trials
        # temp_df = pd.DataFrame(xdata[correct_trials], columns=[f"{subject}_{feature}" for feature in features])
        temp_df = area_data_df  # [correct_trials]
        temp_df['pic'] = pic[~noicy_trials]  # [correct_trials]
        temp_df['repetition'] = repetitions[~noicy_trials]  # [correct_trials]
        temp_df['image_category'] = yreal[:, 0]  # [correct_trials]
        # recording days/sessions
        temp_df['subject_stratification'] = yreal[:, 1] + \
            100*subject_ind  # [correct_trials]
        temp_df['subject'] = subject
    
        # Set an index based on pic and repetition for proper merging later
        # temp_df.set_index(['pic', 'repetition', 'image_category'], inplace=True)
        temp_df = temp_df.set_index(
            ['pic', 'repetition', 'image_category', 'subject'])#, inplace=True)
    
        overall_data.append(temp_df.copy())
    
    overall_df = pd.concat(overall_data)

    # remove columns without data
    non_data_cols = np.sum(np.isnan(overall_df), axis=0) == overall_df.shape[0]
    dropped_cols = overall_df.keys()[non_data_cols]
    
    # print those that are not covered in any patients
    printed = []
    k = 0
    for a in dropped_cols:
        parts = a.split(" | ")
        if parts[0] not in printed:
            print('No data for', parts[0])
            printed.append(parts[0])
            k += 1
    
    print('No data in', k, 'areas')

    overall_df = overall_df.loc[:, ~non_data_cols.values]

# add the image as a feature
# overall_df['image_ID'] = overall_df.index.get_level_values('pic')
# remember to give the image a feature weight in objective

image_labels = overall_df.index.get_level_values('image_category')
correct_trials = (image_labels > 0) & (image_labels < 4)
correct_df = overall_df.iloc[correct_trials]

# stratify for rec session and category
# add the to numbers together so that subject_category gets a unique value that can be used for stratification
stratify_sessions = (correct_df['subject_stratification'].copy(
) + correct_df.index.get_level_values('image_category')).tolist()
correct_df = correct_df.drop('subject_stratification', axis=1)

# turn to integer
stratify_sessions = [int(num) for num in stratify_sessions]

# remove the session id from the stratification list. we use this to give each category/person the same weight to the model.
subject_categories = [(n // 100) * 100 + (n % 10) for n in stratify_sessions]
unique_subject_categories = np.unique(subject_categories)

# #add positional encoding
# xdata = feature_setup.position_embedding(xdata, features, subject)
# wm_xdata = feature_setup.position_embedding(wm_xdata, wm_features, subject)


# correct_responses = (yreal[:, 0]>0) & (yreal[:, 0]<4)
# x_dataset_correct = xdata_df.loc[correct_responses, :]
# y_dataset_correct = yreal[correct_responses, :]

# split train/test. stratified for recording days
test_size = 0.5
val_size = 0.25

# join "sessions" if categories have n=1 from each session
unique_y, overall_count_y = np.unique(stratify_sessions, return_counts=True)
while min(overall_count_y) == 1:
    unique_ind = np.argmin(overall_count_y)
    min_val = unique_y[unique_ind]
    selected = stratify_sessions.index(min_val)
    stratify_sessions[selected] -= 10
    unique_y, overall_count_y = np.unique(
        stratify_sessions, return_counts=True)

image_category = [cat[2] for cat in correct_df.index]

# this scrambles the order. keep original order to make shuffling optimal
original_order = np.arange(0, correct_df.shape[0])


if override_old:
    train_X, test_X, train_y, test_y, train_subject_categories, _, train_original_order, _ = train_test_split(
        correct_df, image_category, subject_categories, original_order, stratify=stratify_sessions, test_size=test_size, random_state=42)
else:
    train_pd_ind = saved_model["models"]["train_data"]
    train_ordered_ind = correct_df.index.get_indexer(saved_model["models"]["train_data"])
    
    test_pd_ind = saved_model["models"]["test_data"]
    test_ordered_ind = correct_df.index.get_indexer(saved_model["models"]["test_data"])
    
    train_X = correct_df.loc[train_pd_ind]
    test_X = correct_df.loc[test_pd_ind]
    train_y = [image_category[i] for i in train_ordered_ind] 
    test_y = [image_category[i] for i in test_ordered_ind] 
    train_subject_categories = [subject_categories[i] for i in train_ordered_ind] 
    train_original_order = original_order[train_ordered_ind]

# reorganize to the original order
train_order = np.argsort(train_original_order)
train_X = train_X.iloc[train_order]
train_y = [train_y[i] for i in train_order]
train_subject_categories = [train_subject_categories[i] for i in train_order]

train_y = np.array(train_y).astype(int)-1
test_y = np.array(test_y).astype(int)-1

# make sure eval fraction allways have enough samples to test all classe
unique_y, counts_y = np.unique(train_y, return_counts=True)
min_eval_fraction = 1 / (np.min(counts_y) * (1-val_size))

grow_policy = ['depthwise', 'lossguide']

space = {'max_depth': hp.quniform("max_depth", 1, 35000, 1),
         # hp.loguniform('min_split_loss', -10, np.log(15))
         'gamma': hp.uniform('gamma', 0, 15),
         'reg_lambda': hp.uniform('reg_lambda', 0, 2000),
         'reg_alpha': hp.uniform('reg_alpha', 0.001, 50),
         # lower than 1?
         'min_child_weight': hp.uniform('min_child_weight', 0, 200),
         'learning_rate': hp.uniform('learning_rate', 0, 6),
         'subsample': hp.uniform('subsample', 0.01, 1),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1),
         'colsample_bylevel': hp.uniform('colsample_bylevel', 0.01, 1),
         'colsample_bynode': hp.uniform('colsample_bynode', 0.01, 1),
         'early_stopping_rounds': hp.quniform("early_stopping_rounds", 100, 35000, 1),
         'n_estimators': hp.quniform("n_estimators", 20, 60000, 1),
         # can fail for higher
         'eval_fraction': hp.loguniform('eval_fraction', np.log(min_eval_fraction), np.log(0.5)),
         # hp.loguniform('max_delta_step', -10, np.log(40)),
         'max_delta_step': hp.uniform('max_delta_step', 0, 150),
         'grow_policy': hp.choice('grow_policy', [0, 1]),
         'max_leaves': hp.qloguniform('max_leaves', -5, np.log(1500), 1),
         'max_bin': hp.qloguniform('max_bin', np.log(2), np.log(4000), 1),
         # 'image_weight': hp.uniform('image_weight', 0, 1), #1 indicates equal to all of the brain data. 0 nothing
         }

# split in train/validation set to keep consistent parts during model training
subsample_X, val_X, subsample_y, val_y, subsample_subject_categories, val_subject_categories = train_test_split(
    train_X, train_y, train_subject_categories, stratify=train_y, test_size=val_size, random_state=42)

# compare to previous search
try:
    
    if not override_old:
            
        old_best_hyperparams = saved_model["models"]["hyperparameters"]
            
    else:
            
        try:
            with open(os.path.join(savedir, hyperparam_filename), 'rb') as f:
                previous_trials = pickle.load(f)
        except:
            print('Did not find hyperparameter file')
            with open(os.path.join(savedir, filename), 'rb') as f:
                previous_trials = pickle.load(f)
    
        old_hyperparams = previous_trials.best_trial['misc']['vals']   
    
        # reformat the lists
        old_best_hyperparams = {}
        for field, val in old_hyperparams.items():
            old_best_hyperparams[field] = val[0]

    if hyperparams_iterations:

        old_loss = objective(old_best_hyperparams)["loss"]

        # old_best_hyperparams = fmin(fn = objective,
        #                 space = space,
        #                 algo = atpe.suggest,
        #                 max_evals = 1,
        #                 trials = old_trials)

        # old_loss = old_trials.best_trial["result"]["loss"]

        print('Old loss:', old_loss)

    else:
        old_loss = -1

except:
    print('No hyperparamters searched previously.')
    old_loss = np.inf


if hyperparams_iterations:
    if continue_hyperparam_search:
        with open(os.path.join(savedir, 'temp_hyperparam'), 'rb') as f:
            trials = pickle.load(f)
        next_iterations_limit = len(trials)
        best_loss = trials.best_trial["result"]["loss"]

        for i, t in enumerate(trials):
            try:
                if t["result"]["loss"] == best_loss:
                    ind = i
            except:
                True

        best_iteration = next_iterations_limit - ind
        
        print(best_iteration)

    else:
        # search from scratch
        trials = Trials()
        next_iterations_limit = 0
        best_loss = 0
        best_iteration = 0

    # check if reached the limit
    while best_iteration < hyperparams_iterations:
        
        print('Best iteration:', best_iteration)
        
        next_iterations_limit += 5
        best_iteration += 5

        best_hyperparams = fmin(fn=objective,
                                space=space,
                                algo=atpe.suggest,
                                max_evals=next_iterations_limit,
                                trials=trials)

        if trials.best_trial["result"]["loss"] < best_loss:
            best_loss = trials.best_trial["result"]["loss"]
            best_iteration = 0
            print(best_hyperparams)

        pickle.dump(trials, open(os.path.join(
            savedir, 'temp_hyperparam'), "wb"))

    # recalculate to be sure that the old/new losses are comparable
    new_loss = objective(best_hyperparams)["loss"]
    print('New loss:', new_loss)

else:
    new_loss = 0



if old_loss < new_loss:
    used_hyperparams = old_best_hyperparams.copy()
    best_hyperparams = old_best_hyperparams.copy()
    print('Using old')
else:
    # save if improved
    used_hyperparams = best_hyperparams.copy()
    print('Using new')
    pickle.dump(trials, open(os.path.join(savedir, hyperparam_filename), "wb"))



eval_fraction = best_hyperparams['eval_fraction']
del best_hyperparams['eval_fraction']

#add eval fraction so that it is stored
used_hyperparams["eval_fraction"] = eval_fraction


# image_weight = best_hyperparams['image_weight']
# del best_hyperparams['image_weight']

best_hyperparams['max_depth'] = int(best_hyperparams['max_depth'])
best_hyperparams['early_stopping_rounds'] = int(best_hyperparams['early_stopping_rounds'])
best_hyperparams['n_estimators'] = int(best_hyperparams['n_estimators'])
best_hyperparams['max_leaves'] = int(best_hyperparams['max_leaves'])

if isinstance(best_hyperparams['grow_policy'], int):
    best_hyperparams['grow_policy'] = grow_policy[best_hyperparams['grow_policy']]
    
best_hyperparams['max_bin'] = int(best_hyperparams['max_bin'])

# get som data for evaluation
fit_X, eval_X, fit_y, eval_y, fit_subject_categories, eval_subject_categories = train_test_split(
    train_X, train_y, train_subject_categories, test_size=eval_fraction, stratify=train_y, random_state=42)


# get class weights
unique_subject_categories = np.unique(fit_subject_categories)
class_weights = compute_class_weight(
    'balanced', classes=unique_subject_categories, y=fit_subject_categories)

fit_sample_weight = []
for y in fit_subject_categories:
    which_class = y == unique_subject_categories
    fit_sample_weight.append(class_weights[which_class])

eval_sample_weight = []
for y in eval_subject_categories:
    which_class = y == unique_subject_categories
    eval_sample_weight.append(class_weights[which_class])

if not 'saved_model' in locals() or not saved_model["models"]["xgboost"]:
    model = xgb.XGBClassifier(**best_hyperparams, num_class=3,
                              objective="multi:softprob", eval_metric="auc")
    
    model.fit(fit_X, fit_y, verbose=False, sample_weight=fit_sample_weight, sample_weight_eval_set=(fit_sample_weight, eval_sample_weight),
              eval_set=[(fit_X, fit_y), (eval_X, eval_y)])
else:
    model = saved_model["models"]["xgboost"]

#in case we load from an old model we need to reorder
trained_features = model.get_booster().feature_names
test_X = test_X.loc[:, trained_features]
train_X = train_X.loc[:, trained_features]


test_pred = model.predict(test_X)
train_pred = model.predict(train_X)

test_accuracy = accuracy_score(test_y, test_pred)
train_accuracy = accuracy_score(train_y, train_pred)

print('accuracy: ', train_accuracy, test_accuracy)


proba_test = model.predict_proba(test_X)
proba_train = model.predict_proba(train_X)

unanswered_trials = image_labels > 3
unanswered_df = overall_df.loc[unanswered_trials]
unanswered_y = [cat[2] for cat in unanswered_df.index]
unanswered_y = [int(y - 3)-1 for y in unanswered_y]
unanswered_df = unanswered_df.drop('subject_stratification', axis=1)

unanswered_df = unanswered_df.loc[:, trained_features]
unanswered_pred = model.predict(unanswered_df)
unanswered_accuracy = accuracy_score(unanswered_y, unanswered_pred)

print('unanswered accuracy: ', unanswered_accuracy)


error_trials = image_labels < 0
error_df = overall_df.loc[error_trials]
error_y = [cat[2] for cat in error_df.index]
error_y = [int(-y)-1 for y in error_y]
error_df = error_df.drop('subject_stratification', axis=1)

error_df = error_df.loc[:, trained_features]
error_pred = model.predict(error_df)
error_accuracy = accuracy_score(error_y, error_pred)

print('error accuracy: ', error_accuracy)


print('roc: ', roc_auc_score(train_y, proba_train, average='macro', multi_class='ovo'),
      roc_auc_score(test_y, proba_test, average='macro', multi_class='ovo'))


print('Removed number feature: ', sum(model.feature_importances_ == 0),
      'of', len(model.feature_importances_ == 0))

save = False
if override_old:
    summary = {'features': overall_df, 'trial_labels': image_labels, 
               'models': {'xgboost': model, 'hyperparameters':  used_hyperparams, 'train_data': train_X.index, 'test_data': test_X.index, 'unanswered_data': unanswered_df.index, 'error_data': error_df.index}
               }
    save = True
else:
    summary = saved_model
    
    
    
# #evaluate how much background dataset is needed


# probe_categories = []

# #calculate shap values with balanced background data from each participant
# for subject_ind, subject in enumerate(subjects):
    
#     subject_df = test_X.xs(subject, level="subject")
#     # Find the largest category count for this subject
#     target_n = min(subject_df.groupby(level="image_category").size())
    
#     if target_n > 50:
#         target_n=50

#     for category, category_df in subject_df.groupby(
#         level="image_category", sort=False):
#         # Sample with replacement only when this category is too small
#         sampled = category_df.sample(
#             n=target_n, random_state=42)
        
#         probe_categories.append(sampled)
        
# probe_dataset = pd.concat(probe_categories)    

# import time

# num_trials_p_s_c = 1

# while True:
    
#     balanced_categories = []
#     print(num_trials_p_s_c)
    
#     #calculate shap values with balanced background data from each participant
#     for subject_ind, subject in enumerate(subjects):
        
#         subject_df = train_X.xs(subject, level="subject")
#         # Find the largest category count for this subject
#         target_n = num_trials_p_s_c
    
#         for category, category_df in subject_df.groupby(
#             level="image_category", sort=False):
            
#             if len(category_df) >= target_n:
#                 # Sample with replacement only when this category is too small
#                 sampled = category_df.sample(
#                     n=target_n, random_state=42)
                
        
        
#             else:
#                 sampled = category_df.sample(
#                     n=target_n,
#                     replace=len(category_df) < target_n,
#                     random_state=42)
                
#             balanced_categories.append(sampled)
            

#     balanced_background = pd.concat(balanced_categories)    
          
    
#     start = time.perf_counter()

#     # calculate shap values
#     explainer = shap.TreeExplainer(model, data=balanced_background)
#     balanced_shap_test_values = np.array(explainer.shap_values(probe_dataset))#, check_additivity=False))

#     elapsed = time.perf_counter() - start
    
#     batch_filepath = os.path.join(savedir, str(num_trials_p_s_c) + '_check_background' + model_filename)
    
#     with open(batch_filepath, 'wb') as handle:
#         pickle.dump([balanced_shap_test_values, elapsed, balanced_background.shape], handle,)
        
#     num_trials_p_s_c = num_trials_p_s_c*2
        

#calculate shap values with balanced background data from each participant and image category
background_data = []


target_n = train_X.groupby(level=["subject", "image_category"]).size().max()

for subject_ind, subject in enumerate(subjects):
    
    #print(subject)
    
    subject_df = train_X.xs(subject, level="subject")
    # Find the largest category count for this subject
    target_n = max(subject_df.groupby(level="image_category").size())

    for category, category_df in subject_df.groupby(
        level="image_category", sort=False):
        # Sample with replacement only when this category is too small
        sampled = category_df.sample(
            n=target_n,
            replace=len(category_df) < target_n,
            random_state=42)
        background_data.append(sampled)

background_df = pd.concat(background_data)    
    
if not "test_shap_values" in summary["models"].keys():
    print('Calculate shap values on test data')
        
    # calculate shap values
    explainer = shap.TreeExplainer(model, data=background_df)
    summary["models"]["test_shap_values"] = np.array(explainer.shap_values(test_X))#, check_additivity=False))
    
    with open(os.path.join(savedir, 'real_data_' + model_filename), 'wb') as handle:
        pickle.dump(summary, handle,)

print('')
if not "unanswered_shap_values" in summary["models"].keys():
    print('Calculate shap values on unanswered data')
       
    # calculate shap values
    explainer = shap.TreeExplainer(model, data=background_df)
    summary["models"]["unanswered_shap_values"]  = np.array(explainer.shap_values(
        unanswered_df))#, check_additivity=False))
    
    with open(os.path.join(savedir, 'real_data_' + model_filename), 'wb') as handle:
        pickle.dump(summary, handle,)

print('')
    
if not "error_shap_values" in summary["models"].keys():
    print('Calculate shap values on error data')
    # calculate shap values
        
    explainer = shap.TreeExplainer(model, data=background_df)
    summary["models"]["error_shap_values"] = np.array(explainer.shap_values(
        error_df))#, check_additivity=False))
    
    with open(os.path.join(savedir, 'real_data_' + model_filename), 'wb') as handle:
        pickle.dump(summary, handle,)
    

print('')

print('Shuffle!')

shuffled_accuracies = []
shuffled_confusion_matrices = []

#first check how many iterations have been done
done_iterations = 0
done_batches = 0
exists = True

shuffles_per_batch = 8

while exists:
    batch_name = f'batch_{done_batches}_shuffle_shap_'
    
    batch_filepath = os.path.join(savedir, batch_name + model_filename)

    file_path = Path(batch_filepath)
    
    if file_path.exists():
        print('Batch', done_batches, 'exists')
        done_batches += 1

        with open(file_path, 'rb') as f:
            batch = pickle.load(f)
            for k in batch['permutated_confusion_mat']:
                shuffled_confusion_matrices.append(k)
                shuffled_accuracies.append(np.trace(k) / np.sum(k))
                
        done_iterations += shuffles_per_batch

    else:
        exists = False
            

#keep on from where it was left
#used to roll.
trial_subjects = correct_df.index.get_level_values('subject').to_numpy()
n = int(pd.Series(trial_subjects).value_counts().min())   # smallest patient
#n = train_X.shape[0]

random_shifts = []

num_batches = int(np.ceil((random_shuffles-done_iterations)/shuffles_per_batch))
num_shuffles = num_batches * shuffles_per_batch + done_iterations

if n-1 < num_shuffles:
    # make sure that each roll is sampled the same amount of times
    while len(random_shifts) < (num_shuffles - (n-1)):
        random_shifts = random_shifts + list(range(1, n))

# then pick more with minimal shifts to fill up
n_missing_values = num_shuffles - len(random_shifts)
random_shifts = random_shifts + \
    list(range(1, int(np.ceil((n_missing_values/2)+1))))
random_shifts = random_shifts + \
    list(range(n - int(np.ceil((n_missing_values/2))), n))


for curr_batch in range(num_batches):

    batch = done_batches + curr_batch
    iteration_start = done_iterations + curr_batch*shuffles_per_batch+1
    iteration_end = done_iterations + curr_batch *shuffles_per_batch + shuffles_per_batch+1
    
    batch_name = f'batch_{batch}_shuffle_shap_'
    print(batch_name)
    batch_filepath = os.path.join(savedir, batch_name + model_filename)

    file_path = Path(batch_filepath)

    if override_old or not file_path.exists():
        # do the shuffling in batche
        shuffled_shap_values = []
        batch_shuffled_confusion_matrices = []
        shifts = []
        shuffled_models = []

        shift_ind = range(iteration_start, iteration_end)

        print('Shuffle batch', batch, iteration_start,
              iteration_end)



        parallel_results = Parallel(n_jobs=shuffles_per_batch)(
            delayed(shuffle_models)(ind, random_shifts[ind]) for ind in shift_ind)
        # for ind in shift_ind:
        #     print(ind)
        #     parallel_results = [shuffle_models(ind, random_shifts[ind])]

        # organize outout

        for res in parallel_results:
            batch_shuffled_confusion_matrices.append(res[0])
            shuffled_confusion_matrices.append(res[0])

            shuffled_shap_values.append(res[1])

            # Compute the number of correct predictions
            correct_predictions = np.trace(res[0])  # Sum of diagonal elements

            # Compute the total number of predictions
            total_predictions = np.sum(res[0])
            
            shifts.append(res[2])
            
            shuffled_models.append(res[3])

            # Compute overall accuracy
            shuffled_accuracies.append(correct_predictions / total_predictions)

        shuffle_summary = {'shuffle_batch': batch, 'permutated_confusion_mat': batch_shuffled_confusion_matrices, 'shap_values': shuffled_shap_values, 'roll_shift': shifts, 'shuffled_models': shuffled_models}
        
        with open(batch_filepath, 'wb') as handle:
            pickle.dump(shuffle_summary , handle,)

print('Permuted 95% percentile accuracy: ',
      np.percentile(shuffled_accuracies, 95))


cm_shuffled = np.zeros((3, 3))
for k in shuffled_confusion_matrices:
    cm_shuffled += k


cm_test = confusion_matrix(test_y, test_pred)
cm_train = confusion_matrix(train_y, train_pred)
cm_error = confusion_matrix(error_y, error_pred)
cm_unanswered = confusion_matrix(unanswered_y, unanswered_pred)

# expektet unanswered based on behaviour and model distribution
# get distribution of correct/error trials for each category from the behaviour.
correct_responses = behaviour / np.sum(behaviour, axis=0)

# initiate
cm_expected = np.zeros((3, 3))

# loop categories
for cat in range(0, 3):
    # get distribution of correct trials for the category
    correct_distribution = correct_responses[0, cat]
    # get number of unanswered trials for that category
    num_unansweres_cat = sum([y == cat for y in unanswered_y])

    # how many correct trials do we expekt?
    expekted_correct_trials = num_unansweres_cat*correct_distribution
    # how does the correct model distribute images from that category
    distribution_correct = cm_test[cat, :] / sum(cm_test[cat, :])
    # distribute the presumed correct unanswered images
    cm_expected[cat, :] += distribution_correct*expekted_correct_trials

    # redo for error trials
    expekted_error_trials = num_unansweres_cat-expekted_correct_trials
    distribution_error = cm_error[cat, :] / sum(cm_error[cat, :])
    cm_expected[cat, :] += distribution_error*expekted_error_trials

cm_expected = cm_expected.astype(int)
expekted_accuracy = (
    cm_expected[0, 0] + cm_expected[1, 1] + cm_expected[2, 2]) / np.sum(cm_expected)


cms = [cm_test, cm_train, cm_error, cm_unanswered, cm_shuffled, cm_expected]
vmin = np.inf
vmax = -np.inf

for cm_ind, cm in enumerate(cms):
    cms[cm_ind] = cms[cm_ind] / np.sum(cm)
    if np.min(cms[cm_ind]) < vmin:
        vmin = np.min(cms[cm_ind])
    if np.max(cms[cm_ind]) > vmax:
        vmax = np.max(cms[cm_ind])


saved = False

# plot performance
f, axes = plt.subplots(2, 3)


while not saved:
    try:
        f, axes = plt.subplots(2, 3)

        axes[0, 0].imshow(cms[0], vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Correct: Test: ' +
                             str(int(np.round(test_accuracy*100))) + '%')
        axes[0, 0].set_xticks([])
        axes[0, 0].set_xlabel('')

        axes[1, 0].imshow(cms[1], vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Correct: Train: ' +
                             str(int(np.round(train_accuracy*100))) + '%')

        axes[0, 1].imshow(cms[2], vmin=vmin, vmax=vmax)
        axes[0, 1].set_title(
            'Error: ' + str(int(np.round(error_accuracy*100))) + '%')
        axes[0, 1].set_yticks([])
        axes[0, 1].set_ylabel('')
        axes[0, 1].set_xticks([])
        axes[0, 1].set_xlabel('')

        axes[0, 2].imshow(cms[3], vmin=vmin, vmax=vmax)
        axes[0, 2].set_yticks([])
        axes[0, 2].set_ylabel('')
        axes[0, 2].set_title(
            'Unanswered: ' + str(int(np.round(unanswered_accuracy*100))) + '%')
        axes[0, 2].set_xticks([])
        axes[0, 2].set_xlabel('')

        axes[1, 2].imshow(cms[5], vmin=vmin, vmax=vmax)
        axes[1, 2].set_title(
            'Exp Unans: ' + str(int(np.round(expekted_accuracy*100))) + '%')
        axes[1, 2].set_yticks([])
        axes[1, 2].set_ylabel('')

        axes[1, 1].imshow(cms[4], vmin=vmin, vmax=vmax)
        axes[1, 1].set_title(
            'Chance: ' + str(int(np.round(np.mean(shuffled_accuracies)*100))) + '%')
        axes[1, 1].set_yticks([])
        axes[1, 1].set_ylabel('')

        plt.savefig(os.path.join(savedir, figure_name))
        plt.show()

        saved = True
    except:
        print('Error: not saved')

log.close()