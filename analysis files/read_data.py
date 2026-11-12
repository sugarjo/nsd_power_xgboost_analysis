import numpy as np
import pandas as pd 

import feature_setup
#from helper_functions import feature_setup

import pickle
import re
from pathlib import Path

from neural_decoding.utils import get_ephys_meta
from neural_decoding.utils import load_db

import os
import glob
from scipy.stats import norm


def load_data(input_dir, field, button_limit, trial_dir=""):
    
    trial_shape = [2,254]
    
    if not type(trial_shape) == list:
        raise TypeError("trial_shape must be a list of integers")
    # reading the data file of the patient

    #check that it exists
    if  os.path.isfile(input_dir + "_" + field):
        print('Precomputed file exists!')
        fft_trials = pd.read_pickle(input_dir + "_" + field)    
    else:
        #load original file
        print('Create file from trials.pkl!')
        original_trials = pd.read_pickle(Path(input_dir).with_name("trials.pkl"))  

        #calculate the bipolar
        if "bipolar" in Path(input_dir).parts[-1]:
            original_trials["bipolar_ephys"] = [[] for _ in range(len(original_trials))]
            original_trials["bipolar_pre_trial_ephys"] = [[] for _ in range(len(original_trials))]
            original_trials["bipolar_channels"] = [[] for _ in range(len(original_trials))]

            for t in range(len(original_trials)):

                bipolar_channels = []
                bipolar_ephys = []
                bipolar_pre_trial_ephys = []

                #loop the channels
                for c_ind, c in enumerate(original_trials.iloc[t].channels):
                    
                    current_channel = get_ephys_meta.standardized_channel_name(c)              
                    probe1, el1 = get_ephys_meta.get_probe_channel(current_channel)

                    #loop to find the neighbour
                    matched = 0
                    for c_ind2, c2 in enumerate(original_trials.iloc[t].channels):
                        next_channel = get_ephys_meta.standardized_channel_name(c2)              
                        probe2, el2 = get_ephys_meta.get_probe_channel(next_channel)

                        if probe1==probe2 and el1+1 == el2:

                            matched += 1
                            bi_ephys = original_trials.iloc[t].ephys[:, c_ind] - original_trials.iloc[t].ephys[:, c_ind2]                        
                            bi_pre_trial_ephys = original_trials.iloc[t].pre_trial_ephys[:, c_ind] - original_trials.iloc[t].pre_trial_ephys[:, c_ind2]    
                            bi_channel = original_trials.iloc[t].channels[c_ind] + " - " +  original_trials.iloc[t].channels[c_ind2]    

                            bipolar_channels.append(bi_channel)
                            bipolar_ephys.append(bi_ephys)
                            bipolar_pre_trial_ephys.append(bi_pre_trial_ephys)
                    
                    #to avoid of printing for each trial
                    if t == 0:
                        if matched == 0:
                            print(c_ind, 'Did not fint bipolar', original_trials.iloc[t].channels[c_ind])
                        if matched > 1:
                            print(c_ind, 'Matched', matched, 'times', original_trials.iloc[t].channels[c_ind])
                   

                original_trials.at[original_trials.index[t], "bipolar_ephys"] = np.ma.vstack(bipolar_ephys).T.copy()
                original_trials.at[original_trials.index[t], "bipolar_pre_trial_ephys"] = np.ma.vstack(bipolar_pre_trial_ephys).T.copy()
                original_trials.at[original_trials.index[t], "bipolar_channels"] = bipolar_channels.copy()


            trials = original_trials[['image_ID', 'clock_time', 'sample_rate', 'last_response',
                'last_motor_response', 'indices_all_responses', 'bipolar_ephys', 'bipolar_channels', "bipolar_pre_trial_ephys"]]

            trials = trials.rename(columns={"bipolar_ephys": "ephys"})
            trials = trials.rename(columns={"bipolar_pre_trial_ephys": "pre_trial_ephys"})
            trials = trials.rename(columns={"bipolar_channels": "channels"})

        trials = trials.sort_values(by='clock_time')
        trials['repetition_num'] = trials.groupby('image_ID').cumcount()

        #get time since last presentation
        last_pres = []
        for t in range(len(trials)):
            i_id = trials.iloc[t].image_ID
            ind = np.where(trials.iloc[:t].image_ID == i_id)[0]

            if len(ind) == 0:
                last_pres.append(0)
            else:
                last_pres.append(trials.iloc[t].clock_time - trials.iloc[ind[-1]].clock_time)



        trials['time_since_last_presentation'] = last_pres
    
        #check if baseline_corrected. if not calculate ffts and baseline correct! each fft has the shape of sec, freq, channels
        if not field in trials.columns:    
            fft_trials = feature_setup.baseline_correct(trials, field, input_dir)
    
    fft_trials = fft_trials.sort_values(by='clock_time')

    #take channel names from a random channel
    channels = fft_trials['channels'][0]
    
    fs= int(fft_trials.sample_rate.loc[0])
    #get trials with premature presses. only consider positive trials that are within the trial window.
    premature_presses = fft_trials.indices_all_responses.apply(
            lambda x: False
            if len([v for v in x if v > 0]) == 0
            else min(v for v in x if v > 0) < fs * button_limit
        )

    # extracting the needed data
    # data that are not control trials
    relevant = (fft_trials['last_response'] == 'old') | (fft_trials['last_response'] == 'new')  | (fft_trials['last_response'] == ' ')
    print(sum(relevant), ' relevant trials')

    #load the interictal exclusion file
    if len(trial_dir) > 0:
        try:
            with open(trial_dir, 'rb') as f:
                trial_splits = pickle.load(f).astype(bool)            
            
            if trial_splits.ndim == 1:
                relevant = (relevant) & (trial_splits)
            else:
                relevant = (relevant) & (trial_splits[:,0])
            print(sum(relevant), 'after removal of interictal trials')
        except:
            print('No file with interictal indices')
            
         
    relevant_trials = fft_trials[relevant & ~premature_presses]   
    print(sum(relevant & ~premature_presses), 'after removal of premature button presses')

    pic = relevant_trials['image_ID'].values
    presentation_time = relevant_trials.clock_time.values
    responses = relevant_trials.last_response.values 
    repetitions = relevant_trials.repetition_num.values 
    time_last_presentation = relevant_trials.time_since_last_presentation

    dataset_shape = [relevant_trials.shape[0], len(channels)]
    dataset_shape.extend(trial_shape)
    #allocating empty matrix
    #shape: trials, channels, time and freq
    data = np.zeros(dataset_shape)

    #loop trials
    for i in range(relevant_trials.shape[0]):
        #loop channels
        for j in range(len(channels)):
            #calculate from scratch
            if field == 'ephys':
                signal = relevant_trials['ephys'].values[i][:, j]
                #from beginining of trial
                s=0
                #fft of the first second
                fft = np.fft.fft(signal[s:s+fs])[:254]
                power = np.abs(fft)**2
                data[i, j, 0, :] = power
                
                #fft of the second second
                fft = np.fft.fft(signal[s+fs:s+2*fs])[:254]
                power = np.abs(fft)**2
                data[i, j, 1, :] = power
                
                fft_freq = np.fft.fftfreq(fs, 1/fs)[:254]
                
            else:
                data[i, j, :, :] = relevant_trials[field].values[i][:, :, j]
                fft_freq = np.linspace(1, 254, 254)
                
                
            #else:
                #raise ValueError("Shape of individual trials is neither 2D or 3D")
    return data, fft_freq, channels, [pic, presentation_time, responses, repetitions, time_last_presentation]


def morlet_convolve(fs, cycles, f0, signal):
     # sampling rate in Hz
    # target frequency in Hz
   # wavelet width
    
    # Temporal standard deviation of the Gaussian envelope
    sigma_t = n_cycles / (2 * np.pi * f0)
    
    # Make the wavelet long enough to capture its Gaussian tails
    half_duration = cycles/2 * sigma_t
    t = np.arange(-half_duration, half_duration + 1/fs, 1/fs)
    
    # Complex Morlet wavelet
    wavelet = np.exp(2j * np.pi * f0 * t) * np.exp(-(t**2) / (2 * sigma_t**2))
    
    # signal: your 1D NumPy array sampled at 512 Hz
    # convolution result has the same length as signal
    tf_complex = fftconvolve(signal, wavelet, mode="same")
    
    return tf_complex




def load_morlet_data(input_dir, extract_freq, button_limit, trial_dir=""):
    

    # reading the data file of the patient
    trials = pd.read_pickle(input_dir + '.pkl')    
    
    trials['repetition_num'] = trials.groupby('image_ID').cumcount()
        
    #check if baseline_corrected. if not do it!
    if not extract_freq in trials.columns:    
        trials = feature_setup.convolve(trials, extract_freq, input_dir)
    
    trials=trials.sort_values(by='clock_time')

    #take channel names from a random channel
    channels = trials['channels'][0]
    
    fs=trials.sample_rate.loc[0]
    #get trials with premature presses
    premature_presses = trials.indices_all_responses.apply(lambda x: False if len(x) == 0 else min(x) < fs*button_limit)

    # extracting the needed data
    # data that are not control trials
    relevant = (trials['last_response'] == 'old') | (trials['last_response'] == 'new')  | (trials['last_response'] == ' ')
    print(sum(relevant), ' relevant trials')
    
    if len(trial_dir) > 0:
        try:
            with open(trial_dir, 'rb') as f:
                trial_splits = pickle.load(f).astype(bool)            
            
            if trial_splits.ndim == 1:
                relevant = (relevant) & (trial_splits)
            else:
                relevant = (relevant) & (trial_splits[:,0])
            print(sum(relevant), 'after removal of interictal trials')
        except:
            print('No file with interictal indices')
            
         
    relevant_trials = trials[relevant & ~premature_presses]   
    print(sum(relevant & ~premature_presses), 'after removal of premature button presses')

    pic = relevant_trials['image_ID'].values
    presentation_time = relevant_trials.clock_time.values
    responses = relevant_trials.last_response.values 
    repetitions = relevant_trials.repetition_num.values 

    dataset_shape = [relevant_trials.shape[0], len(channels)]
    #dataset_shape.extend(trial_shape)
    #allocating empty matrix
    data = np.zeros(dataset_shape)

    #loop trials
    for i in range(relevant_trials.shape[0]):
        #loop channels
        for j in range(len(channels)):              
                
                
            pre_trial_signal = trials["pre_trial_ephys"].iloc[i][j]
            trial_signal = trials["ephys"].iloc[i][j]
            
            signal = pre_trial_signal + trial_signal
            
            convolved_signal = morlet_convolve(fs, cycles, f0, signal)
                
                
                
            signal = relevant_trials['ephys'].values[i][:, j]
            #quarter of a sec
            s=128
            #fft of the first second
            fft = np.fft.fft(signal[s:s+512])[:254]
            power = np.abs(fft)**2
            data[i, j, 0, :] = power
            
            #fft of the second second
            fft = np.fft.fft(signal[s+512:s+2*512])[:254]
            power = np.abs(fft)**2
            data[i, j, 1, :] = power
            
            fft_freq = np.fft.fftfreq(512, 1/512)[:254]
            
                
            #else:
                #raise ValueError("Shape of individual trials is neither 2D or 3D")
    return data, fft_freq, channels, [pic, presentation_time, responses, repetitions]




def clean_dataset(data, channels, method='median', threshold=False):
    
    channels_to_remove = []
    
    num_nans = 0
    num_infs = 0
    num_zeros = 0
    
    original_data_size = data[:, 0].size
    
    #if not given set treshold to half of the trials
    if not threshold:
        threshold = int(np.floor(data[:, 0].size/2))
        
    #finding the features with zeros, nans or infs
    #loop channels
    for i in range (data.shape[1]):
        nans = np.isnan(data[:, i])
        infs = np.isinf(data[:, i])
        zeros =  data[:, i] == 0
        
        num_nans += np.sum(nans)
        num_infs += np.sum(infs)
        num_zeros += np.sum(zeros)
        
        #print((np.sum(nans) + np.sum(infs) + np.sum(zeros)))
        
        #delete if above treshold
        if (np.sum(nans) + np.sum(infs) + np.sum(zeros)) >= threshold:
            channels_to_remove.append(i)
        
        #insert median/mean values for nans
        else:
            replace_bool =np.logical_or.reduce((nans, infs, zeros))
            
            if np.sum(replace_bool) == 0:
                continue
            #insert values instead of nans. 
            #or median to avoid outlayers?
            if method == 'median':
                data[:, i][replace_bool] = np.median(data[:, i][~replace_bool])
            elif method == 'nan':
                data[:, i][replace_bool] = np.nan
                
    if method == 'median':
        print('Interpolated nans: ', (num_nans)/data.size)
        print('Interpolated zeros: ', (num_zeros)/data.size)
        print('Interpolated infs: ', (num_infs)/data.size)
    elif method == 'nan':
        print('Kept nans: ', (num_nans)/data.size)
        print('Zeros to nans: ', (num_zeros)/data.size)
        print('Infs to nans: ', (num_infs)/data.size)

    # removing features with a lot of nans
    print('Removed channels becaus of Nans: ', len(channels_to_remove))
    for ch in sorted(channels_to_remove, reverse=True):
        del channels[ch]
    
    data = np.delete(data, channels_to_remove, axis=1)
    
    return data, channels

def load_ephys_data(input_dir):
        
    trial_shape = [512*3]
    
    if not type(trial_shape) == list:
        raise TypeError("trial_shape must be a list of integers")
    # reading the data file of the patient
    trials = pd.read_pickle(input_dir + '.pkl')
    trials=trials.sort_values(by='clock_time')

    #take channel names from a random channel
    channels = trials['channels'][0]

    # extracting the needed data
    # data that are not control trials
    relevant_trials = trials[(trials['last_response'] == 'old') | (trials['last_response'] == 'new')  | (trials['last_response'] == ' ')]
    
    pic = relevant_trials['image_ID'].values
    presentation_time = relevant_trials.clock_time.values
    responses = relevant_trials.last_response.values 

    dataset_shape = [relevant_trials.shape[0], len(channels)]
    dataset_shape.extend(trial_shape)
    #allocating empty matrix
    data = np.zeros(dataset_shape)

    #loop trials
    for i in range (relevant_trials.shape[0]):
        #loop channels
        for j in range(len(channels)):

            ephys = relevant_trials['ephys'].values[i][:, j]
            #quarter of a sec
            #fft of the first second
            signal = ephys[:512*3]

            data[i, j, :] = signal
                

    return data, channels, [pic, presentation_time, responses]