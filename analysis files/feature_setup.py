import os
from pathlib import Path
import numpy as np
import pandas as pd 
import pickle
from sklearn.utils.class_weight import compute_class_weight
from neural_decoding.utils.analyze_timeseries import fft_analysis
from scipy import signal

from neural_decoding.utils import anatomy
from neural_decoding.utils import get_ephys_meta
import re   
from collections import OrderedDict

def load_data(
    subject: str,
    filename: str,
    root: str | Path = Path("//hypatia.uio.no/lh-med-imb-cnp/data_seeg/"),
    folder: str = "analysis_natural_scenes_task"
):
    """
    Loads LFP data from specified file.

    Args:
        subject (str): Subject identifier.
        filename (str): Name of the file to be loaded.
        root (str | Path, optional): Base path for the data. Default is the given path.
        folder (str, optional): Subfolder for analysis. Default is the given path.

    Returns:
        Outfile contents (typically a dictionary or other data structure).
    """
    try:
        if type(root) == str:
            root = Path(root)
        file_path = root / subject / folder / filename
        with open(file_path, 'rb') as file:
            outfile = pickle.load(file)
    except Exception as e:
        print(f"Error Subject {subject} - Loading files: {str(e)}")
    return outfile


def data_labelling(pic, presentation_time, responses, time_last_presentation):
    
    last_trial_time = 0
    curr_group = 0
    #preallocate with label and group
    yreal = np.zeros((pic.shape[0], 2))
    for i in range(0 , pic.shape[0]):
        if pic[i] == 0:
            if (responses[i] == 'baseline'):
                label = 100 # baseline, accidental button press
            elif (responses[i] == 'mock'):
                label = 101 # baseline, no motor response
        #seen before
        elif time_last_presentation[i] > 0:

            #more than 3 hour since shown?
            if time_last_presentation[i] > 60*60*3:         
                label = 3 #old
            else:
                label = 2 #recently shown            
            
            #relabel if wrong
            if responses[i] == 'new':
                label = -label
            elif responses[i] == 'old':
                label = label
            elif responses[i] == ' ':
                # label = -label
                label = label+3
            else:
                label = 404
        #new
        else:
            if responses[i] == 'new':
                label = 1 #new correct
            elif responses[i] == 'old':
                label = -1 #new wrong
            elif responses[i] == ' ':
                label = 4 #unanswered
            else:
                label = 404

        #switch recording session      
        if presentation_time[i] - last_trial_time > 60*60*3:
            curr_group += 10
        
        last_trial_time = presentation_time[i]

        yreal[i, 0] = label
        yreal[i, 1] = curr_group
    return yreal


def time_bins(timeseries_data):
    # deviding the time series data into time bins
    size = 512
    s = 128
    fdata = np.zeros((timeseries_data.shape[0] , timeseries_data.shape[1] , 2 , size))
    fdata[: , : , 0 , :] = timeseries_data[: , : , s:s+size]
    fdata[: , : , 1 , :] = timeseries_data[: , : , s+size:s+2*size]
    return fdata


def area_median(xdata, subject, features, bipolar, glasser_areas):
    # merge across glasser areas
    # anatomy folder
    coreg_chans = anatomy.get_brain_areas(
        subject, hipp=True, atlas='Glasser', lobes='Glasser')

    # get brain area per feature
    freqs = []
    areas = []

    wm_areas = ['WM-hypointensities',
    'Right-Cerebral-White-Matter',
    'Left-Cerebral-White-Matter']

    for f in features:
        ch, freq = f.split(" | ")
        freqs.append(freq)

        if bipolar:
            ch1 = ch.split(" - ", 1)[0].strip()
            selected = coreg_chans.Probe == get_ephys_meta.standardized_channel_name(ch1)
            destrieux_label1 = coreg_chans[selected].labels.iloc[0]
            glasser_label1 = coreg_chans[selected].Glasser_labels.iloc[0]

            ch2 = ch.split(" - ", 1)[1].strip()
            selected2 = coreg_chans.Probe == get_ephys_meta.standardized_channel_name(ch2)
            destrieux_label2 = coreg_chans[selected2].labels.iloc[0]
            glasser_label2 = coreg_chans[selected2].Glasser_labels.iloc[0]


        else:
            selected = coreg_chans.Probe == get_ephys_meta.standardized_channel_name(ch)
            destrieux_label1 = coreg_chans[selected].labels.iloc[0]
            glasser_label1 = coreg_chans[selected].Glasser_labels.iloc[0]
            #set it to something to make it work
            destrieux_label2 = coreg_chans[selected].labels.iloc[0]
            glasser_label2 = coreg_chans[selected].Glasser_labels.iloc[0]

        #check
        if not sum(selected) == 1:
            print('Error', ch)

        #first check if subcortical
        if 'Amygdala' in destrieux_label1 or  'Amygdala' in destrieux_label2:
            area = 'Amygdala'
        elif 'Hippocampus' in destrieux_label1 or 'Hippocampus' in destrieux_label2:
            if 'Hippocampus' in destrieux_label1:
                AP_coordinate = coreg_chans[selected].y_vol.iloc[0]
            else:
                AP_coordinate = coreg_chans[selected2].y_vol.iloc[0]
                #area = 'Hippocampus'                
            if AP_coordinate > -23:
                area = 'Hippocampus-anterior'
            else:
                area = 'Hippocampus-posterior'   
        elif 'Putamen' in destrieux_label1 or 'Putamen' in destrieux_label2:
            area = 'Putamen'
        # Commented because one can be in whitematter and have unknown glasser label 
        # elif 'Unknown' in glasser_label1 or 'Unknown' in glasser_label2 or 'unknown' in glasser_label1 or 'unknown' in glasser_label2:
        #     area = ""   
        #if none; then it is in grey
        #check if electrode 1 is in white matter
        elif destrieux_label1 in wm_areas:
            parts = glasser_label2.split("_")
    
            # Extracting the part(s) between the underscores
            if len(parts) > 1:
                area = parts[1]  # This gets the part between the first and second underscore
            else:
                if bipolar:
                    print('Could not parse', coreg_chans[selected2].Glasser_labels)
                else:
                    print('Could not parse', coreg_chans[selected].Glasser_labels)
                area = ""
        else:
            parts = glasser_label1.split("_")
    
            # Extracting the part(s) between the underscores
            if len(parts) > 1:
                area = parts[1]  # This gets the part between the first and second underscore
            else:
                print('Could not parse', coreg_chans[selected].Glasser_labels)
                area = ""
        
        areas.append(area)


    # Initialize a dictionary to hold the sum of cortex_data and the count of occurrences
    area_data = {}
    c = 0
    check_indices = []

    # Loop through the unique_freqs
    for a in glasser_areas:
        for f in np.unique(freqs):
            indices = [i for i, val in enumerate(
                areas) if val == str(a) and freqs[i] == f]

            if len(indices) == 0:
                # insert nan data if area is not covered
                averaged_data = np.full(xdata.shape[0], np.nan)
            else:
                # use this to double check that we get all the data

                for k in indices:
                    check_indices.append(k)

                # median across electrodes within ROI
                # this will give a runtime warning if nan for all features in a trial
                averaged_data = np.nanmedian(xdata[:, indices], axis=1)

                if sum(np.isnan(averaged_data)) > len(averaged_data)/2:
                    print('ERROR MORE THAN HALF OF DATA IS NAN')

            area_data[str(a) + ' | ' + f] = averaged_data

    if not len(check_indices) == xdata.shape[1]:
        print('ERROR: Featured covered more than once')

    # turn into dataframe
    area_data_df = pd.DataFrame(area_data)

    return area_data_df


def remove_ch(subject, channels, data, bipolar, save=True, *bands, **savedir):

    if not bands:
        #later output is shaped: trials x features, where the features are ordered: bands, seconds and channels
        bands = ['Delta1' , 'Theta1' , 'Alpha1' , 'Beta1' , 'GammaL1' , 'GammaH1' , 'Ripple1', 'Delta2' , 'Theta2' , 'Alpha2' , 'Beta2' , 'GammaL2' , 'GammaH2', 'Ripple2']
    else:
        bands = bands[0]

    exclude_areas = ['Unknown',
                     'unknown',
                     'Left-Lateral Ventricle',
                     'Right-Lateral Ventricle',
                     'Left-choroid-plexus',
                     'Right-choroid-plexus',
                     'Right-Inf-Lat-Vent',
                     'Left-Inf-Lat-Vent',
                     'Right-Lateral-Ventricle',
                     'Left-Lateral-Ventricle']
    
    wm_areas = ['WM-hypointensities',
                'Right-Cerebral-White-Matter',
                'Left-Cerebral-White-Matter']

    
    coreg_chans = anatomy.get_brain_areas(subject, hipp=True, atlas = 'Glasser', lobes='Glasser')

    if not 'Lesion' in coreg_chans:
        print('Have no information about electrodes in lesioned areas. Assume all are healthy.')
    if not 'Noicy' in coreg_chans:
        print('Have no information about noicy channels. Assume all are healthy.')    
        
    chanlocs = coreg_chans['labels'].values
    gm_dels = []
    wm_dels = []
    
    print(len(coreg_chans), ' coregistered channels')
    
    noicy_count = 0
    lesioned_count = 0
    excluded_count = 0
    
    #get region of each channels and those that should be deleted based on the colocalization
    for i in range(len(channels)):

        if bipolar:
            unipolar_channel = channels[i].split(" - ", 1)[0].strip()
            probe, channel = get_ephys_meta.get_probe_channel(unipolar_channel)
        else:
            probe, channel = get_ephys_meta.get_probe_channel(channels[i])   

        #MATCH THE ELECTRODE
        #count the number of times matched
        matched = 0

        #loop the electrode cheet to find the same
        for j in range(len(coreg_chans)):
            
            #this comes from the excel sheet
            coreg_probe, coreg_channel = get_ephys_meta.get_probe_channel(coreg_chans['Probe'][j])

            #NB: make sure no probes have same names in the two hemispheres
            if coreg_probe == probe and channel == coreg_channel:
                #print(coreg_probe, i, j)
                
                #GET NEIGHBOUR
                bipolar_matched = False
                #loop them again to find the neighbour
                for k in range(len(coreg_chans)):
                    neighbour_coreg_probe, neighbour_coreg_channel = get_ephys_meta.get_probe_channel(coreg_chans['Probe'][k])
                    
                    if neighbour_coreg_probe == coreg_probe and neighbour_coreg_channel == coreg_channel+1 and not bipolar_matched:
                        bipolar_matched = True
                        bip_coreg_ind = k
                    elif neighbour_coreg_probe == coreg_probe and neighbour_coreg_channel == coreg_channel+1 and bipolar_matched:
                        print('Error bipolar already matched', i, j)
                          
                neighbour_coreg_probe, neighbour_coreg_channel = get_ephys_meta.get_probe_channel(coreg_chans['Probe'][bip_coreg_ind])
                
                if bipolar and bipolar_matched:
                    matched += 1     
                    if not ((probe == neighbour_coreg_probe) and (neighbour_coreg_channel == channel+1)):
                        print('ERROR', probe, channel, 'with', neighbour_coreg_probe, neighbour_coreg_channel)
                    



                elif bipolar:
                    print('Not matched', probe, channel)
                    gm_dels.append(i)
                    wm_dels.append(i)
                    continue
                    
                elif not bipolar:
                    #if referential set index to be identical to j so that rest of code works
                    bip_coreg_ind = j
                    matched += 1   
                else:
                    print('This should not happen')
                    gm_dels.append(i)
                    wm_dels.append(i)
                    continue
                
                #check if lesioned
                lesioned = False
                #if column exist
                if 'Lesion' in coreg_chans:
                    #if either are in lesion
                    if coreg_chans['Lesion'][j] or coreg_chans['Lesion'][bip_coreg_ind]:
                        lesioned = True
                        lesioned_count += 1
                
                noicy = False
                #if either are noicy
                if 'Noicy' in coreg_chans:
                    if coreg_chans['Noicy'][j] or coreg_chans['Noicy'][bip_coreg_ind]:
                        noicy = True
                        noicy_count += 1
                        
                excluded_area = False
                #if either are in excluded
                if chanlocs[j] in exclude_areas or chanlocs[bip_coreg_ind] in exclude_areas:
                    excluded_area = True
                    excluded_count += 1

                #exclude
                if excluded_area or noicy or lesioned:
                    gm_dels.append(i)
                    wm_dels.append(i)
                #if both are in white
                elif (chanlocs[j] in wm_areas) and (chanlocs[bip_coreg_ind] in wm_areas):
                    gm_dels.append(i)
                #at least one is in grey, so includ
                else:
                    wm_dels.append(i)
                    
        #check if some electrodes are not matched or matched multiple times
        if not matched==1:
            if matched == 0:
                #channels are useless if not matched
                gm_dels.append(i)
                wm_dels.append(i)
                print(i, channels[i], ' coregistered ' + str(matched) + ' times and deleted!')
            else:
                print(i, channels[i], ' coregistered ' + str(matched) + ' times and is kept!.')
    
    print(excluded_count, ' channels are in exclusion areas and are removed.')
    print(lesioned_count, ' channels are labeled as in lesioned area and are removed.')
    print(noicy_count, ' channels are labeled as noicy and are removed.')
    
    #delete data from lists    
    gm_data = np.delete(data, gm_dels, axis=1)
    wm_data = np.delete(data, wm_dels, axis=1)
    
    gm_channels = np.delete(channels, gm_dels)
    
    #get the lobes
    gm_glasser_lobes = []
    for ch in gm_channels:
        
        if bipolar:

            unipolar_ch =  ch.split(" - ", 1)[0].strip()

            channel = get_ephys_meta.standardized_channel_name(unipolar_ch)
            selected = coreg_chans.Probe == channel

            neighbour_channel = ch.split(" - ", 1)[1].strip()
            channel2 = get_ephys_meta.standardized_channel_name(neighbour_channel)
            selected_neighbour = coreg_chans.Probe == channel2
        else:
            channel = get_ephys_meta.standardized_channel_name(ch)
            selected = coreg_chans.Probe == channel
            selected_neighbour = selected
        
        
        if sum(selected) == 1 and sum(selected_neighbour)==1:
            
            lobe1 = coreg_chans['brain_lobe'][selected].iloc[0]
            lobe2 = coreg_chans['brain_lobe'][selected_neighbour].iloc[0]
            
            if lobe1 in wm_areas:
                gm_glasser_lobes.append(lobe2)
            else:
                gm_glasser_lobes.append(lobe1)
                
            if coreg_chans['brain_lobe'][selected].iloc[0] == 'unknown' or coreg_chans['brain_lobe'][selected_neighbour].iloc[0] == 'unknown':
                print(get_ephys_meta.standardized_channel_name(ch), ' is Glasser unknown')
            
        else:
            print('ERROR: wrong electrodes found', ch)
    
    unique_lobes = np.unique(gm_glasser_lobes)
    gm_lobe_weights = compute_class_weight('balanced', classes=unique_lobes, y=gm_glasser_lobes)
      
    #create feature names
    gm_features = []
    gm_feature_weights = []
    
    #loops channels
    for i in range(len(gm_channels)):
        
        lobe_index = gm_glasser_lobes[i] == unique_lobes
        lobe_weight = gm_lobe_weights[lobe_index][0]
        
        #loops to 6
        for j in range(len(bands)):
            
            s = gm_channels[i] + " | "+ bands[j]
            gm_features.append(s)
            gm_feature_weights.append(lobe_weight)
            
    wm_channels = np.delete(channels, wm_dels)
    #create feature names
    wm_features = []
    for i in range (len(wm_channels)):
        for j in range (len(bands)):
            s = wm_channels[i] + " | "+ bands[j]
            wm_features.append(s)
    
    
    #save feature names for later
    if save:
        if not savedir:
            hypatia_path = os.environ["CNP_HYPATIA"]
            savedir = os.path.join(hypatia_path, "data_seeg", subject, "analysis_natural_scenes_task", "SVM")
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        np.savetxt(savedir + "/feature-names.csv", np.array(gm_features).astype(object) , fmt='%s', delimiter= ',')
        gm_features = np.genfromtxt(savedir + "/feature-names.csv" , dtype = str , delimiter=',').tolist()
        
    return gm_data, gm_features, wm_data, wm_features, gm_feature_weights


def compute_band_means(fft, frequency_ranges, freq_index, extract_field):
 # Initialize empty numpy array for result
    xdata = np.zeros((fft.shape[0], fft.shape[1], fft.shape[2], len(frequency_ranges)))
    
    for i, frange in enumerate(frequency_ranges):
        #print(i)
        # Extract indexes for each frequency range
        freq_indexes = np.where((frange[0] < freq_index) & (freq_index <= frange[1]))[0]
        
        # Take the mean of the relevant elements in the baseline_fft array and add to xdata_base
        #log transform if not baseline corrected ffts
        if extract_field == 'ephys':
            
            xdata[: , :  , :  , i] = np.mean(np.log(fft[: , : , : , freq_indexes] + 1e-10), axis = 3)
            
        else:
            #can give mean slice nan warning if the fft for a particular trial is nan for a channel
            xdata[: , :  , :  , i] = np.nanmean(fft[: , : , : , freq_indexes] , axis = 3)
        
    return xdata


def replace_naninf(data, features, method='median', threshold=False):
    z = []
    
    #if not given set treshold to half of the trials
    if not threshold:
        threshold = int(np.floor(len(data)/2))
        
    numbers_changed = 0

    #finding the features with zeros
    #loop channels
    for i in range (data.shape[1]):
        num_inf_nan = np.count_nonzero(np.logical_or(~np.isfinite(data[:,i]), data[:,i] == 0))
        
        numbers_changed  += num_inf_nan
        #delete if above treshold
        if num_inf_nan >= threshold:
            z.append(i)
        #insert median/mean values
        else:
            inf_nan_idx = np.argwhere(np.logical_or(~np.isfinite(data[:,i]), data[:,i] == 0))
            
            finite = np.argwhere(~(np.logical_or(~np.isfinite(data[:,i]), data[:,i] == 0)))
            
            if len(inf_nan_idx) == 0:
                continue
            #insert values instead of nans. 
            #or median to avoid outlayers?
            if method == 'median':
                data[inf_nan_idx, i] = np.nanmedian(data[finite, i])
            elif method == 'nan':
                data[inf_nan_idx, i] = np.nan

    # removing features with alot of zeros
    for i in range(len(z)):
        m = len(z)-i-1
        data = np.delete(data , z[m] , axis=1)
        del features[z[m]]
        
    print('Removed or interpolated values: ', numbers_changed/data.size)

    return data, features


def trial_removal(xdata, yreal, wm_xdata, criterion):
    # remove trials with features above criterion std

    # input is shaped: trials, channels, seconds, bands in ascending order
    xdata = np.reshape(xdata , (xdata.shape[0] , 7*xdata.shape[1]*xdata.shape[2]))
    #output is shaped: trials x features, where the features are ordered: bands, seconds and channels
    #e.g. trial1, 
    
    wm_xdata = np.reshape(wm_xdata , (wm_xdata.shape[0] , 7*wm_xdata.shape[1]*wm_xdata.shape[2]))

    #mean/std for each feature (across trials)
    mean = np.nanmean(xdata, axis=0)
    std = np.nanstd(xdata, axis=0)
    zs = (xdata - mean) / std
    #initiate
    # meanzs = np.zeros((xdata.shape[0] , int(xdata.shape[1])))
    # #loop the features
    # for i in range(meanzs.shape[1]):
    #     #for each feature calculate the mean z-score
    #     meanzs[: , i] = np.nanmean(zs[:,i*1:(i+1)*1] , axis = 1)


    badts = []
    #loop the trials. if any of the features is above the criterion sd. remove it.
    for i in range (xdata.shape[0]):
        if (np.any(np.abs(zs[i,:])>criterion)):
            badts.append(i)
            
    noicy_trials = np.zeros(xdata.shape[0], dtype=bool)
    noicy_trials[badts] = True
    
    #enable if we want to remove bad trials
    xdata = np.delete(xdata , badts , axis = 0)
    yreal = np.delete(yreal , badts , axis = 0)
    
    wm_xdata = np.delete(wm_xdata , badts , axis=0)
    
    return xdata, yreal, wm_xdata, noicy_trials


def baseline_correct(trial_file, correction_type, input_dir):
    
    num_baseline_nans = 0
    
    if correction_type == 'session_corrected_fft':
        string_save = 'session_corrected_fft'
        time_jump = 3600
        session_correction = True
        print('Perform session correction')
        
    elif correction_type == 'block_corrected_fft':
        string_save = 'block_corrected_fft'
        time_jump = 10
        session_correction = False
        print('Perform block correction')
        
        
    #trial_file = file.load_subject_ephys(subj, "analysis_natural_scenes_task")
    #block_idx = np.floor(trial_file["last_response_ind"].keys()/75).astype(int)

    baseline = trial_file["last_response"] == "baseline"
    baseline_trials = trial_file[baseline]

    mock = trial_file["last_response"] == "mock"
    mock_trials = trial_file[mock]
    
    #not needed nan detection taken care of down stream
    
    # nan_detection = np.vstack(trial_file['ephys'])
    # nan_mask = file.ch_nan_filter(nan_detection, 10000, row_nan=False)[1]

    # if not len(nan_mask)==0:
    #     # deletes the 'dead' channels
    #     nan_trials = [np.delete(tr, nan_mask, axis=1) for tr in trial_file['ephys']]
    #     trial_file["channels"] = [np.delete(chan, nan_mask) for chan in trial_file["channels"]]
        
    #     print('Removed nan channels')
        

    #ffts of all baseline trials of this subject. 3 time windows. trials x channels x freq
    fft_baseline = []
    
    #ffts of all image trials of this subject. 3 time windows.
    fft_t1 = []
    fft_t2 = []
    fs = int(trial_file.iloc[0].sample_rate)
    #window = signal.windows.hamming(int(fs*1.5))# + 256)
    window = signal.windows.hann(int(fs*1.5))# + 256)

    feature_fft = [None] * len(trial_file)

    per_change_s1 = []
    per_change_s2 = []
    index_tracker = []
            
    #loop all trials
    for index, trial_data in enumerate(trial_file.iterrows()):

        combined = np.ma.concatenate([trial_data[1].pre_trial_ephys, trial_data[1].ephys], axis=0)

        trial = np.ma.array(
                combined.data,
                mask=np.ma.getmaskarray(combined),
                copy=False,
                shrink=False,
            )

        trial_start_index = trial_data[1].pre_trial_ephys.shape[0]
        #include 250 ms before. the hamming window will make it contribute vwey ittle
        fft_start = int(trial_start_index - fs*0.25)

        #insert nan value which will be left for baseline data.
        feature_fft[index] = np.nan

        #0.250 ms before the trial until 3.25ms after trial onset
        all_trial_data = trial[fft_start : fft_start + int(fs*3.5), :]

        #check if it is a baseline trial.
        if trial_data[1]["last_response"] == "baseline":
            
            #ffts of this trial. 3 time windows.
            fft_baseline_trial = []

            
            #loop all channels
            # broadband filter + FFT calculation 
            for channel in range(np.shape(trial)[1]):
                #3 consequtive seconds with 0.25 sec overlap
                windowed1 = (all_trial_data[:int(fs*1.5), channel] - all_trial_data[:int(fs*1.5), channel].mean()) * window
                windowed2 = (all_trial_data[int(fs) : int(fs*2.5), channel] - all_trial_data[int(fs) : int(fs*2.5), channel].mean()) * window
                windowed3 = (all_trial_data[int(fs*2) : int(fs*3.5), channel] - all_trial_data[int(fs*2) : int(fs*3.5), channel].mean()) * window


                # tmp1 = fft_analysis(windowed1[128:-128], 254, fs, 254)[1][0]
                # tmp2 = fft_analysis(windowed2[128:-128], 254, fs, 254)[1][0]
                # tmp3 = fft_analysis(windowed3[128:-128], 254, fs, 254)[1][0]

                tmp1 = fft_analysis(windowed1.filled(0), 254, fs, 254)[1][0]
                tmp2 = fft_analysis(windowed2.filled(0), 254, fs, 254)[1][0]
                tmp3 = fft_analysis(windowed3.filled(0), 254, fs, 254)[1][0]

                fft_baseline_trial.append([tmp1, tmp2, tmp3])
            fft_baseline.append(fft_baseline_trial)
        
        #if an image trial
        if trial_data[1]["image_ID"] > 0:
        #NB also includes baseline trials with clicks...
        #if np.logical_and(trial_file["last_response"][index] != "baseline", trial_file["last_response"][index] != "mock"):
            
            index_tracker.append(index)
        
            tmp_1 = []
            tmp_2 = []
        
            #loop all channels
            for channel in range(np.shape(trial)[1]):

                #3 consequtive seconds with 0.25 sec overlap
                windowed1 = (all_trial_data[:int(fs*1.5), channel] - all_trial_data[:int(fs*1.5), channel].mean()) * window
                windowed2 = (all_trial_data[int(fs) : int(fs*2.5), channel] - all_trial_data[int(fs) : int(fs*2.5), channel].mean()) * window

                # tmp1 = fft_analysis(windowed1[128:-128], 254, fs, 254)[1][0]
                # tmp2 = fft_analysis(windowed2[128:-128], 254, fs, 254)[1][0]
                tmp1 = fft_analysis(windowed1.filled(0), 254, fs, 254)[1][0]
                tmp2 = fft_analysis(windowed2.filled(0), 254, fs, 254)[1][0]

                tmp_1.append(tmp1)
                tmp_2.append(tmp2)
            fft_t1.append(np.transpose(tmp_1))
            fft_t2.append(np.transpose(tmp_2))

        #if last trial in session/block (identified by assessing time jumpt to next trial. look for more than 10/3600 sec timesteps to next trial).
        if index == len(trial_file)-1 or np.diff(trial_file["clock_time"].iloc[index:index+2]) > time_jump:
            
            if len(index_tracker) > 0:
        
                #print('Last block trial: ', index, (index + 1) % 75)
                
                #get the mean of the baseline ffts for each channel, across trials (0) and windows (2)
                fft_base_avg = np.nanmean(np.array(fft_baseline), axis=(0,2))
                            
                #loop the image trials
                for s1, s2, num in zip(fft_t1, fft_t2, index_tracker):
                    # decibel change
                        dsec1 = 10*np.log10(np.divide(s1, fft_base_avg.T))
                        dsec2 = 10*np.log10(np.divide(s2, fft_base_avg.T))
                        feature_fft[num] = np.array([dsec1, dsec2])
                    # percentage change
                    # if
                        #do the corrections
                        # per_change_s1 = (100*(np.divide(s1 - fft_base_avg.T, fft_base_avg.T)))
                        # per_change_s2 = (100*(np.divide(s2 - fft_base_avg.T, fft_base_avg.T)))
                        # feature_fft[num] = np.array([per_change_s1, per_change_s2])
                        
                        #check that fft_base_avg is not zero
                        if np.sum(fft_base_avg==0) > 0:
                            print('Nan in average: ', num)
                            sys.exit()
                
                #number of channels with nan in baseline
                num_baseline_ch_nans = np.max(np.sum(np.isnan(fft_base_avg), axis=0))
                #number of trials
                num_session_trials = len(index_tracker)
                num_baseline_nans += num_session_trials*num_baseline_ch_nans
                
            else:
                print(f'Skipping! No trials in block/session.')
                
            
            #reset variables
            fft_t1 = []
            fft_t2 = []
            fft_baseline = []
            index_tracker = []
    
    #print the number of channel-trials affected by the baseline nans
    print(f'Baseline channels nans: {num_baseline_nans}')
         
        
        #else:
            #print('Trial: ', index)

    fft_trials = trial_file[['image_ID', 'clock_time', 'sample_rate', 'last_response',
       'last_motor_response', 'indices_all_responses', 'channels',
       'repetition_num']].copy()

    fft_trials[string_save] = feature_fft
    
    fft_trials.to_pickle(input_dir + "_" + string_save)

    return fft_trials
  

def transform_string(s):
  # Split the string at '|', take the first part, and strip any whitespace
  part = s.split('|')[0].strip()
  # Remove the single quote
  part = part.replace("'", "")
  part = part.replace(" ", "")
  # Extract the letter and number parts using slicing
  letter = part[0]
  number = part[1:]
  # Zero-pad the number if it has a single digit
  number = number.zfill(2)
  
  # Split the string at '|'
  part = s.split('|')[1].strip()
  # Remove any digits from the brain region (using regular expressions)
  freq = re.sub(r'\d+', '', part)
  
  
  return f"{letter}{number}", freq


def add_positional_encoding(unique_var_list, var_list, xdata):
    
    #everythiong that is equal to the first element:
    preceding = [unique_var_list[0] in s for s in var_list]
    
    #loop through theothers
    for t in unique_var_list[1:]:
        selected = [t in s for s in var_list]
        #separate the population from the preceding population
        #max_value of the earlier categories
        max_value = np.nanmax(xdata[:, preceding])
        #min value of the current category
        min_value = np.nanmin(xdata[:, selected])
        
        #add the difference so that all values exceeds the previous category. add 1 toget a separation
        add_constant = max_value - min_value
        if add_constant < 0:
            do_nothing = 1
            #print(t, 'no necessary correction')
        else:
            xdata[:, selected] = xdata[:, selected] + np.max([(add_constant+1), 1])
            #print(t, max_value, min_value)
        
        preceding = [a or b for a, b in zip(preceding, selected)]
        
    return xdata


def position_embedding(xdata, features, subject):
    
    electrodes = [transform_string(s)[0] for s in features]
    freqs = [transform_string(s)[1] for s in features]
    times = [s[-1] for s in features]
    probes = [s[0] for s in features]
    
    df_anatomy = anatomy.get_brain_areas(subject, hipp=True, atlas = 'Glasser', labels='Glasser')
    glasser_lobes = []
    glasser_areas = []
    destrieux_areas = []
    for el in electrodes:
        lobe = df_anatomy.brain_lobe[df_anatomy.Probe.isin([el])]
        g_area = df_anatomy.Glasser_labels[df_anatomy.Probe.isin([el])]
        d_area = df_anatomy.labels[df_anatomy.Probe.isin([el])]
        if len(lobe) == 1:
            glasser_lobes.append(lobe.iloc[0])
            glasser_areas.append(g_area.iloc[0])
            destrieux_areas.append(d_area.iloc[0])
        else:
            print('ERROR MORE THAN ONE ELECTRODE MATCHED')
            
    df_anatomy = anatomy.get_brain_areas(subject, hipp=True, atlas = 'Glasser', labels='Sugar')
    brain_lobes = []
    for el in electrodes:
        area = df_anatomy.brain_lobe[df_anatomy.Probe.isin([el])]
        if len(area) == 1:
            brain_lobes.append(area.iloc[0])
        else:
            print('ERROR MORE THAN ONE ELECTRODE MATCHED')
    
    
    unique_brain_lobes = list(OrderedDict.fromkeys(brain_lobes))
    unique_glasser_lobes = list(OrderedDict.fromkeys(glasser_lobes))
    unique_destrieux_areas = list(OrderedDict.fromkeys(destrieux_areas))
    unique_glasser_areas = list(OrderedDict.fromkeys(glasser_areas))
    unique_probes = list(OrderedDict.fromkeys(probes))
    unique_electrodes = list(OrderedDict.fromkeys(electrodes))
    unique_freqs = ['Delta', 'Theta', 'Alpha', 'Beta', 'GammaL', 'GammaH', 'Ripple']
    unique_times = list(OrderedDict.fromkeys(times))

    #do the absolute positional encoding 1st level: time. 2nd level: electrode. 3rd level brain area
    #time
    xdata = add_positional_encoding(unique_times, times, xdata)
    #frequency
    xdata = add_positional_encoding(unique_freqs, freqs, xdata)
    #electrode
    xdata = add_positional_encoding(unique_electrodes, electrodes, xdata)
    #probes
    xdata = add_positional_encoding(unique_probes, probes, xdata)
    #glasser
    xdata = add_positional_encoding(unique_glasser_areas, glasser_areas, xdata)  
    #destrieux
    xdata = add_positional_encoding(unique_destrieux_areas, destrieux_areas, xdata)  
    #glasser lobes
    xdata = add_positional_encoding(unique_glasser_lobes, glasser_lobes, xdata)
    #brain lobes
    xdata = add_positional_encoding(unique_brain_lobes, brain_lobes, xdata)
    
    #normalize the data from -1 to 1
    #lower bound is zero
    min_val = np.nanmin(xdata) 
    xdata = xdata - min_val
    #upper bound is 1
    max_val = np.nanmax(xdata)
    xdata = xdata / max_val
    #expand to -1
    xdata = (xdata*2) - 1
    
    return xdata
    
    
    
    
    


