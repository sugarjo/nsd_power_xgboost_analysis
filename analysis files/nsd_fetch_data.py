import h5py
import numpy as np
import pandas as pd
import os
import glob
from neural_decoding.utils import load_db, get_ephys_meta, analyze_timeseries
import scipy.io
from pathlib import Path
import time
from joblib import Parallel, delayed
import re


#turns .db files and ephys files to .pkl files in the nsd experiment

subjects =  ["CNP04", "CNP05", "CNP08", "CNP09", "CNP10", "CNP11", "CNP12",  "CNP14", "CNP18", "CNP19", "CNP20", "CNP21", "CNP22", "CNP26", "CNP27", "CNP28"]

#these will be excluded from including uanaligned baseline trials intrial file. include if rerun
consistent_export = ["CNP04", "CNP08", "CNP09", "CNP10", "CNP11", "CNP12", "CNP13", "CNP14", "CNP18", "CNP19", "CNP20", "CNP21", "CNP22", "CNP26", "CNP27"]

def fetch_data(subject):

    print(subject)
    
    try:

        db_folders = []
        do_exist = True
        n=1
    
        while do_exist:

            hypatia_root = Path(r"/run/user/339313/gvfs/smb-share:server=hypatia.uio.no,share=lh-med-imb-cnp/")
            path = hypatia_root / f"data_seeg/{subject}/natural_scenes_task_{n}"

            #path = f"//hypatia.uio.no/lh-med-imb-cnp/data_seeg/{subject}/natural_scenes_task_{n}"
            if os.path.exists(path):
                db_folders.append(path)
                n += 1
            else:
                do_exist = False

        
    
        experimental_folder = os.path.split(db_folders[0])[0]
        patient = os.path.split(experimental_folder)[1]
        storage_folder = os.path.split(experimental_folder)[0]
    
        output_folder = experimental_folder + '\\analysis_' + os.path.split(db_folders[0])[1][:-2]
    
    
        #create folder
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
    
        #get .db files
        age =  []
        gender = []
        datasets = [None] * len(db_folders)
        for exp_folder_ind, folder in enumerate(db_folders):
            databases = glob.glob(folder + '/*.db')
    
            #datasets[exp_folder_ind] = [None] * len(databases)
            dbs = []
            for db_ind, db in enumerate(databases):
                
                #find start of db_file
                match_date = re.search(r'data_(\d{4})-(\d{1,2})-(\d{1,2})_(\d{1,2})-(\d{1,2})', db)
                
                if not match_date:
                    print('Skipping', db)
                    continue

                #load data
                dataset = load_db.load(db)
    
                age.append(dataset["Info"][0]["Age"])
                gender.append(dataset["Info"][0]["Gender"])
                
                #find start of db_file
                match1 = re.search(r'data_(\d{4})-(\d{1,2})-(\d{1,2})_(\d{1,2})-(\d{1,2})', db)
                
                year_str = match_date.group(1)      # YYYY
                month_str = match_date.group(2).zfill(2)  # Ensure month is two digits
                day_str = match_date.group(3).zfill(2)    # Ensure day is two digits
                hour_str = match_date.group(4).zfill(2)    # Ensure hour is two digits
                minute_str = match_date.group(5).zfill(2)  # Ensure minute is two digits
            
                # Formulate the correct datetime string
                datetime_str = f"{year_str}-{month_str}-{day_str}T{hour_str}:{minute_str}:00.000000"

                
                # Convert to numpy datetime64
                datetime_np = np.datetime64(datetime_str)
                dataset["Info"][0]["Start_stimulus_db"] = datetime_np
                
                    
                if 'Ephys_filepaths' in dataset:
                    dbs.append(dataset)
    
            datasets[exp_folder_ind] = dbs
    
        #initate variables
        ephys = []
        trials = []
    
        downsample_factor = 4
        pre_trial_time_duration_sec = 1
        
        unipolar = False
    
    
        #loop through foldersdb_ind = 0
        for exp_ind in range(len(datasets)):
    
            print(subject, exp_ind)
    
            #loop through databases
            for dset_ind in range(len(datasets[exp_ind])):
    
                dset = datasets[exp_ind][dset_ind]
                
                #get ephys lengths to be able to calculate ephys indices from db
                ephys_lengths = np.cumsum(dset['Ephys_filepaths']['ephys_length'].values)
    
                #keep track of which trials have been stored in db
                loaded_trials = np.full(len(dset['DisplayedImages'])-1, False)                
    
                #loop through all ephys files
                for ephys_ind in range(len(dset['Ephys_filepaths']['path'])):
    
                    ephys_filename = os.path.split(dset['Ephys_filepaths']['path'][ephys_ind])[1]
                    ephys_filepath = os.path.join(storage_folder, patient, 'ephys', ephys_filename)
    
                    if not Path(ephys_filepath).is_file():
                        #sleep to make progressbar ot disappear
                        time.sleep(0.1)
                        #print('No file', ephys_filepath)
                        continue
    
                    #figure out which images are in the current ephys. and how much we need to subtract from the ephys indices in .db to get into ephys files
                    #(the ephys indices are cumulative indices across all files)
                    if ephys_ind > 0:
                        selected_images = np.logical_and(dset['DisplayedImages']['Onset_Ephys_index'] <= ephys_lengths[ephys_ind], dset['DisplayedImages']['Onset_Ephys_index'] > ephys_lengths[ephys_ind-1])
                        subtract_idx = ephys_lengths[ephys_ind-1]
                    else:
                        selected_images = dset['DisplayedImages']['Onset_Ephys_index'] <= ephys_lengths[ephys_ind]
                        subtract_idx = 0
    
                    #skip if no images
                    if sum(selected_images) == 0:
                        #print('No images')
                        continue
    
                    #load_ephys file
                    extension = Path(ephys_filename).suffix
                    
                    #print('Load', ephys_filename)
                    
                    if extension == '.mat':
                        with h5py.File(ephys_filepath, 'r') as ephys_data:
                            ephys_data = h5py.File(ephys_filepath, 'r')
                            fs = get_ephys_meta.fs(ephys_data)
    
                            #load all channels whole file
                            eeg = np.transpose(ephys_data['DATAOUT']['data'])
    
                            #get channels
                            ch_names, _ = get_ephys_meta.channel_string_names(ephys_data)
    
                        #load meta
                        meta_filepath = ephys_filepath.replace('_data_', '_meta_')
    
                        ephys_meta = scipy.io.loadmat(meta_filepath)
    
                        #get the max abs value. used to take out tresholded trials.
                        treshold = np.max(np.array([ephys_meta['sFile']['header'][0][0][0][0][-4][0][i][11] for i in range(len(ephys_meta['sFile']['header'][0][0][0][0][-4][0]))]).squeeze())
    
                        start_time = get_ephys_meta.get_start_time(meta_filepath)
    
                    elif extension == ".npz":
                        ephys_data = np.load(ephys_filepath)
    
                        channels = ephys_data["signal_channels"]
    
                        fs = int(channels[0][2])
    
                        eeg_int = ephys_data["raw_data"]
    
                        additions = np.array([t[6] for t in channels])
                        multipliers = np.array([t[5] for t in channels])
    
                        eeg = ((eeg_int * multipliers) + additions).T
    
                        #get meta
                        ch_names = [t[0] for t in channels]
                        
                        #to mitigate the 0s that the clinicans entered in stead of an Os
                        if subject == 'CNP27':
                            ch_names = ['O' + s[1:] if s.startswith('0') else s for s in  ch_names]
    
                        treshold = np.max([np.ceil(np.max(additions)), np.abs(np.floor(np.min(additions)))])
                        # for weired data:
                        if treshold == 0:
                            print('WARNING: CHECK DATA FORMAT')
                            treshold = np.max([np.ceil(np.max(eeg)), np.abs(np.floor(np.min(eeg)))])
                                
                        start_time = ephys_data["start_time"]
    
                    #remove service channels. The AX, BX etc is the clincial sync channels.
                    ch_keep = []
                    for string in ch_names:
                        if string[1]=='X' or 'PSYK' in string or 'MKR' in string or 'Psyk' in string or not any(char.isdigit() for char in string):
                            ch_keep.append(False)
                        else:
                            ch_keep.append(True)
    
                    #ch_ref, _ = get_ephys_meta.healthy_channels(ch_names, subject)
    
                    #reference = np.mean(eeg[ch_ref, :], axis=0)
    
                    ephys_shape = eeg.shape
    
                    #calculate ind of mouse presses
                    #since we subtract the preceding indices we end up with the index in the current ephys file.
                    #subtract 1 since conversion is done in matlab.
                    mouse_presses_ts = dset['MousePress']['Timestamp_Ephys_index'] - subtract_idx - 1
                    mouse_presses = dset['MousePress']['KeyCode']
    
                    selected_left = [(x[0] == "1" and x[1:3] == "00") for x in  mouse_presses]
                    selected_top = [(x[0] == "0" and (x[1] == "1" or x[2] == "1")) for x in  mouse_presses]
    
                    selected = [left or top for left, top in zip(selected_left, selected_top)]
    
                    mouse_presses_ts = mouse_presses_ts[selected]
                    mouse_presses = mouse_presses[selected]
    
                    #initate variable to know if you have loaded the data
                    ch2_loaded = -1
    
                    #indicates that missing data should be reported
                    report_missing_data = True
                    
                    #print('Loop channels')
    
                    #loop all channels (not the last one since we do bipolar)
                    for ch_ind in range(ephys_shape[0]):
    
                        #if we will keep both
                        #NB only works for bipolar if last channel is rejected.
                        if ch_keep[ch_ind]:
                            #if allready loaded
                            if ch2_loaded == ch_ind:
                                ch1 = ch2
                                filtered_ch1 = filtered_ch2
                            else:
                                ch1 = eeg[ch_ind]#-reference
                                #filter and convolve to prepare for downsampling (no need to keep frequences above the downsampled limit)
                                filtered_ch1 = np.convolve(analyze_timeseries.butterworth(ch1, [0.75, fs/downsample_factor/2], fs), np.ones(downsample_factor)/downsample_factor, mode='same')
    
                            #skip if dead channel
                            if len(np.unique(ch1)) == 1:
                                continue

                            if not unipolar:
                                
                                if not ch_keep[ch_ind+1]:
                                    continue
                                
                                probe1 = re.sub(r"[\d' ]", "", ch_names[ch_ind])
                                probe2 = re.sub(r"[\d' ]", "", ch_names[ch_ind + 1])
                                
                                
                                
                                #check if same letter in the two (necessary for bipolar)
                                if not any(c in probe2 for c in probe1):
                                    # print('Not same probe', ch_names[ch_ind], ch_names[ch_ind+1])
                                    continue
                                # else:
                                #     print('Kept', ch_names[ch_ind], ch_names[ch_ind+1])
                                
                                ch2 = eeg[ch_ind+1]#-reference
                                ch2_loaded = ch_ind+1

                                #skip if dead channel
                                if len(np.unique(ch2)) == 1:
                                    continue
                                
    
                                #filter and convolve to prepare for downsampling (no need to keep frequences above the downsampled limit)
                                filtered_ch2 = np.convolve(analyze_timeseries.butterworth(ch2, [0.75, fs/downsample_factor/2], fs), np.ones(downsample_factor)/downsample_factor, mode='same')
                            
                            #loop through all images in dataset (not the last one since it is a response)
                            for image_ind in range(len(dset['DisplayedImages'])-1):
    
                                # #do not include images from corrupted file
                                # if ephys_filepath == '\\\\hypatia.uio.no\\lh-med-imb-cnp\\data_seeg\\2023-11\\ephys\\ephys_data_EEG_101631.mat' and image_ind >= 300:
                                #     continue
    
    
    
                                #subtract one since table was created in matlab
                                trial_image_ind = dset['DisplayedImages']['ID'][image_ind]-1
                                try:
                                    image_path = dset['Images']['ImagePath'].iloc[trial_image_ind]
                                except:
                                    image_path = dset['Images']['imagePath'].iloc[trial_image_ind]
    
                                if image_path == 'no_path':
                                    trial_image = 0
                                else:
                                    filename = os.path.basename(image_path)
                                    trial_image = int(filename[:-4])
    
    
    
                                #subtract 1 since conversion is done in matlab
                                start = int(dset['DisplayedImages']['Onset_Ephys_index'][image_ind]) - subtract_idx - 1
                                stop_enc = int(dset['DisplayedImages']['Offset_Ephys_index'][image_ind]) - subtract_idx - 1
                                stop_ret = int(dset['DisplayedImages']['Offset_Ephys_index'][image_ind+1]) - subtract_idx - 1
                                #force the trial to be exactly 4 sec (=equal lengths of each trial facilitates processing later)
                                stop = int(start + fs*4)
    
                                length_enc = stop_enc-start
                                length_ret = stop_ret-stop_enc
    
    
                                if length_ret < 0:
                                    #add info about previous retrieval in case ephys file switch in the middle of trial (which leads to negative numbers)
                                    if image_ind > 0:
                                        prev_ret_start = int(dset['DisplayedImages']['Onset_Ephys_index'][image_ind-1]) - subtract_idx - 1
                                        length_prev_ret = start- prev_ret_start
                                    else:
                                        length_prev_ret = 0
    
                                    test_length_ret = length_prev_ret
                                else:
                                    test_length_ret = length_ret
    
                                #skip if extending outside of ephys
                                if (start <= 0 or start > len(filtered_ch1)) and (stop_ret <= 0 or stop_ret > len(filtered_ch1)):
                                    continue
    
                                #continue if we are processing an image (enc is longer than retrieval) and the next image (response is a grey one) and the index is within the current  and neither of the ch is tresholded
                                if length_enc > test_length_ret and dset['DisplayedImages']['ID'][image_ind+1] == 1:
    
                                    #start and stop is unreliable if start < 0
                                    if start <= 0:
                                        stop = stop_ret
                                        start = int(stop - fs*4)
                                        if report_missing_data:
                                            print('\nUnreliable start for ' + str(image_ind) +' in exp: ' + str(exp_ind) + ' ephys: ' + str(ephys_ind) + '.\n')
                                        
                                    
                                    #in case end of trial is missing (I don't know id start > 0 is needed here. Just to be sure...)
                                    if start > 0 and stop_ret <= 0:
                                        #unreliable if 0...
                                        stop_ret =  int(start + fs*4)
                                        if report_missing_data:
                                            print('\nUnreliable stop for ' + str(image_ind) +' in exp: ' + str(exp_ind) + ' ephys: ' + str(ephys_ind) + '.\n')
                                        
    
                                    #if partly matched
                                    if report_missing_data:
                                        if start <= 0:
                                            print('\nOnly include last ' + str(stop_ret/fs) + ' sec of image: ' + str(image_ind) +' in exp: ' + str(exp_ind) + ' ephys: ' + str(ephys_ind) + '.\n')
                                            #make sure that the loop reenters the saving
                                            loaded_trials[image_ind] = False
    
                                        if stop_ret > len(filtered_ch1):
                                            print('\nOnly include first ' + str((len(filtered_ch1)-start)/fs) + ' sec of image: ' + str(image_ind) +' in exp: ' + str(exp_ind) + ' ephys: ' + str(ephys_ind) + '.\n')
                                            #loaded_trials[image_ind] = False
    
                                    #get time of trial
                                    add_time = np.timedelta64(int(start/fs*1000), 'ms')
                                    trial_time = np.int64(start_time + add_time)/1e6
    
                                    #add data to behaviour db if not already saved
                                    if not loaded_trials[image_ind]:
    
                                        #get last response before stop. accept 100ms after onset of next trial
                                        
                                        #this might causes a button press to appear in two trials! but we exclude premature button presses so probably ok.
                                        start_ind = mouse_presses_ts.values - start
                                        #accept button presses shortly after onset of next image
                                        stop_ind = stop - (mouse_presses_ts.values-0.1*fs)
    
                                        #get presses after start and before stop
                                        selected_ind = (start_ind > 0) & (stop_ind > 0)
                                        pretrial_selected_ind = (start_ind >= start_ind - (fs*pre_trial_time_duration_sec)) & (start_ind <= 0)
    
                                        motor_response = ' '
                                        all_responses = []
    
                                        #if any button clicks before the end of trial: get the motor response of the last press
                                        if any(selected_ind):
                                            #find index of max value of those negative (less than stop)
                                            press_inds = np.argwhere(selected_ind)
    
                                            last_button = mouse_presses.values[press_inds[-1]]
                                            #check if last button press involved left button
                                            if last_button[0][0] == '1':
                                                motor_response = 'index'
                                            else:
                                                motor_response = 'thumb'
    
                                            #get the indices of all presses
                                            #get index of all responses within trial
                                            all_responses = (start_ind[selected_ind]/downsample_factor).tolist()
                                        
                                        pretrial_responses = []
                                        if any(pretrial_selected_ind):
                                            pretrial_responses = (start_ind[pretrial_selected_ind]/downsample_factor).tolist()
                                            
    
                                        #insert ' ' for all grey screen trials
                                        response = ' '
                                        #identify mock trials
                                        if trial_image == 0:
                                            response = 'baseline'
                                            #check if last image was also a mock trial: set it as a baseline trial
                                            if image_ind > 1:
                                                #this will remove first trial after pause
                                                if dset['DisplayedImages']['Onset_Ephys_index'][image_ind] - dset['DisplayedImages']['Onset_Ephys_index'][image_ind-2] < fs*16:
                                                    #check if previous was not grey screen. if so it is a mock
                                                    if not dset['DisplayedImages']['ID'][image_ind-2] == 1:
                                                        response = 'mock'
    
    
                                        #technical error in button press registraiton on block 13 (exp 2 block 1)
                                        if subject == 'CNP04':
                                            if exp_ind == 1 and dset_ind == 0:
                                                if response == ' ':
                                                    motor_response = 'technical_error: unknown'
                                                    response = 'technical_error: unknown'
                                                    all_responses = []
    
    
                                        trials.append(
                                            {
                                                'image_ID': trial_image,
                                                'clock_time': trial_time,
                                                'sample_rate': fs/downsample_factor,
                                                'last_response': response,
                                                'last_motor_response': motor_response,
                                                'indices_all_responses': all_responses,
                                                'indices_pretrial_responses': pretrial_responses
                                            }
                                        )
    
    
                                        loaded_trials[image_ind] = True
    
                                    if start < 0:
                                        #make sure we get end of trial
                                        start = 0
                                        #continue
    
    
                                    #continue only if the signal is varying (some channels are dead) and no timepoint is tresholded
                                    if len(np.unique(ch1[start:stop])) > 1:
                                        if not unipolar:
                                            if len(np.unique(ch2[start:stop])) == 1:
                                                #print(ch_names[ch_ind+1], ' is saturated 1!')  
                                                continue
                                            #else:
                                                #max_value = np.max(np.abs(ch2[start:stop]))
                                                #if max_value>=treshold-1:
                                                    #continue
                                    else:
                                        if len(np.unique(ch1[start:stop])) == 1:
                                            #print(ch_names[ch_ind], ' is saturated 1!') 
                                            a=1
                                        continue
    
                                    
                                    
                                    pre_start_ind = start-(int(fs*pre_trial_time_duration_sec))
    
                                    #get ephys
                                    if unipolar:
                                        #calculate moving average to prepare for downsampling
                                        trial_ephys = filtered_ch1[start:stop]
                                        #saturated = np.abs(ch1[start:stop]+reference[start:stop]) > treshold-1
                                        saturated = np.abs(ch1[start:stop]) > treshold-1
    
                                        #add half a second of pre_trial data
                                        pre_trial_ephys = filtered_ch1[pre_start_ind:start]
                                        #pre_trial_sat = np.abs(ch1[pre_start_ind:start]+reference[pre_start_ind:start]) > treshold-1
                                        pre_trial_sat = np.abs(ch1[pre_start_ind:start]) > treshold-1
                                    else:
                                        #skip if the bipolar signal cross two electrodes
                                        #onlu works if ch_ind+1 is rejected in start of ch-loop
                                        # if not bipolar_ch_keep[ch_ind]:
                                        #     continue
                                        #calculate moving average to prepare for downsampling
                                        trial_ephys = (filtered_ch1 - filtered_ch2)[start:stop]
                                        #saturated = np.logical_or(np.abs(ch1[start:stop]+reference[start:stop]) > treshold-1, np.abs(ch2[start:stop]+reference[start:stop]) > treshold-1)
                                        saturated = np.logical_or(np.abs(ch1[start:stop]) > treshold-1, np.abs(ch2[start:stop]) > treshold-1)
    
                                        pre_trial_ephys = (filtered_ch1 - filtered_ch2)[pre_start_ind:start]
                                        #pre_trial_sat = np.logical_or(np.abs(ch1[pre_start_ind:start]+reference[pre_start_ind:start]) > treshold-1, np.abs(ch2[pre_start_ind:start]+reference[pre_start_ind:start]) > treshold-1)
                                        pre_trial_sat = np.logical_or(np.abs(ch1[pre_start_ind:start]) > treshold-1, np.abs(ch2[pre_start_ind:start]) > treshold-1)
        
    
                                    #skip saturated trials
                                    if saturated.all():
                                        #print(ch_names[ch_ind], ' is saturated for image', image_ind)  
                                        continue
    
                                    trial_ephys = np.ma.masked_array(trial_ephys, mask=saturated, dtype='int16')[::downsample_factor]
    
                                    pre_trial_ephys = np.ma.masked_array(pre_trial_ephys, mask=pre_trial_sat, dtype='int16')[::downsample_factor]
                                    
                                    if unipolar:
                                        save_ch_name = ch_names[ch_ind]
                                    else:
                                        save_ch_name = ch_names[ch_ind] + ' - ' + ch_names[ch_ind+1]
                                    
                                    ephys.append(
                                        {
                                            'channel': save_ch_name,
                                            'ephys': trial_ephys,
                                            'clock_time': trial_time,
                                            'pre_trial_ephys': pre_trial_ephys
                                        }
                                                )
                                    
                            report_missing_data = False
                            
                            
                            
                print(exp_ind, 'Matched', sum(loaded_trials), 'of', int(np.ceil(len(loaded_trials)/2)))
                
                #handle any missing ephys data. Only necessary for CNP05. For the rest there are only single grey trials that are missing
                #remove when reexported. this will include unaligned trials
                if subject not in consistent_export:
                    if sum(loaded_trials) < int(np.ceil(len(loaded_trials)/2)):
                        
                        print('Handle missing data')
                        
                        #a numpy datetime
                        start_time = dset["Info"][0]['Start_stimulus_db']
                        start_time_system  = dset["Info"][0]['StartSystem']
            
                        #calculate ind of mouse presses
                        #since we subtract the preceding indices we end up with the index in the current ephys file.
                        #subtract 1 since conversion is done in matlab.
                        mouse_presses_ts = dset['MousePress']['TimestampSystem'] - start_time_system
                        mouse_presses = dset['MousePress']['KeyCode']
        
                        selected_left = [(x[0] == "1" and x[1:3] == "00") for x in  mouse_presses]
                        selected_top = [(x[0] == "0" and (x[1] == "1" or x[2] == "1")) for x in  mouse_presses]
        
                        selected = [left or top for left, top in zip(selected_left, selected_top)]
        
                        mouse_presses_ts = mouse_presses_ts[selected]
                        mouse_presses = mouse_presses[selected]
    
                        #loop through all images in dataset (not the last one since it is a response)
                        for image_ind in range(len(dset['DisplayedImages'])-1):
                            
                            if not (dset['DisplayedImages']['Onset_Ephys_index'][image_ind] == 0 and dset['DisplayedImages']['Offset_Ephys_index'][image_ind] == 0):
                                continue
                                
    
    
                            #subtract one since table was created in matlab
                            trial_image_ind = dset['DisplayedImages']['ID'][image_ind]-1
                            try:
                                image_path = dset['Images']['ImagePath'].iloc[trial_image_ind]
                            except:
                                image_path = dset['Images']['imagePath'].iloc[trial_image_ind]
    
                            if image_path == 'no_path':
                                trial_image = 0
                            else:
                                filename = os.path.basename(image_path)
                                trial_image = int(filename[:-4])
    
                            #subtract 1 since conversion is done in matlab
                            start = int(dset['DisplayedImages']['OnsetSystem'][image_ind]) - start_time_system
                            stop_enc = int(dset['DisplayedImages']['OffsetSystem'][image_ind]) - start_time_system
                            stop_ret = int(dset['DisplayedImages']['OffsetSystem'][image_ind+1]) - start_time_system
                            #force the trial to be exactly 4 sec (=equal lengths of each trial facilitates processing later)
                            stop = int(start + 4*1e6)
    
                            length_enc = stop_enc-start
                            length_ret = stop_ret-stop_enc
    
    
                            if length_ret < 0:
                                #add info about previous retrieval in case ephys file switch in the middle of trial (which leads to negative numbers)
                                if image_ind > 0:
                                    prev_ret_start =  int(dset['DisplayedImages']['OnsetSystem'][image_ind-1]) - start_time_system
                                    length_prev_ret = start- prev_ret_start
                                else:
                                    length_prev_ret = 0
    
                                test_length_ret = length_prev_ret
                            else:
                                test_length_ret = length_ret
    
                            #continue if we are processing an image (enc is longer than retrieval) and the next image (response is a grey one) and the index is within the current  and neither of the ch is tresholded
                            if length_enc > test_length_ret and dset['DisplayedImages']['ID'][image_ind+1] == 1:
    
                                #get time of trial
                                add_time = np.timedelta64(start, 'us')
                                trial_time = np.int64(start_time + add_time)/1e6
    
                                #add data to behaviour db if not already saved
                                if not loaded_trials[image_ind]:
    
                                    #get last response before stop. accept 100ms after onset of next trial
                                    
                                    start_ind = mouse_presses_ts.values - start
                                    #accept button presses shortly after onset of next image
                                    stop_ind = stop - (mouse_presses_ts.values-0.1*1e6)
    
                                    #get presses after start and before stop
                                    selected_ind = (start_ind > 0) & (stop_ind > 0)
                                    pretrial_selected_ind = (start_ind >= start_ind - (fs*pre_trial_time_duration_sec)) & (start_ind <= 0)
    
    
                                    motor_response = ' '
                                    all_responses = []
    
                                    #if any button clicks before the end of trial: get the motor response of the last press
                                    if any(selected_ind):
                                        #find index of max value of those negative (less than stop)
                                        press_inds = np.argwhere(selected_ind)
    
                                        last_button = mouse_presses.values[press_inds[-1]]
                                        #check if last button press involved left button
                                        if last_button[0][0] == '1':
                                            motor_response = 'index'
                                        else:
                                            motor_response = 'thumb'
    
                                        #get the indices of all presses
                                        #get index of all responses within trial
                                        all_responses = (start_ind[selected_ind]/1e6*fs).tolist()
                                        
                                    pretrial_responses = []
                                    if any(pretrial_selected_ind):
                                        pretrial_responses = (start_ind[pretrial_selected_ind]/downsample_factor).tolist()
                                       
    
                                    #insert ' ' for all grey screen trials
                                    response = ' '
                                    #identify mock trials
                                    if trial_image == 0:
                                        response = 'baseline'
                                        #check if last image was also a mock trial: set it as a baseline trial
                                        if image_ind > 1:
                                            #this will remove first trial after pause
                                            if dset['DisplayedImages']['Onset_Ephys_index'][image_ind] - dset['DisplayedImages']['Onset_Ephys_index'][image_ind-2] < fs*16:
                                                #check if previous was not grey screen. if so it is a mock
                                                if not dset['DisplayedImages']['ID'][image_ind-2] == 1:
                                                    response = 'mock'
    
                                    trials.append(
                                        {
                                            'image_ID': trial_image, #image_path
                                            'clock_time': trial_time, #start of ephys file + ephys index in ms
                                            'sample_rate': fs/downsample_factor,
                                            'last_response': response,
                                            'last_motor_response': motor_response,
                                            'indices_all_responses': all_responses,
                                            'indices_pretrial_responses': pretrial_responses
                                            
                                        }
                                    )
    
    
                                    loaded_trials[image_ind] = True
                                
                                    #add a random ephys to make script work downstream 
                                
                                    empty_ephys = np.empty(len(ephys[0]["ephys"]))
                                    empty_ephys[:] = np.nan
                                    
                                    empty_pre_ephys = np.empty(len(ephys[0]["pre_trial_ephys"]))
                                    empty_pre_ephys[:] = np.nan
                                
                                    ephys.append(
                                        {
                                            'channel': ephys[0]["channel"],
                                            'ephys': empty_ephys,
                                            'clock_time': trial_time,
                                            'pre_trial_ephys': pre_trial_ephys
                                        })

        
  
                                  
        ephys = pd.DataFrame(ephys)
        trials = pd.DataFrame(trials)
    
        #get start time of all trials
        trials_ts = np.unique(trials['clock_time'])
        #get channels. the order will be used for all trials.
        all_channels = np.unique([ephys['channel']])
    
        #create dataframes
        ephys_trials = pd.DataFrame(columns=["ephys"])
        ephys_trials["ephys"] = ephys_trials["ephys"].astype(object)
        ephys_pretrials = pd.DataFrame(columns=["pre_trial_ephys"])
        ephys_pretrials["pre_trial_ephys"] = ephys_pretrials["pre_trial_ephys"].astype(object)
    
        ephys_channels = pd.DataFrame(columns=["channels"])
        ephys_channels["channels"] = ephys_channels["channels"].astype(object)
    
        #tf_trials = pd.DataFrame(columns=["time_frequency"])
        #tf_trials["time_frequency"] = tf_trials["time_frequency"].astype(object)
    
        #loop trials
        for trial_time in trials_ts:
    
            #find the relevant ephys data for this trial
            selected_ephys = ephys["clock_time"] == trial_time
            selected_trial = np.where(trials["clock_time"] == trial_time)[0][0]
    
            #get the relevant data
            recorded_trial_ephys = ephys[selected_ephys][["ephys", "pre_trial_ephys", "channel"]]
            #set the index to the channel name
            recorded_trial_ephys = recorded_trial_ephys.set_index('channel')

    
            #get the relevant data
            #recorded_trial_time_power = ephys[selected_ephys][["time_power", "channel"]]
            #set the index to the channel name
            #recorded_trial_time_power = recorded_trial_time_power.set_index('channel')
    
            #create empty dataframe with all possible channels. this will make sure that the same channels are included for all trials and that their order will be conistent
            backbone_trial_ephys = pd.DataFrame({'channel': all_channels})
            #set the index to the channel
            backbone_trial_ephys = backbone_trial_ephys.set_index('channel')
    
            #merge the two. will leave nan in channels with no data
            merged_trial_ephys = pd.concat([backbone_trial_ephys, recorded_trial_ephys], axis=1, join="outer")
            #insert missing data
            #create empty masked array with length of time series
            empty_ephys = np.ma.masked_all(recorded_trial_ephys["ephys"].apply(len).max())
            empty_ephys.set_fill_value(np.nan)
    
            #find those that have nans
            missing = merged_trial_ephys["ephys"].isna()
            if any(missing):
                #exchange nans with vector of nans
                merged_trial_ephys.loc[missing, 'ephys'] = merged_trial_ephys.loc[missing,'ephys'].apply(lambda x: empty_ephys)
    
    
            #create empty masked array with length of time series
            empty_ephys = np.ma.masked_all(recorded_trial_ephys["pre_trial_ephys"].apply(len).max())
            empty_ephys.set_fill_value(np.nan)
            #find those that have nans
            missing = merged_trial_ephys["pre_trial_ephys"].isna()
            if any(missing):
                #exchange nans with vector of nans
                merged_trial_ephys.loc[missing, 'pre_trial_ephys'] = merged_trial_ephys.loc[missing,'pre_trial_ephys'].apply(lambda x: empty_ephys)
    
            #merge the two. will leave nan in channels with no data
            #merged_trial_time_power = pd.concat([backbone_trial_ephys, recorded_trial_time_power], axis=1, join="outer")
            #insert missing data
            #empty_tp = np.empty((len(trials.iloc[selected_trial]["frequencies"]), len(trials.iloc[selected_trial]["gaze"])))
            #empty_tp[:] = np.nan
            #exchange nans with matrix of nans
            #merged_trial_time_power.loc[missing,['time_power']] = merged_trial_time_power.loc[missing,'time_power'].apply(lambda x: empty_tp)
    
    
            #convert to numpy and transpose so that time is rows
            ephys_trials.loc[selected_trial, "ephys"] = np.ma.vstack(merged_trial_ephys["ephys"]).astype('int16').T
            ephys_pretrials.loc[selected_trial, "pre_trial_ephys"] = np.ma.vstack(merged_trial_ephys["pre_trial_ephys"]).astype('int16').T
    
            ephys_channels.loc[selected_trial, "channels"] = list(backbone_trial_ephys.index)
    
        #insert into dataframe
        trials[["ephys"]] = ephys_trials
        trials[["pre_trial_ephys"]] = ephys_pretrials
        trials[["channels"]] = ephys_channels
    
        #order trials chronological
        unique_idx = np.unique(trials['clock_time'], return_index=True)
        trials = trials.iloc[unique_idx[1]]
        #order
        trials = trials.sort_values(by='clock_time')
        trials = trials.reset_index(drop=True)
    
        #look after trials which are nor complete and clost in time
        #iti = np.diff(trials.clock_time)
        
        trial_durations = [k[1].ephys.shape[0] for k in trials.iterrows()]
        
        neighbour_trialduration = [trial_durations[i] + trial_durations[i + 1] for i in range(len(trial_durations) - 1)]
        
        #merge_trials = np.where(iti < 2)
        merge_trials = np.argwhere([k <= fs*4/downsample_factor for k in neighbour_trialduration])
        for trial_pair_array in merge_trials:
            trial_pair = trial_pair_array[0]
            #see i fthey can fill up a trial
            if trials.iloc[trial_pair].ephys.shape[0] + trials.ephys.iloc[trial_pair+1].shape[0] < fs*4/downsample_factor:
                print('Merging trial ' + str(trial_pair))
    
                #create new merged data
                empty_trials = np.ma.masked_array(np.zeros((int(fs*4/downsample_factor), trials.ephys.iloc[trial_pair].shape[1])), mask=True)
                #insert the first in the beginning
                empty_trials[:trials.ephys.iloc[trial_pair].shape[0], :] = trials.ephys.iloc[trial_pair]
                #insert the second at the end
                empty_trials[-trials.ephys.iloc[trial_pair+1].shape[0]:, :] = trials.ephys.iloc[trial_pair+1]
    
                start_index_last_part = empty_trials.shape[0]-trials.ephys.iloc[trial_pair+1].shape[0]
                #deal with the response
                responses_in_merged = trials.indices_all_responses.iloc[trial_pair] + [k+start_index_last_part for k in trials.indices_all_responses.iloc[trial_pair+1]]
                # #if the ephys index of the last response in the trial is larger than the length of the ephys. then we use the 
                # if trials.indices_all_responses.iloc[trial_pair][-1] > trials.ephys.iloc[trial_pair].shape[0]:
                #     trials.indices_all_responses.iloc[trial_pair] = trials.indices_all_responses.iloc[trial_pair+1]
                
                #only take the pretrial of the first one
                pretrial_responses_in_merged = trials.indices_all_responses.iloc[trial_pair]
                #insert the data in the pandas
                trials.ephys.iloc[trial_pair] = empty_trials.copy()
    
                #delete the other trial from the pandas
                trials = trials.drop(trial_pair+1)
    
                trials = trials.reset_index()
                
        trial_durations = [k[1].ephys.shape[0] for k in trials.iterrows()]
        
        #figure out if too short trials are the beginning or the end
        too_short_trials = np.argwhere([k < fs*4/downsample_factor for k in trial_durations])
        too_short_trials = [k[0] for k in too_short_trials]
        #loop backwards since the end is most likely to be the wrong one. correct this first!
        for short_trial_ind in too_short_trials[::-1]:
            
            #check if length of previous trial is also too short:            
            if trials.iloc[short_trial_ind-1]["ephys"].shape[0] < fs*4/downsample_factor:
                empty_trials = np.ma.masked_array(np.zeros((int(fs*4/downsample_factor), trials.ephys.iloc[short_trial_ind].shape[1])), mask=True)
                #insert the second at the end
                empty_trials[-trials.ephys.iloc[short_trial_ind].shape[0]:, :] = trials.ephys.iloc[short_trial_ind]
                
                trials.ephys.iloc[short_trial_ind] = empty_trials.copy()
                
            
            #check if next trial has a nan in the beginnning
            elif trials.iloc[short_trial_ind+1]["ephys"][0,0].mask:
                empty_trials = np.ma.masked_array(np.zeros((int(fs*4/downsample_factor), trials.ephys.iloc[short_trial_ind].shape[1])), mask=True)
                #insert the second at the end
                empty_trials[:trials.ephys.iloc[short_trial_ind].shape[0]:, :] = trials.ephys.iloc[short_trial_ind]
                
                trials.ephys.iloc[short_trial_ind] = empty_trials.copy()  
            else:
                print('Short trial. Unaligned in beginning or end?',  short_trial_ind)
            
            
            
            
    
        #identify if patient reported new/old based on accuracy
        time_since_last = []
        #get correct image responses
        for image_ind, image_ID in enumerate(trials['image_ID']):
    
            last_clock_diff = 0
    
            if image_ID > 0:
                clock_time_current = trials['clock_time'][image_ind]
    
                selected = np.argwhere(np.array(trials['image_ID']) == image_ID)
    
                clock_diff = clock_time_current - trials['clock_time'][selected[:, 0]]
    
                if np.max(clock_diff) > 0:
                    #get the smallest of those larger than zero (preceding in time)
                    selected = clock_diff > 0
                    last_clock_diff = clock_diff.iloc[np.argmin(clock_diff[selected])]
    
            time_since_last.append(last_clock_diff)
    
        time_since_last = np.array(time_since_last)
    
        #new/old_trials with any response:
        answered_trials = trials['indices_all_responses'].apply(lambda x: len(x) > 0)
    
        new_trials = np.logical_and(time_since_last==0, np.logical_and(trials['image_ID'] > 0, answered_trials))
        old_trials = np.logical_and(time_since_last>0, np.logical_and(trials['image_ID'] > 0, answered_trials))
    
        thumb_trials = (trials['last_motor_response'] == 'thumb') & (trials['image_ID'] > 0)
        index_trials = (trials['last_motor_response'] == 'index') & (trials['image_ID'] > 0)
    
        #loop through the trials and figure out what are the correct responses
        #assumes that the patient is mainly correct
        while any(trials['last_response'][thumb_trials] == ' ') or any(trials['last_response'][index_trials] == ' '):
            no_answer = np.where((trials['last_response'] == ' ') & (thumb_trials | index_trials))[0]
            first_image = trials['clock_time'][no_answer[-1]]
            selected = np.abs(trials['clock_time'] - first_image) < 2*60*60
    
            accuracy_thumb = sum((trials['last_motor_response'][selected] == 'thumb') & (old_trials[selected])) + sum((trials['last_motor_response'][selected] == 'index') & (new_trials[selected]))
            accuracy_index = sum((trials['last_motor_response'][selected] == 'index') & (old_trials[selected])) + sum((trials['last_motor_response'][selected] == 'thumb') & (new_trials[selected]))
    
            if accuracy_thumb > accuracy_index:
                trials.loc[(selected) & (thumb_trials), 'last_response'] = 'old'
                trials.loc[(selected) & (index_trials), 'last_response'] = 'new'
                #print('Thumb is old, index is new', accuracy_thumb/(accuracy_index+0.01))
            else:
                trials.loc[(selected) & (thumb_trials), 'last_response'] = 'new'
                trials.loc[(selected) & (index_trials), 'last_response'] = 'old'
                #print('Thumb is new, index is old', accuracy_thumb/(accuracy_index+0.01))
    
        #store to file
        if unipolar:
            trials.to_pickle(os.path.join(output_folder, 'unipolar_trials' + '.pkl'))
        else:
            trials.to_pickle(os.path.join(output_folder, 'bipolar_trials' + '.pkl'))
                 
        print(trials.shape, 'done')
        print(subject, 'done')
        
    except Exception as e:
        print(e)
        return

if len(subjects) > 1:
    #parallel_results = Parallel(n_jobs=5)(delayed(fetch_data)(subject) for subject in subjects)
    
    for subject in subjects:
        fetch_data(subject)
    
else:
    fetch_data(subjects[0])