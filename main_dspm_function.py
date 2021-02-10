#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'qt')
import mne
import numpy as np
import matplotlib.pyplot as plt
import os.path as op

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

mne.set_log_level('WARNING')

from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs, write_inverse_operator, estimate_snr

from mayavi import mlab
# from IPython.display import Image


# In[ ]:


FIRST_EV = [65.583, 67.787, 68.485, 51.884, 48.919, 55.836, 1, 93, 48.372, 55.292, 64.907, 68.374, 66.375, 66.901, 67.261]


# In[ ]:


def var_reject(epochs, plow, phigh, to_plot=True):
    """
    Variance based trial rejection function
    """
    badtrls = []
    trl_var, trlindx = np.empty((0,1),'float'), np.arange(0,len(epochs))
    for trnum in range(len(epochs)):
        trl_var = np.vstack((trl_var, max(np.var(np.squeeze(epochs[trnum].get_data()), axis=1))))
    lim1 = (trl_var < np.percentile(trl_var, plow, interpolation='midpoint')).flatten()
    lim2 = (trl_var > np.percentile(trl_var, phigh, interpolation='midpoint')).flatten()
    outlr_idx = trlindx[lim1].tolist() + trlindx[lim2].tolist()

    if to_plot:
        plt.figure(), plt.scatter(trlindx, trl_var, marker='o', s=50, c='g', label='Good trials'),
        plt.ylabel('Max. variance across channels-->')
        plt.scatter(outlr_idx, trl_var[outlr_idx], marker='o', s=50, c='r', label='Variance based bad trials'),
        plt.xlabel('Trial number-->')
        plt.scatter(badtrls, trl_var[badtrls], marker='o', s=50, c='orange', label='Manually assigned bad trials')
        plt.ylim(min(trl_var)-min(trl_var)*0.01, max(trl_var)+max(trl_var)*0.01), plt.title('Max. variance distribution')
        plt.legend()
        plt.show()
    bad_trials = np.union1d(badtrls, outlr_idx)
#     print('Removed trials: %s\n'%bad_trials)
    return bad_trials


# In[ ]:


def set_params(iSub, ctrlwin=[-0.5, 0], actiwin=[0, 1], plow=2, phigh=98, dominant_brain="left"):
    """
    Set parameters, directories and filenames for the subject
    """

    par = {'ctrlwin': ctrlwin, 'actiwin': actiwin}
    par['plow'], par['phigh'] = plow, phigh

    par['data_dir'] = op.expanduser("~/data/pic-name-data-bids/")
    sSub = '%02d' % iSub
    session , task, run = '01', 'picturenaming', '01'

    par['data_path'] = op.join(par['data_dir'], 'MEG')
    subjects_dir = op.join(par['data_dir'], 'MRI')
    subject = 'sub-' + sSub
    par['res_dir'] = op.join(op.expanduser("~/research/results/picname"), subject)

    par['bids_fname'] = subject + '_ses-' + session + '_task-' + task + '_run-' + run + '_meg.fif'
    par['bids_path'] = op.join(par['data_path'], subject, 'ses-'+session, 'meg')
    par['raw_fname'] = op.join(par['bids_path'], par['bids_fname'])

    par['evoked_fname'] = op.join(par['res_dir'], subject+'-ave.fif')
    par['trans_fname'] = op.join(par['bids_path'], subject+'-trans.fif')
    par['fwd_fname'] = op.join(par['bids_path'], subject + '-cort-meg-fwd.fif')
    par['mrifile'] = op.join(subjects_dir, subject, 'mri/T1.mgz')
    par['surffile'] = op.join(subjects_dir, subject,
                              'bem/watershed', subject+'_brain_surface')
    par['stc_fname'] = op.join(par['res_dir'], 'dspm_' + subject)
    par['info'] = mne.io.read_info(par['raw_fname'])
    par['dominant_brain'] = dominant_brain

    # Set time instant for cropping raw data in the beginning
    par['raw_tmin'] = FIRST_EV[iSub-1] - 0.501

    # Changing of EEG channels to EOG/ECG
    if subject == 'sub-06':
        par['change_eeg_channels'] = True
    else:
        par['change_eeg_channels'] = False

    # Applying EOG/ECG SSP
    if iSub in [1, 2, 3, 4, 6]:
        par['apply_ssp_eog'] = True
        par['apply_ssp_ecg'] = True
    elif iSub in range(7, 16):
        par['apply_ssp_eog'] = True
        par['apply_ssp_ecg'] = False
    elif iSub == 100:
        par['apply_ssp_eog'] = False
        par['apply_ssp_ecg'] = True
    elif iSub == 5:
        par['apply_ssp_eog'] = False
        par['apply_ssp_ecg'] = False

    return par, subject, subjects_dir


# In[ ]:


def preprocess(par, subject, subjects_dir, more_plots=False):
    """
    Preprocess data, load epochs, and get evoked response
    """
    raw = mne.io.read_raw_fif(par['raw_fname'], allow_maxshield=False, preload=True, verbose=True)
    raw.crop(tmin=par['raw_tmin'], tmax=None)

    events = mne.find_events(raw, stim_channel='STI101', min_duration=0.002, shortest_event=2)
    delay = int(round(0.056 * raw.info['sfreq']))
    events[:, 0] = events[:, 0] + delay
    if more_plots:
        mne.viz.plot_events(events, first_samp=0, event_id=None, equal_spacing=True, show=True)

    # Change channel types for sub-06
    if par['change_eeg_channels']:
        print("Renaming EEG channels and changing channel types for subject:", subject)
        raw.set_channel_types({'EEG061': 'eog', 'EEG062': 'ecg'})
        raw.rename_channels({'EEG061': 'EOG061', 'EEG062': 'ECG062'})

    # Create EOG projectors and apply SSP
    if par['apply_ssp_eog']:
        print('Computing EOG projectors')
        # Create EOG projectors and apply SSP
        eog_epochs = mne.preprocessing.create_eog_epochs(raw)
        eog_epochs.average().plot_joint(picks='mag')
        plt.savefig(op.join(par['res_dir'], 'eog_evoked_' + subject + '.pdf'))
        plt.close()
        projs_eog, _ = mne.preprocessing.compute_proj_eog(raw, n_mag=3, n_grad=3, average=True)

    # Create ECG projectors and apply SSP
    if par['apply_ssp_ecg']:
        print('Computing ECG projectors')
        # Create ECG projectors and apply SSP
        ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
        ecg_epochs.average().plot_joint(picks='mag')
        plt.savefig(op.join(par['res_dir'], 'ecg_evoked_' + subject + '.pdf'))
        plt.close()
        projs_ecg, _ = mne.preprocessing.compute_proj_ecg(raw, n_mag=3, n_grad=3, average=True)

    picks = mne.pick_types(raw.info, meg=True, exclude='bads')

    raw.filter(1, 20, picks=picks, filter_length='auto', n_jobs=1,
          method='fir', iir_params=None, phase='zero', fir_window='hamming',
          fir_design='firwin', skip_by_annotation=('edge', 'bad_acq_skip'),
          pad='reflect_limited', verbose=True)
    if more_plots:
        raw.plot_psd(fmin=0, fmax=25, proj=False, verbose=True)

    epochs = mne.Epochs(raw, events, tmin=par['ctrlwin'][0], tmax=par['actiwin'][1],
                       baseline=(par['ctrlwin'][0],par['ctrlwin'][1]), picks=picks,
                       preload=True, proj=False, verbose=True)

    # Add desirable projectors
    if par['apply_ssp_eog'] & par['apply_ssp_ecg']:
        print('EOG+ECG:', subject)
        epochs.del_proj()
        epochs.add_proj(projs_eog[::3] + projs_ecg[::3]);
        epochs.apply_proj();
    elif par['apply_ssp_eog'] | par['apply_ssp_ecg']:
        if par['apply_ssp_eog']:
            print('EOG:', subject)
            epochs.del_proj()
            epochs.add_proj(projs_eog[::3]);
            epochs.apply_proj();
        else:
            print('ECG:', subject)
            epochs.del_proj()
            epochs.add_proj(projs_ecg[::3]);
            epochs.apply_proj();
    else:
        print('No SSP:', subject)
        epochs.del_proj()

    bad_trials = var_reject(epochs, par['plow'], par['phigh'], to_plot=True)
    plt.savefig(op.join(par['res_dir'], 'trial_variances_' + subject + '.pdf'))
    plt.close()
    epochs.drop(bad_trials, reason='variance based rejection', verbose=True);

    evoked = epochs.average()
    evoked.save(par['evoked_fname'])
    evoked.plot(spatial_colors=True, gfp=True, time_unit='ms')
    plt.savefig(op.join(par['res_dir'], 'evoked_' + subject + '.pdf'))
    plt.close()

    # plt.close('all')
    return epochs, evoked


# In[ ]:


def forward_solution(par, subject, subjects_dir, to_make=True):
    """
    Generate forwards solution and source space
    """
    src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir,
                                add_dist=False)

    model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.33,),
                          subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)

    if to_make:
        fwd = mne.make_forward_solution(par['info'], trans=par['trans_fname'],
                    src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=1)
        mne.write_forward_solution(par['fwd_fname'], fwd, overwrite=True)
    else:
        fwd = mne.read_forward_solution(par['fwd_fname'])

    fwd = mne.convert_forward_solution(fwd, surf_ori=True)

    print("Leadfield size : %d sensors x %d dipoles" % fwd['sol']['data'].shape)
    return fwd


# In[ ]:


def get_roi_labels(subject, subjects_dir, name, dominant='left'):
    if name == "LO":
        labels = mne.read_labels_from_annot(subject, parc='aparc', subjects_dir=subjects_dir)
        ROI = ['lateraloccipital-lh', 'lateraloccipital-rh']
        labels_roi = []
        for lbl in labels:
            if lbl.name in ROI:
                print(lbl.name)
                labels_roi.append(lbl)
        label = labels_roi[0]
        for i in range(1, len(labels_roi)):
            label = label + labels_roi[i]
        return label
    elif name == "Wernicke":
        labels = mne.read_labels_from_annot(subject, parc='PALS_B12_Brodmann', subjects_dir=subjects_dir)
        if dominant == 'left':
            ROI = ['Brodmann.22-lh']
        elif dominant == "right":
            ROI = ['Brodmann.22-rh']
        else:
            raise ValueError("$dominant can either be 'left' or 'right'. Check input value.")
        labels_roi = []
        for lbl in labels:
            if lbl.name in ROI:
                print(lbl.name)
                labels_roi.append(lbl)
        label = labels_roi[0]
        for i in range(1, len(labels_roi)):
            label = label + labels_roi[i]
        return label
    elif name == "Broca":
        labels = mne.read_labels_from_annot(subject, parc='PALS_B12_Brodmann', subjects_dir=subjects_dir)
        if dominant == 'left':
            ROI = ['Brodmann.44-lh', 'Brodmann.45-lh']
        elif dominant == "right":
            ROI = ['Brodmann.44-rh', 'Brodmann.45-rh']
        else:
            raise ValueError("$dominant can either be 'left' or 'right'. Check input value.")
        labels_roi = []
        for lbl in labels:
            if lbl.name in ROI:
                print(lbl.name)
                labels_roi.append(lbl)
        label = labels_roi[0]
        for i in range(1, len(labels_roi)):
            label = label + labels_roi[i]
        return label
    else:
        raise ValueError('No such label available. Check input value.')
    return None


# In[ ]:


def inverse_solution(par, subject, subjects_dir, epochs, evoked, fwd, to_save=True):
    """
    Compute inverse solution, estimate snr, and show cortical activations
    """
    noise_cov = mne.compute_covariance(epochs,
                    tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1],
                    method=['shrunk', 'empirical'], rank='info', verbose=True)

    inverse_operator = make_inverse_operator(par['info'], fwd, noise_cov,
                                        loose=0.2, depth=0.8)

    method, lambda2 = "dSPM", 1 / 3 ** 2
    stc = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None)
    if to_save:
        stc.save(par['stc_fname'])

    # Visualize dSPM values
    plt.figure()
    plt.plot(1e3 * stc.times, stc.data[::100, :].T)
    plt.xlabel('Time (ms)')
    plt.ylabel('%s value' % method)
    plt.savefig(op.join(par['res_dir'], 'evoked_dspm_100_' + subject + '.pdf'))
    plt.close()

    # Source localisation on cortical surface
    peak_times = [0.097, .142, .221, .337, .39]

    for tp in peak_times:
        surfer_kwargs = dict(
            hemi='both', subjects_dir=subjects_dir,
            clim=dict(kind='value', lims=[5, 10, 15]), views=['lateral', 'dorsal'],
            initial_time=tp, time_unit='s', size=(800, 800), smoothing_steps=10,
            time_viewer=False)
        brain = stc.plot(**surfer_kwargs)
        # Draw figure and save image
        dspm_fname = op.join(par['res_dir'], 'dspm_' + subject + '_%03f.png' % tp)
        brain.save_image(dspm_fname)
        brain.close()
    # Generate and save movie
    dspm_movie_fname = op.join(par['res_dir'], 'dspm_movie_' + subject + '.mov')
    brain = stc.plot(**surfer_kwargs)
    brain.save_movie(dspm_movie_fname, tmin=0.0, tmax=0.55, interpolation='linear',
                    time_dilation=20, framerate=10, time_viewer=True)
    brain.close()

    # Estimate peak SNR and find corresponding time
    snr, _ = estimate_snr(evoked, inverse_operator, verbose=True)
    nt_snr = np.argmax(snr)
    SNR = snr[nt_snr]
    print('\nMax SNR at %0.3f s : %0.3f' % (evoked.times[nt_snr], SNR))

    # Evoked responses of required brain sources
    src = inverse_operator['src']
    ROIs = ["LO", "Wernicke", "Broca"]

    if par['dominant_brain'] == 'find':
        # Run first for right
        evoked_roi = {}
        for roi in ROIs:
            label = get_roi_labels(subject, subjects_dir, roi, 'right')
            label_ts = mne.extract_label_time_course(stc, label, src, mode='mean_flip',
                                                    return_generator=True)
            label_ts = label_ts.transpose()
            evoked_roi[roi] = label_ts
        times = 1e3 * stc.times # Times in ms
        n_times = len(times)
        times = times.reshape((n_times, 1))
        # Draw time series
        plt.figure()
        h0 = plt.plot(times, evoked_roi["LO"], 'k', linewidth=3)
        h1, = plt.plot(times, evoked_roi["Wernicke"], 'g', linewidth=3)
        h2, = plt.plot(times, evoked_roi["Broca"], 'r', linewidth=3)
        plt.legend((h0[0], h1, h2), ('Lateral-Occipital', "Wernicke's", "Broca's"))
        plt.xlim([min(times), max(times)])
        plt.xlabel('Time (ms)')
        plt.ylabel('dSPM value')
        plt.grid(True)
        plt.savefig(op.join(par['res_dir'], 'evoked_dspm_label_' + subject + '_right-dominant.pdf'))
        plt.close()

        # Then run for left
        evoked_roi = {}
        for roi in ROIs:
            label = get_roi_labels(subject, subjects_dir, roi, 'left')
            label_ts = mne.extract_label_time_course(stc, label, src, mode='mean_flip',
                                                    return_generator=True)
            label_ts = label_ts.transpose()
            evoked_roi[roi] = label_ts
        times = 1e3 * stc.times # Times in ms
        n_times = len(times)
        times = times.reshape((n_times, 1))
        # Draw time series
        plt.figure()
        h0 = plt.plot(times, evoked_roi["LO"], 'k', linewidth=3)
        h1, = plt.plot(times, evoked_roi["Wernicke"], 'g', linewidth=3)
        h2, = plt.plot(times, evoked_roi["Broca"], 'r', linewidth=3)
        plt.legend((h0[0], h1, h2), ('Lateral-Occipital', "Wernicke's", "Broca's"))
        plt.xlim([min(times), max(times)])
        plt.xlabel('Time (ms)')
        plt.ylabel('dSPM value')
        plt.grid(True)
        plt.savefig(op.join(par['res_dir'], 'evoked_dspm_label_' + subject + '_left-dominant.pdf'))
        plt.close()
    elif par['dominant_brain'] == 'right':
        evoked_roi = {}
        for roi in ROIs:
            label = get_roi_labels(subject, subjects_dir, roi, 'right')
            label_ts = mne.extract_label_time_course(stc, label, src, mode='mean_flip',
                                                    return_generator=True)
            label_ts = label_ts.transpose()
            evoked_roi[roi] = label_ts
        times = 1e3 * stc.times # Times in ms
        n_times = len(times)
        times = times.reshape((n_times, 1))
        # Draw time series
        plt.figure()
        h0 = plt.plot(times, evoked_roi["LO"], 'k', linewidth=3)
        h1, = plt.plot(times, evoked_roi["Wernicke"], 'g', linewidth=3)
        h2, = plt.plot(times, evoked_roi["Broca"], 'r', linewidth=3)
        plt.legend((h0[0], h1, h2), ('Lateral-Occipital', "Wernicke's", "Broca's"))
        plt.xlim([min(times), max(times)])
        plt.xlabel('Time (ms)')
        plt.ylabel('dSPM value')
        plt.grid(True)
        plt.savefig(op.join(par['res_dir'], 'evoked_dspm_label_' + subject + '_right-dominant.pdf'))
        plt.close()
    elif par['dominant_brain'] == 'left':
        evoked_roi = {}
        for roi in ROIs:
            label = get_roi_labels(subject, subjects_dir, roi, 'left')
            label_ts = mne.extract_label_time_course(stc, label, src, mode='mean_flip',
                                                    return_generator=True)
            label_ts = label_ts.transpose()
            evoked_roi[roi] = label_ts
        times = 1e3 * stc.times # Times in ms
        n_times = len(times)
        times = times.reshape((n_times, 1))
        # Draw time series
        plt.figure()
        h0 = plt.plot(times, evoked_roi["LO"], 'k', linewidth=3)
        h1, = plt.plot(times, evoked_roi["Wernicke"], 'g', linewidth=3)
        h2, = plt.plot(times, evoked_roi["Broca"], 'r', linewidth=3)
        plt.legend((h0[0], h1, h2), ('Lateral-Occipital', "Wernicke's", "Broca's"))
        plt.xlim([min(times), max(times)])
        plt.xlabel('Time (ms)')
        plt.ylabel('dSPM value')
        plt.grid(True)
        plt.savefig(op.join(par['res_dir'], 'evoked_dspm_label_' + subject + '_left-dominant.pdf'))
        plt.close()
    else:
        raise ValueError("$dominant can either be 'left'/'right'/'find'. Check input value.")

    # plt.close('all')
    return None


# In[ ]:


def run_dspm(iSub, dominant_brain):
    par, subject, subjects_dir = set_params(iSub, dominant_brain=dominant_brain)
    epochs, evoked = preprocess(par, subject, subjects_dir)
    fwd = forward_solution(par, subject, subjects_dir)
    inverse_solution(par, subject, subjects_dir, epochs, evoked, fwd)


# In[ ]:


# run_dspm(1, dominant_brain="right")


# In[ ]:


# for i in range(1, 15):
# # for i in [0]:
#     sub = i+1
#     print('sub-%02d' % sub)
#     run_dspm(iSub=sub, dominant_brain="right")


# In[ ]:
