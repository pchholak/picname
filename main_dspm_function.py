#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'qt')
import mne
import numpy as np
import matplotlib.pyplot as plt
import os.path as op

import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

from mne_bids import read_raw_bids, make_bids_basename

mne.set_log_level('WARNING')

from mne.minimum_norm import (make_inverse_operator, apply_inverse, write_inverse_operator,
                             estimate_snr)

from mayavi import mlab
from IPython.display import Image


# In[2]:


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
    print('Removed trials: %s\n'%bad_trials)
    return bad_trials


# In[3]:


def set_params(iSub, ctrlwin=[-0.5,0], actiwin=[0,1], plow=2, phigh=98):
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
    par['res_dir'] = op.join(op.expanduser("~/research/results/pic_name"), subject)

    par['bids_basename'] = make_bids_basename(subject=sSub, session=session,
                                      task=task, run=run)
    par['bids_fname'] = par['bids_basename'] + '_meg.fif'
    par['bids_path'] = op.join(par['data_path'], subject, 'ses-'+session, 'meg')
    par['raw_fname'] = op.join(par['bids_path'], par['bids_fname'])
    par['trans_fname'] = op.join(par['bids_path'], subject+'-trans.fif')
    par['fwd_fname'] = op.join(par['bids_path'], subject + '-cort-meg-fwd.fif')
    par['mrifile'] = op.join(subjects_dir, subject, 'mri/T1.mgz')
    par['surffile'] = op.join(subjects_dir, subject,
                              'bem/watershed', subject+'_brain_surface')
    par['stc_fname'] = op.join(par['res_dir'], 'dspm_' + subject)
    par['info'] = mne.io.read_info(par['raw_fname'])

    return par, subject, subjects_dir


# In[4]:


def preprocess(par, subject, subjects_dir, more_plots=False):
    """
    Preprocess data, load epochs, and get evoked response
    """
    raw = read_raw_bids(par['bids_fname'], par['data_path'],
                        extra_params=dict(allow_maxshield=False, preload=True))
#     raw.plot();
#     raw.annotations.save(op.join(par['bids_path'], subject + '-annot.csv'))

    events, event_id = mne.events_from_annotations(raw)
    if more_plots:
        mne.viz.plot_events(events, first_samp=0, event_id=event_id,
                           equal_spacing=True, show=True)

    picks = mne.pick_types(raw.info, meg=True, eog=True, ecg=True, stim=False, exclude='bads')

    raw.filter(2, 40, picks=picks, filter_length='auto', n_jobs=1,
          method='fir', iir_params=None, phase='zero', fir_window='hamming',
          fir_design='firwin', skip_by_annotation=('edge', 'bad_acq_skip'),
          pad='reflect_limited', verbose=True)
    if more_plots:
        raw.plot_psd(fmin=0, fmax=45, proj=False, verbose=True)

    epochs = mne.Epochs(raw, events, event_id, par['ctrlwin'][0], par['actiwin'][1],
                       baseline=(par['ctrlwin'][0],par['ctrlwin'][1]), picks=picks,
                       preload=True, reject=None, flat=None, proj=False, decim=1,
                       reject_tmin=None, reject_tmax=None, detrend=None,
                       on_missing='error', reject_by_annotation=True,
                       verbose=True)
    epochs.pick_types(meg=True)

    bad_trials = var_reject(epochs, par['plow'], par['phigh'], to_plot=False)
    epochs.drop(bad_trials, reason='variance based rejection', verbose=True)

    evoked = epochs.average()
    evoked.plot(spatial_colors=True, gfp=True, proj=False, time_unit='ms')

    return epochs, evoked


# In[5]:


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


# In[6]:


def inverse_solution(par, subject, subjects_dir, epochs, evoked, fwd, to_save=True):
    """
    Compute inverse solution, estimate snr, and show cortical activations
    """
    noise_cov = mne.compute_covariance(epochs,
                    tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1],
                    method='empirical', rank='info', verbose=True)

    inverse_operator = make_inverse_operator(par['info'], fwd, noise_cov,
                                        loose=0.2, depth=0.8)

    method, lambda2 = "dSPM", 1 / 3 ** 2
    stc = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None)
    if to_save:
        stc.save(par['stc_fname'])

    stc_abs = np.abs(stc)
    _, t_peak = stc_abs.get_peak()
    print('Absolute source peaked at = %0.3f' % t_peak)
    nt_src_peak = int(t_peak//stc.tstep - stc.times[0]//stc.tstep)

    snr, _ = estimate_snr(evoked, inverse_operator, verbose=True)
    nt_snr = np.argmax(snr)
    SNR = snr[nt_snr]
    print('\nMax SNR at %0.3f s : %0.3f' % (evoked.times[nt_snr], SNR))

    brain = stc.plot(surface='inflated', hemi='both', subjects_dir=subjects_dir,
                time_viewer=False)
    brain.set_data_time_index(nt_src_peak)
    brain.scale_data_colormap(fmin=4, fmid=7, fmax=10, transparent=True)
    brain.show_view('parietal')

    dspm_fname = op.join(par['res_dir'], 'dspm_' + subject + '.png')
    brain.save_image(dspm_fname)
    mlab.close()

    labels = mne.read_labels_from_annot(subject, 'HCPMMP1', 'both', subjects_dir=subjects_dir)
    labels_vis = []
    ROI = ['L_V1_ROI-lh', 'R_V1_ROI-rh']
    for lbl in labels:
        if lbl.name in ROI:
            labels_vis.append(lbl)
    label = labels_vis[0]
    for i in range(1, len(labels_vis)):
        label = label + labels_vis[i]

    flip = mne.label_sign_flip(label, inverse_operator['src'])

    stc_evoked = apply_inverse(evoked, inverse_operator, lambda2, method, pick_ori="normal")
    stc_evoked_label = stc_evoked.in_label(label)
    label_mean_evoked = np.mean(stc_evoked_label.data, axis=0)
    label_mean_evoked_flip = np.mean(flip[:, np.newaxis] * stc_evoked_label.data, axis=0)

    times = 1e3 * stc_evoked.times # times in ms
    plt.figure()
    h0 = plt.plot(times, stc_evoked_label.data.T, 'k')
    h1, = plt.plot(times, label_mean_evoked, 'r', linewidth=3)
    h2, = plt.plot(times, label_mean_evoked_flip, 'g', linewidth=3)
    plt.legend((h0[0], h1, h2), ('all dipoles in label', 'mean', 'mean with flip'))
    plt.xlabel('time (ms)')
    plt.ylabel('dSPM value')
    plt.show()
    plt.savefig(op.join(par['res_dir'], 'evoked_label_' + subject + '.pdf'))


# In[7]:


def run_dspm(iSub):
    badtrls = []
    par, subject, subjects_dir = set_params(iSub)
    epochs, evoked = preprocess(par, subject, subjects_dir)
    fwd = forward_solution(par, subject, subjects_dir)
    inverse_solution(par, subject, subjects_dir, epochs, evoked, fwd)
