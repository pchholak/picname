#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'qt')
import mne
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
from nilearn.plotting import plot_stat_map
from nilearn.image import index_img
import warnings
from mne_bids import read_raw_bids, make_bids_basename
from mne.time_frequency import tfr_morlet

mne.set_log_level('WARNING')
warnings.simplefilter("ignore", category=DeprecationWarning)


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


def run_lcmv(iSub):
    par, subject, subjects_dir = set_params(iSub)
    epochs, evoked = preprocess(par, subject, subjects_dir)
    # fwd = forward_solution(par, subject, subjects_dir)
    # frequency_anal(par, subject, epochs)
    # inverse_solution(par, subject, subjects_dir, epochs, evoked, fwd)


# In[1]:


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
    par['epochs_fname'] = op.join(par['bids_path'],
                                  par['bids_fname'].replace('_meg.fif', '-epo.fif'))
    par['evoked_fname'] = op.join(par['res_dir'], subject+'-ave.fif')
    par['trans_fname'] = op.join(par['bids_path'], subject+'-trans.fif')
    par['fwd_fname'] = op.join(par['bids_path'], subject + '-vol-meg-fwd.fif')
    par['mrifile'] = op.join(subjects_dir, subject, 'mri/T1.mgz')
    par['surffile'] = op.join(subjects_dir, subject,
                              'bem/watershed', subject+'_brain_surface')
    par['bem_fname'] = op.join(par['bids_path'], subject + '-bem.fif')
    par['stc_fname'] = op.join(par['res_dir'], 'lcmv_' + subject)
    par['info'] = mne.io.read_info(par['raw_fname'])

    return par, subject, subjects_dir


# In[2]:


def preprocess(par, subject, subjects_dir, make_watershed=False, review_raw=False,
               more_plots=False):
    """
    Preprocess data, load epochs, and get evoked response
    """
    raw = read_raw_bids(par['bids_fname'], par['data_path'],
                        extra_params=dict(allow_maxshield=False, preload=True))

    if make_watershed:
        mne.bem.make_watershed_bem(subject, subjects_dir)

    if review_raw:
        raw.plot();
        raw.annotations.save(op.join(par['bids_path'], subject + '-annot.csv'))

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
    # epochs.save(par['epochs_fname'], overwrite=True)

    # epochs.plot_image(picks='MEG2343', sigma=1);
    # plt.savefig(op.join(par['res_dir'], 'epochs_MEG2343_' + subject + '.pdf'))

    evoked = epochs.average()
    evoked.save(par['evoked_fname'])

    # evoked.plot(spatial_colors=True, gfp=True, proj=False, time_unit='ms')
    # plt.savefig(op.join(par['res_dir'], 'evoked_' + subject + '.pdf'))
    #
    # times = [0.15, 0.25] # Highly data dependent
    # evoked.plot_joint(times=times, picks='mag');
    # plt.savefig(op.join(par['res_dir'], 'evoked_joint_plot' + subject + '.pdf'))
    #
    # evoked.plot_topomap(times=np.linspace(0.1, 0.3, 5), ch_type='mag');
    # plt.savefig(op.join(par['res_dir'], 'evoked_topomap_mag_' + subject + '.pdf'))
    #
    # plt.close('all')

    return epochs, evoked


# In[6]:


def frequency_anal(par, subject, epochs):
    """
    Frequency analysis and inter-trial coherence
    """
    freqs = np.logspace(*np.log10([2, 30]), num=20)
    n_cycles = freqs / 2

    epochs.plot_psd_topomap(ch_type='mag', normalize=True, cmap='viridis');
    plt.savefig(op.join(par['res_dir'], 'psd_topomap_mag_' + subject + '.pdf'))

    epochs.plot_psd_topomap(ch_type='grad', normalize=True, cmap='viridis');
    plt.savefig(op.join(par['res_dir'], 'psd_topomap_grad_' + subject + '.pdf'))

    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                           return_itc=True, decim=3, n_jobs=1)

    power.crop(-0.1, 0.8)
    itc.crop(-0.1, 0.8)
    baseline_mode = 'logratio'
    baseline = (par['ctrlwin'][0], par['ctrlwin'][1])

    power.plot([267], baseline=baseline, mode=baseline_mode);
    plt.savefig(op.join(par['res_dir'], 'power_MEG2343_' + subject + '.pdf'))

    fig, axis = plt.subplots(1, 3, figsize=(7, 4))
    power.plot_topomap(ch_type='grad', tmin=0., tmax=0.6, fmin=4, fmax=7,
                      baseline=baseline, mode=baseline_mode, axes=axis[0],
                      title='Theta', show=False, contours=1)
    power.plot_topomap(ch_type='grad', tmin=0., tmax=0.6, fmin=8, fmax=12,
                      baseline=baseline, mode=baseline_mode, axes=axis[1],
                      title='Alpha', show=False, contours=1)
    power.plot_topomap(ch_type='grad', tmin=0., tmax=0.6, fmin=15, fmax=30,
                      baseline=baseline, mode=baseline_mode, axes=axis[2],
                      title='Beta', show=False, contours=1)
    mne.viz.tight_layout()
    plt.show()
    plt.savefig(op.join(par['res_dir'], 'power_topomap_grad_t=0-0p6_' + subject + '.pdf'))

    itc.plot([267], baseline=baseline, mode=baseline_mode);
    plt.savefig(op.join(par['res_dir'], 'itc_MEG2343_' + subject + '.pdf'))

    itc.plot_topomap(ch_type='mag', tmin=0.1, tmax=0.3, fmin=3.5, fmax=7.,
                baseline=baseline, mode='mean', size=6)
    mne.viz.tight_layout()
    plt.savefig(op.join(par['res_dir'], 'itc_topomap_' + subject + '.pdf'))

    plt.close('all')

    return None


# In[7]:


def forward_solution(par, subject, subjects_dir, write_bem=True, to_make=True):
    """
    Generate forwards solution and source space
    """
    src = mne.setup_volume_source_space(subject=subject, pos=5.0,
                mri=par['mrifile'], bem=None, surface=par['surffile'], mindist=2.5,
                exclude=10, subjects_dir=subjects_dir, volume_label=None,
                add_interpolator=None, verbose=True)

    model = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.33,),
                subjects_dir=subjects_dir, verbose=True)
    bem = mne.make_bem_solution(model)
    if write_bem:
        mne.bem.write_bem_solution(par['bem_fname'], bem)

    if to_make:
        fwd = mne.make_forward_solution(par['info'], trans=par['trans_fname'], src=src,
                    bem=bem, meg=True, eeg=False, mindist=2.5, n_jobs=1)
        mne.write_forward_solution(par['fwd_fname'], fwd, overwrite=True)
    else:
        fwd = mne.read_forward_solution(par['fwd_fname'])

    print("Leadfield size : %d sensors x %d dipoles" % fwd['sol']['data'].shape)

    return fwd


# In[8]:


def inverse_solution(par, subject, subjects_dir, epochs, evoked, fwd, to_save=True):
    """
    Compute inverse solution, estimate snr, and show cortical activations
    """
    noise_cov = mne.compute_covariance(epochs,
                    tmin=par['ctrlwin'][0], tmax=par['ctrlwin'][1],
                    method='empirical', rank='info', verbose=True)

    data_cov = mne.compute_covariance(epochs,
                    tmin=par['actiwin'][0], tmax=par['actiwin'][1],
                    method='empirical', rank='info', verbose=True)

    evoked.plot_white(noise_cov);
    plt.savefig(op.join(par['res_dir'], 'evoked_plot_white_' + subject + '.pdf'))

    inverse_operator = mne.minimum_norm.make_inverse_operator(par['info'],
                    fwd, noise_cov, rank=None, loose=1, depth=0.199, verbose=True)

    filters = mne.beamformer.make_lcmv(par['info'], fwd, data_cov, reg=0.05,
                noise_cov=noise_cov, pick_ori='max-power', rank=None,
                weight_norm='nai', reduce_rank=True, verbose=True)

    stc = mne.beamformer.apply_lcmv(evoked, filters, max_ori_out='signed', verbose=True)
    if to_save:
        stc.save(par['stc_fname'])

    stc = np.abs(stc)
    _, t_peak = stc.get_peak()
    print('Absolute source peaked at = %0.3f' % t_peak)
    nt_src_peak = int(t_peak//stc.tstep - stc.times[0]//stc.tstep)

    snr, _ = mne.minimum_norm.estimate_snr(evoked, inverse_operator, verbose=True)
    nt_snr = np.argmax(snr)
    SNR = snr[nt_snr]
    print('\nMax SNR at %0.3f s : %0.3f' % (evoked.times[nt_snr], SNR))

    img = stc.as_volume(fwd['src'], dest='mri', mri_resolution=False, format='nifti1')

    plot_stat_map(index_img(img, nt_src_peak), par['mrifile'], threshold=stc.data.max()*0.70)
    plt.savefig(op.join(par['res_dir'],
            'stat_map_' + 'time=%0.3fs_'%(stc.times[nt_src_peak]) + subject + '.pdf'))

    plot_stat_map(index_img(img, nt_snr), par['mrifile'], threshold=stc.data.max()*0.30)
    plt.savefig(op.join(par['res_dir'],
            'stat_map_' + 'time=%0.3fs_'%(stc.times[nt_snr]) + subject + '.pdf'))

    return None
