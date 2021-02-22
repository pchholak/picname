function evoked = getEvokedResp(iSubs, roi, params)

% Load evoked data
fname_evoked = fullfile(params.res_path, ['evoked_matrix_', roi, '.mat']);
load(fname_evoked)

% Add NaN series for Subject-7
stcs = [stcs(1:6, :); nan(1, numel(t)); stcs(7:14, :)];

% Calculate evoked response using weighted average
wts = params.n_trials(iSubs);
evoked = sum(stcs(iSubs, :) .* wts') / sum(wts);
