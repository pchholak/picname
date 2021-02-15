function [Amps, Lats] = stats_find_erps_roi(roi, params)

% Load evoked data
fname_evoked = fullfile(params.res_path, ['evoked_matrix_', roi, '.mat']);
load(fname_evoked)

% Add NaN series for Subject-7
stcs = [stcs(1:6, :); nan(1, numel(t)); stcs(7:14, :)];

% Find peak responses
nSub = size(stcs, 1);
subjects = 1:nSub; subjects(params.exc) = [];
Amps = nan(1, nSub); Lats = Amps;
for iSub=subjects
	[Amps(iSub), Lats(iSub)] = find_erps_ordered(stcs(iSub, :), t, ...
	params.twin, params.r_prom, params.plot_erps);
end
