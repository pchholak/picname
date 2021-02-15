close all
clear
clc

% Given
params.ROIs = {'LO', 'Wernicke', 'Broca', 'Fusiform'};
params.exc = [7];
params.twin = [0, 1];
params.r_prom = 0.05;
params.plot_erps = false;
params.res_path = '/home/wilson/research/results/picname';
params.group = [2, 3, 1, 1, 1, 2, 0, 3, 3, 2, 2, 3, 1, 1, 2];
params.N = [5, 5, 4, 14]; % Sample size of each group
params.n_trials = [120, 120, 120, 120, 120, 120, 111, 98, 82, 89, 85, 87, ...
	87, 89, 83];
% params.n_trials = 120 * ones(1, 15);

% Create database of evoked reponses in all ROIs for all subjects
Amps = zeros(numel(params.group), numel(params.ROIs)); Lats = Amps;
for q=1:numel(params.ROIs)
	[Amps(:,q), Lats(:,q)] = stats_find_erps_roi(params.ROIs{q}, params);
end

% % Find statistics of relative ERPs in ROIs between groups
% for q=1:numel(params.ROIs)
% 	sprintf('Printing statistics for %s', params.ROIs{q})
% 	stats_roi(Amps(:,q), Lats(:,q), params);
% end

% Find statistics of relative ERPs in groups between ROIs
for k=1:4
	sprintf('Printing statistics for Group-%d', k)
	if k<4
		inx = params.group == k;
		wts = params.n_trials(inx);
		stats_group(Amps(inx,:), Lats(inx,:), params.N(k), wts);
	else
		inx = params.group > 0;
		wts = params.n_trials(inx);
		stats_group(Amps(inx,:), Lats(inx,:), params.N(k), wts);
	end
end
