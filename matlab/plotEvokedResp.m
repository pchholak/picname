close all
clear
clc

% Given
t = -500 : 1000; Fs = 1000;
plot_groups = true; plot_rois = true;
group_names = {'Group-1', 'Group-2', 'Group-3', 'All'};
params.exc = [7];
params.twin = [0, 1];
params.r_prom = 0.05;
params.plot_erps = false;
params.ROIs = {'LO', 'Wernicke', 'Broca', 'Fusiform'};
params.res_path = '/home/wilson/research/results/picname';
params.groups = [2, 3, 1, 1, 1, 2, 0, 3, 3, 2, 2, 3, 1, 1, 2];
params.N = [5, 5, 4, 14]; % Sample size of each group
params.n_trials = [120, 120, 120, 120, 120, 120, 111, 98, 82, 89, 85, 87, ...
	87, 89, 83];

%% Create database of ERP features in all ROIs for all subjects
A = zeros(numel(params.groups), numel(params.ROIs)); L = A;
for q=1:numel(params.ROIs)
	[A(:,q), L(:,q)] = stats_find_erps_roi(params.ROIs{q}, params);
end

%% Evaluate ER and store in matrix
evoked = nan(numel(params.ROIs), numel(t), numel(params.N));
erp_feats = nan(numel(params.ROIs), numel(params.N), 2);
for i=1:numel(params.ROIs)
	for j=1:numel(params.N)-1
		inxSubs = params.groups == j;
		evoked(i, :, j) = getEvokedResp(inxSubs, params.ROIs{i}, params);
		wts = params.n_trials(inxSubs);
		erp_feats(:, j, 1) = (sum(A(inxSubs, :) .* wts') / sum(wts))';
		erp_feats(:, j, 2) = (sum(Fs * L(inxSubs, :) .* wts') / sum(wts))';
	end
	inxSubs = params.groups > 0;
	evoked(i, :, 4) = getEvokedResp(inxSubs, params.ROIs{i}, params);
	wts = params.n_trials(inxSubs);
	erp_feats(:, 4, 1) = (sum(A(inxSubs, :) .* wts') / sum(wts))';
	erp_feats(:, 4, 2) = (sum(Fs * L(inxSubs, :) .* wts') / sum(wts))';
end

% Assign error bar values
err_x = nan(numel(params.N), numel(params.ROIs)); err_y = err_x;
for j=1:numel(params.N)-1
	inxSubs = params.groups == j;
	err_y(j,:) = std(A(inxSubs,:));
	err_x(j,:) = std(Fs * L(inxSubs,:));
end
inxSubs = params.groups > 0;
err_y(4,:) = std(A(inxSubs,:));
err_x(4,:) = std(Fs * L(inxSubs,:));

%% Visualise
cmap = [0.7  0    0;
		0    0.7  0;
		0    0    0.7;
		0    0    0];
if plot_rois
	for j=1:numel(params.N)
		figure('Position', [50, 50, 900, 600], 'PaperPositionMode', 'auto', ...
		'PaperOrientation', 'landscape', 'DefaultAxesColorOrder', cmap)
		plot(t, evoked(:, :, j), 'LineWidth', 2)
		hold on
		for i=1:numel(params.ROIs)
			errorbar(erp_feats(i,j,2), erp_feats(i,j,1), err_y(j,i), 'o', ...
			'Color', cmap(i,:), 'LineWidth', 2)
			errorbar(erp_feats(i,j,2), erp_feats(i,j,1), err_x(j,i), ...
			'horizontal', 'Color', cmap(i,:), 'LineWidth', 2)
		end
		hold off
		xlabel('Time (ms)')
		ylabel('dSPM value')
		set(gca, 'FontSize', 22)
		grid minor
		title(group_names{j})
		legend(params.ROIs, 'FontSize', 20)
		fname_image = fullfile(params.res_path, ...
		['evoked_label_', group_names{j}, '.pdf']);
		print(gcf, fname_image, '-dpdf', '-fillpage')
	end
end

if plot_groups
	for i = 1:numel(params.ROIs)
		figure('Position', [50, 50, 900, 600], 'PaperPositionMode', 'auto', ...
		'PaperOrientation', 'landscape', 'DefaultAxesColorOrder', cmap)
		plot(t, permute(evoked(i, :, 1:3), [3 2 1]), 'LineWidth', 2)
		hold on
		for j=1:numel(params.N)-1
			errorbar(erp_feats(i,j,2), erp_feats(i,j,1), err_y(j,i), 'o', ...
			'Color', cmap(j,:), 'LineWidth', 2)
			errorbar(erp_feats(i,j,2), erp_feats(i,j,1), err_x(j,i), ...
			'horizontal', 'Color', cmap(j,:), 'LineWidth', 2)
		end
		hold off
		xlabel('Time (ms)')
		ylabel('dSPM value')
		set(gca, 'FontSize', 22)
		grid minor
		title(params.ROIs{i})
		legend(group_names(1:3), 'FontSize', 20)
		fname_image = fullfile(params.res_path, ...
		['evoked_label_', params.ROIs{i}, '.pdf']);
		print(gcf, fname_image, '-dpdf', '-fillpage')
	end
end
