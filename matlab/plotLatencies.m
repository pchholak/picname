close all
clear
clc

%% Given
exc = [7];
subj_indices = 1:15; nsub = length(subj_indices);
label_gap = .1;
stroop = [70.00, 60.98, 69.98, 75.54, 73.35, 72.78, 62.54, 61.34, 63.00, ...
          69.65, 69.78, 59.60, 69.97, 73.54, 74.00];
oxf =    [50.7, 55.4, 38.9, 42.2, 39.7, 46.3, 53.2, 57.0, 56.5, 45.8, 45.5, ...
          55.2, 40.2, 44.0, 43.7];
P = [285, 266, 445, 299, 412, 329;
     276, 216, 539, 234, 606, 658;
     286, 334, 634, 128, 372, 461;
     270, 106, 395, 183, 339, 298;
     304, 304, 636, 110, 330, 242;
     232, 258, 458, 160, 426, 367;
     NaN, NaN, NaN, NaN, NaN, NaN;
     191, 208, 453, 208, 437, 281;
     288, 140, 364, 140, 434, 448;
     202, 252, 481, 167, 446, 245;
     233, 249, 386, 189, 359, 261;
     261, 146, 387, 148, 336, 278;
     163, 226, 697, 141, 415, 279;
     221, 174, 397, 198, 351, 274;
     226, 166, 592, 206, 571, 256];

names = cell(1, nsub);
for iSub=1:nsub
    names{iSub} = sprintf("%d", iSub);
end

%% ERTD
ertd_BA = P(:,3) - P(:,2);
ertd_FG = P(:,5) - P(:,4);

%% Visualise
figure
[~, lm] = custom_scatter(oxf, ertd_BA, exc, label_gap, names);
xlabel('Proficiency'), ylabel('ERTD-BA')
grid
set(gca, 'FontSize', 14)
% pval = lm.Coefficients.pValue; str_pval = sprintf('p-val = %0.3f', pval(2));
% text(.62, .85, str_pval, 'Units', 'normalized', 'FontSize', 14, 'Color', 'r')
% fname_plot = fullfile(info.res_path_coh, 'Coherence_Noise.png');
% saveas(gcf, fname_plot);

% figure
% custom_scatter(N, Pm, exc, label_gap, names);
% xlabel('Noise'), ylabel('P')
% grid
% set(gca, 'FontSize', 14)
