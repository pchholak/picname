function [amps, lats] = find_erps_ordered(x, t, twin, r_prom, to_plot)
% Find evoked response peak amplitudes and latencies.

% ===== PARSE INPUTS =====
if nargin < 5 || isempty(to_plot)
    to_plot = false;
end

% Fs = 1 / (t(2)-t(1));
iwin = (t>=twin(1)) & (t<=twin(2));
y = x(iwin); t0 = t(iwin);

[pks, locs] = findpeaks(y, t0, 'MinPeakProminence', r_prom*peak2peak(y), ...
				'SortStr', 'descend');
if numel(pks)<2
	[pks, locs] = findpeaks(y, t0, 'SortStr', 'descend');
end
amps = pks(1); lats = locs(1);

if to_plot
	figure
	plot(t, x, locs, pks, 'kv', 'MarkerFaceColor', 'k')
	hold on
	plot(lats, amps, 'mv', 'MarkerFaceColor', 'm')
	if numel(pks)>1
		plot(locs(2), pks(2), 'gv', 'MarkerFaceColor', 'g')
	end
end
