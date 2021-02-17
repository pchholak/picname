function stats = stats_group(Amps, Lats, n, wts)

% Derive stats
[P_amps, M_amps] = power_anal_group(Amps, n, wts)
[P_lats, M_lats] = power_anal_group(Lats, n, wts)
