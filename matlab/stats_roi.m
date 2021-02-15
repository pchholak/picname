function stats = stats_roi(Amps, Lats, params)

% Derive stats
[P_amps, M_amps] = power_anal_roi(Amps, params)
[P_lats, M_lats] = power_anal_roi(Lats, params)
