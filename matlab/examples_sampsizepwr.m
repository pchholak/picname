close all
clear
clc

%% Example-1
% nout = sampsizepwr('t', [100 5], 102, 0.80);
%
% nn = 1:100;
% pwrout = sampsizepwr('t', [100 5], 102, [], nn);

% figure
% plot(nn, pwrout, 'b-', nout, 0.8, 'ro')
% title('Power versus Sample Size')
% xlabel('Sample Size')
% ylabel('Power')

%% Example-2
% power = sampsizepwr('t', [20 5], 25, [], 5, 'Tail', 'right');
% beta = 1 - power;
%
% nout = sampsizepwr('t', [20 5], 25, 0.99, [], 'Tail', 'right');
%
% p1out = sampsizepwr('t', [20 5], [], 0.95, 10, 'Tail', 'right');

%% Example-3
% napprox = sampsizepwr('p', 0.30, 0.36, 0.8);
%
% nn = 1:500;
% pwrout = sampsizepwr('p', 0.30, 0.36, [], nn);
% nexact = min(nn(pwrout>=0.8))
%
% figure
% plot(nn, pwrout, 'b-', [napprox, nexact], pwrout([napprox, nexact]), 'ro')
% grid on

%% Example-4
pwr = sampsizepwr('t2', [1.4 0.2], 1.7, [], 5, 'Ratio', 2);

n = sampsizepwr('t2', [1.4 0.2], 1.7, 0.9, []);

[n1out, n2out] = sampsizepwr('t2', [1.4 0.2], 1.7, 0.9, [], 'Ratio', 2);
