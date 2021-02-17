function [P, Wavg] = power_anal_roi(V, params)
% Perform group-wise power analysis for variable 'V'
Wavg = zeros(1, 3); S = Wavg;
for g=1:3
	wts = params.n_trials(params.group == g);
	x = V(params.group == g)';
	Wavg(g) = sum(wts.*x) / sum(wts);
	S(g) = std(x);
end

P = zeros(3);

for i=1:2
	for j=i+1:3
		s = sqrt( ( (params.N(i)-1)*S(i)^2 + (params.N(j)-1)*S(j)^2 ) / ...
					( params.N(i) + params.N(j) - 2 ) );
		if Wavg(i) > Wavg(j)
			pwr = sampsizepwr('t', [Wavg(i), s], Wavg(j), [], ...
				min([params.N(i), params.N(j)]), 'Tail', 'left');
		else
			pwr = sampsizepwr('t', [Wavg(i), s], Wavg(j), [], ...
				min([params.N(i), params.N(j)]), 'Tail', 'right');
		end
		P(i,j) = pwr;
	end
end
