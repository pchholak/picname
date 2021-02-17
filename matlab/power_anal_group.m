function [P, Wavg] = power_anal_group(V, n, wts)
% Perform group-wise power analysis for variable 'V'

Wavg = zeros(1, 4); S = Wavg;
for k=1:4
	x = V(:,k)';
	Wavg(k) = sum(wts.*x) / sum(wts);
	S(k) = std(x);
end

P = zeros(4);

for i=1:3
	for j=i+1:4
		s = sqrt( ( S(i)^2 + S(j)^2 ) / 2 );
		if Wavg(i) > Wavg(j)
			pwr = sampsizepwr('t', [Wavg(i), s], Wavg(j), [], n, 'Tail', 'left');
		else
			pwr = sampsizepwr('t', [Wavg(i), s], Wavg(j), [], n, 'Tail', 'right');
		end
		P(i,j) = pwr;
	end
end
