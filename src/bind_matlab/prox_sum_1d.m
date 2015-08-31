function [prox] = prox_sum_1d( idx, count, fn, a, b, c, d, e )
% computes prox of general class of functions
% sum_{i=1}^{count} c*f(ax - b) + dx + 0.5ex^2 
% fn is a string describing the 1d function f
data = { fn, a, b, c, d, e };
prox = { 'sum_1d', idx, count, 1, false, true, data };

end