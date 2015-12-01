function [prox] = prox_1d(idx, count, fn, a, b, c, d, e, alpha, beta)
% computes prox of general class of functions
% sum_{i=1}^{count} c*f(ax - b) + dx + 0.5ex^2 
% fn is a string describing the 1d function f
    
    
switch nargin
  case 8
    alpha = 0;
    beta = 0;
end

data = { fn, a, b, c, d, e, alpha, beta };
prox = { '1d', idx, count, 1, false, true, data };

end