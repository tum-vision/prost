function [prox] = sum_1d(idx, count, fun, a, b, c, d, e, alpha, beta)
% computes prox of general class of functions
% sum_{i=1}^{count} c*f(ax - b) + dx + 0.5ex^2 
% fn is a string describing the 1d function f
% alpha and beta are parameters which further parametrize
% the function f.
    
switch nargin
  case 8
    alpha = 0;
    beta = 0;
    
  case 9
    beta = 0;
end

coeffs = { a, b, c, d, e, alpha, beta };
prox = { strcat('elem_operation:1d:', fun), idx, count, true, { ...
    count, 1, false, coeffs } };

end
