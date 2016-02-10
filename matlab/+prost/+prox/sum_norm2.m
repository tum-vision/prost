function [prox] = sum_norm2(idx, count, dim, interleaved, fun, a, ...
                             b, c, d, e, alpha, beta)
% computes prox of general class of functions
% sum_{i=1}^{count} h(||x||_2), where ||.||_2 denotes the dim-dimensional
% euclidean norm. h is given as h(x) = c f(ax - b) + dx + 0.5ex^2 
% fn is a string describing the 1d function f

switch nargin
  case 11
    beta = 0;
    
  case 10
    alpha = 0;
    beta = 0;
end

coeffs = { a, b, c, d, e, alpha, beta };
prox = { strcat('elem_operation:norm2:', fun), idx, count*dim, false, { count, dim, interleaved, coeffs } };

end
