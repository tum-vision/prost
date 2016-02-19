function [prox] = sum_singular_nx2(idx, count, dim, interleaved, fun, a, ...
                             b, c, d, e, alpha, beta)
                         
switch nargin
  case 11
    beta = 0;
    
  case 10
    alpha = 0;
    beta = 0;
end

coeffs = { a, b, c, d, e, alpha, beta };
prox = { strcat('elem_operation:singular_nx2:', fun), idx, count*dim, false, { count, dim, interleaved, coeffs } };

end
