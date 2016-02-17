function [prox] = sum_singular_3x2(idx, count, interleaved, fun, a, ...
                             b, c, d, e, alpha, beta)
                         
switch nargin
  case 11
    beta = 0;
    
  case 10
    alpha = 0;
    beta = 0;
end

coeffs = { a, b, c, d, e, alpha, beta };
prox = { strcat('elem_operation:singular_3x2:', fun), idx, count*6, false, { count, 6, interleaved, coeffs } };

end
