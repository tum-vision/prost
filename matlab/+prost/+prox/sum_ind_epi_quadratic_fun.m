function [prox] = sum_ind_epi_quadratic_fun(idx, count, dim, interleaved, a, b, c)

coeffs = { a, b, c };
prox = { 'ind_epi_quadratic_fun', idx, count*dim, false, ... 
         { count, dim, interleaved, coeffs } };

end
