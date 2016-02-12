function [prox] = sum_ind_epi_conjquad(idx, count, dim, interleaved, ...
                                       a, b, c, alpha, beta)

coeffs = { a, b, c, alpha, beta };
prox = { 'ind_epi_conjquad', idx, count*dim, false, ...
         { count, dim, interleaved, coeffs } };

end
