function [prox] = sum_ind_epi_quad(idx, count, dim, interleaved, a, b, c)

    coeffs = { a, b, c };
    prox = { 'ind_epi_quad', idx, count*dim, false, ... 
             { count, dim, interleaved, coeffs } };

end
