function [prox] = sum_ind_epi_polyhedral(idx, count, dim, interleaved, a, b, count_vec, index_vec)
    
    coeffs = { a, b, count_vec, index_vec };
    prox = { 'ind_epi_polyhedral', idx, count*dim, false, ... 
             { count, dim, interleaved, coeffs } };

end
