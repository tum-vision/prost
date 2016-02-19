function [prox] = sum_ind_epi_polyhedral(idx, count, dim, a, b, count_vec, index_vec)
% computes projection onto epigraph of function given as maximum
% over linear functions specified by a, b, count_vec, index_vec.
    
    coeffs = { a, b, count_vec, index_vec };
    prox = { 'ind_epi_polyhedral', idx, count*dim, false, ... 
             { count, dim, false, coeffs } };

end
