function [func] = sum_ind_epi_polyhedral(dim, interleaved, a, b, count_vec, index_vec)
% SUM_IND_EPI_POLYHEDRAL   func = sum_ind_epi_polyhedral
%
%   Computes projection onto epigraph of function given as maximum
%   over linear functions specified by a, b, count_vec, index_vec.
   
    coeffs = { a, b, count_vec, index_vec };
    func = @(idx, count) { 'ind_epi_polyhedral', idx, count, false, ... 
                        { count / dim, dim, interleaved, coeffs } };
    
end
