function [func] = sum_ind_halfspace(dim, interleaved, a, b)
% SUM_IND_EPI_QUAD  func = sum_ind_epi_quad(dim, interleaved, a, b)
%
%   Computes projection onto a halfspace given by 
%   a^T x <= b.
%
%   The interleaved keyword is ignored by the prox (always set to false).
%
%   Let count be the number of functions in the sum. Then:
%   - Dimension of a can be either dim or dim * count.
%   - Dimension of b can be either 1 or count. 
%

    coeffs = { a, b };

    func = @(idx, count) { 'ind_halfspace', idx, count, false, ... 
                        { count / dim, dim, interleaved, coeffs } };

end
