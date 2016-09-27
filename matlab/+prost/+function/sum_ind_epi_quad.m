function [func] = sum_ind_epi_quad(dim, interleaved, a, b, c)
% SUM_IND_EPI_QUAD  func = sum_ind_epi_quad(dim, interleaved, a,b,c)
%
%   Computes projection onto epigraph of a parabola function given as
%   a x^T x + b^T x + c
%
%   Assumes that the input variables have the same ordering,
%   the interleaved keyword is ignored by the prox.
%
%   The ordering of the variables for a d-dimensional projection
%   of (x, y) should be 
%
%   (x_1, ..., x_d, y), where the size of x_1 ... x_d and y should
%   be the same (e.g. number of pixels) 
%

    coeffs = { a, b, c };

    func = @(idx, count) { 'ind_epi_quad', idx, count, false, ... 
                        { count / dim, dim, interleaved, coeffs } };

end
