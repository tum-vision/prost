function [func] = sum_ind_epi_quad(dim, interleaved, a, b, c)
% SUM_IND_EPI_QUAD  func = sum_ind_epi_quad(dim, interleaved, a,b,c)
%
%   Computes projection onto epigraph of a parabola function given as
%   a x^T x + b^T x + c

    func = @(idx, count) prost.prox.sum_ind_epi_quad(idx, count / dim, ...
                                                     dim, interleaved, ...
                                                     a, b, c);

end
