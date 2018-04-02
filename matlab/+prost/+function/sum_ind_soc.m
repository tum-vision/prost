function [func] = sum_ind_soc(dim, interleaved, alpha)
% SUM_IND_SOC  func = sum_ind_soc(dim, interleaved, alpha)
%
%   Computes projection onto the second order cone
%   \alpha ||x|| \leq y
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

    func = @(idx, count) { 'ind_soc', idx, count, false, ... 
                        { count / dim, dim, interleaved, alpha } };

end
