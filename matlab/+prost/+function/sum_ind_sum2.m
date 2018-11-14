function [func] = sum_ind_sum2(dim, inds, s1, dim2, inds2, s2)
% SUM_IND_SUM2  func = sum_ind_sum2(dim, interleaved)
%
%   Implements the seperable sum of dim-dimensional indicator
%   functions of the sum-to-one constraint, i.e., sum_i x_i = 1
%   according to index arrays.
    
    if nargin == 3
        func = @(idx, count) { 'ind_sum', idx, count, true, { dim, inds, s1 } };
    else if nargin == 6
        func = @(idx, count) { 'ind_sum', idx, count, true, { dim, inds, s1, dim2, inds2, s2 } };
    end
end

