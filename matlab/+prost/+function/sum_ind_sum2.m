function [func] = sum_ind_sum2(dim, inds, dim2, inds2)
% SUM_IND_SUM2  func = sum_ind_sum2(dim, interleaved)
%
%   Implements the seperable sum of dim-dimensional indicator
%   functions of the sum-to-one constraint, i.e., sum_i x_i = 1
%   according to index arrays.
    
    if nargin == 2
        func = @(idx, count) { 'ind_sum', idx, count, true, { dim, inds } };
    else if nargin == 4
        func = @(idx, count) { 'ind_sum', idx, count, true, { dim, inds, dim2, inds2 } };
    end
end

