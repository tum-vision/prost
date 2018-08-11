function [func] = sum_ind_sum(dim, interleaved)
% SUM_IND_SUM  func = sum_ind_sum(dim, interleaved)
%
%   Implements the seperable sum of dim-dimensional indicator
%   functions of the sum-to-one constraint, i.e., sum_i x_i = 1.
%   The interleaved keyword is handled as in sum_norm2. 
   
    func = @(idx, count) { 'elem_operation:ind_sum', idx, count, false, { count / dim, dim, interleaved } };
end
