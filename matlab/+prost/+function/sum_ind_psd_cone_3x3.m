function [func] = sum_ind_psd_cone_3x3(interleaved)
% SUM_IND_SIMPLEX  func = sum_ind_psd_cone_3x3(interleaved)
%
%   Implements the seperable sum of dim-dimensional indicator
%   functions of the unit simplex, i.e., sum_i x_i = 1, x_i >= 0.
%   The interleaved keyword is handled as in sum_norm2. 

    dim = 9;
    func = @(idx, count) { 'elem_operation:ind_psd_cone_3x3', idx, count, false, { count / dim, dim, interleaved } };
end
