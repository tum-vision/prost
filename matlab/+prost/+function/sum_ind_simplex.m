function [func] = sum_ind_simplex(dim, interleaved)
% SUM_IND_SIMPLEX  func = sum_ind_simplex(dim, interleaved)
%
%   Implements the seperable sum of dim-dimensional indicator
%   functions of the unit simplex, i.e., sum_i x_i = 1, x_i >= 0.
%   The interleaved keyword is handled as in sum_norm2. 
   
    func = @(idx, count) prost.prox.sum_ind_simplex(idx, count / dim, dim, ...
                                                    interleaved);
end
