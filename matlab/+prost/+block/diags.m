function [func] = diags(nrows, ncols, factors, offsets)  
% DIAGS func = diags(nrows, ncols, factors, offsets)
%
% Implements a linear operator with diagonals specified by the
% factors and offsets array. 
%
% Example: a tridiagonal matrix with values -2 1 4
% 
%    1  4  0  0  0
%   -2  1  4  0  0
%    0 -2  1  4  0
%    0  0 -2  1  4
%    0  0  0 -2  1
%
%  is given by factors = [-2 1 4] and offsets = [-1 0 1].
        
    sz = { nrows, ncols };
    data = { nrows, ncols, factors, offsets };
    
    func = @(row, col, nrows, ncols) { { 'diags', row, col, data }, sz };
end
