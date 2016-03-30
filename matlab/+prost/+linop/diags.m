function [linop] = diags(factors, offsets)
% DIAGS  linop = diags(factors, offsets)
%
%  Constructs a matrix with values factors on the diagonals
%  specified by the offsets matrix.
    
    linop = @(row, col, nrows, ncols) prost.block.diags(row, col, ...
                                                      nrows, ncols, ...
                                                      factors, offsets);
    
end
