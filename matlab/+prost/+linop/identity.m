function [linop] = identity(scaling)  
% IDENTITY linop = identity(scaling)
%  
%  Implements the identity operator, optionally scaled by a scalar.

    if nargin < 1
        scaling = 1;
    end

    linop = @(row, col, nrows, ncols) prost.block.identity(row, col, nrows, ...
                                               ncols, scaling);
    
end

