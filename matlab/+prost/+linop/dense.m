function [linop] = dense(K)
% DENSE  linop = dense(K)
%
%   Constructs linear operator from dense matrix K.
    
    linop = @(row, col, nrows, ncols) prost.block.dense(row, col, ...
                                                      nrows, ncols, ...
                                                      K);
    
end