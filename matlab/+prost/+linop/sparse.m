function [linop] = sparse(K)    
% SPARSE  linop = sparse(K)
%
%  Creates a linear operator from the sparse matrix K.
    
    linop = @(row, col, nrows, ncols) prost.block.sparse(row, col, ...
                                                      nrows, ncols, ...
                                                      K);
end
