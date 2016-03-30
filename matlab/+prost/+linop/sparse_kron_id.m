function [linop] = sparse_kron_id(K, diaglength)    
% SPARSE_KRON_ID  linop = sparse_kron_id(K, diaglength)
%
%  Creates a linear operator consisting of the Kronecker product 
%  of the sparse matrix K with the identity matrix of size diaglength.
    
    linop = @(row, col, nrows, ncols) prost.block.sparse_kron_id(row, col, ...
                                                      nrows, ncols, ...
                                                      K, diaglength);
end
