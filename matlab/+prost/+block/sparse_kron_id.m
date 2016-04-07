function [func] = sparse_kron_id(K, diaglength) 
% SPARSE_KRON_ID  func = sparse_kron_id(K, diaglength) 
%
% Linear operator that implements Kronecker product between K and
% an identity matrix of size diaglength. 
%
% Equivalent to prost.block.sparse(kron(K, speye(diaglength))) but
% more efficient if K is small and diaglength is big.
        

    sz = { size(K, 1) * diaglength, size(K, 2) * diaglength };
    data = { K, diaglength };
    func = @(row, col, nrows, ncols) { { 'sparse_kron_id', row, col, ...
                        data }, sz };
end
