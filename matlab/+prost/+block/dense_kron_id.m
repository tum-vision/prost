function [func] = dense_kron_id(K, diaglength) 
% DENSE_KRON_ID  func = dense_kron_id(K, diaglength) 
%
% Linear operator that implements Kronecker product between an 
% identity matrix of size diaglength and K.
%
% Equivalent to prost.block.dense(full(kron(K, speye(diaglength)))) but
% more efficient if K is small and diaglength is big.

    sz = { size(K, 1) * diaglength, size(K, 2) * diaglength };
    data = { K, diaglength };
    func = @(row, col, nrows, ncols) { { 'dense_kron_id', row, col, ...
                        data }, sz };
end
