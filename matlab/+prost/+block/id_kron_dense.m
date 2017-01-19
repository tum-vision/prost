function [func] = id_kron_dense(K, diaglength) 
% ID_KRON_DENSE  func = id_kron_dense(K, diaglength) 
%
% Linear operator that implements Kronecker product between an 
% identity matrix of size diaglength and K.
%
% Equivalent to prost.block.dense(full(kron(speye(diaglength), K))) but
% more efficient if K is small and diaglength is big.

    sz = { size(K, 1) * diaglength, size(K, 2) * diaglength };
    data = { K, diaglength };
    func = @(row, col, nrows, ncols) { { 'id_kron_dense', row, col, ...
                        data }, sz };
end
