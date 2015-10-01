function [K] = spmat_data_prec_2(nx, ny, L, left, right)
%SPMAT_DATA_PREC Summary of this function goes here
%   Detailed explanation goes here

    N = nx*ny;
    K = spmat_data_prec(nx, ny, L, left, right);
    Q = kron(speye(N), -ones(L-1, 1))';
    K = cat(1, K, cat(2, sparse(N, N*L + N*(L-1)), Q));
    K = cat(1, K, cat(2, sparse(2*N*(L-1), N*L), speye(2*N*(L-1))));
end

