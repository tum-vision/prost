function [K] = spmat_data_prec(nx, ny, L, left, right)
%SPMAT_DATA_PREC Summary of this function goes here
%   Detailed explanation goes here

    delta_t = (right - left) / (L-1);
    t = (left:delta_t:right)';
    
    N = nx*ny;

    B = spdiags([t(2:L)/delta_t, -t(1:L-1)/delta_t], [0, 1], L-1, L)';
    C = spdiags([ones(L-1, 1)/delta_t, -ones(L-1, 1)/delta_t], [0, 1], L-1, L)';
    
    K = sparse(0, 0);
    
    for l=1:L
        K = cat(1, K, cat(2, kron(speye(N), C(l, :)), kron(speye(N), B(l, :))));
    end

    K = cat(2, speye(N*L), K);
end

