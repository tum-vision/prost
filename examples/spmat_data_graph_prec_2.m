function [K] = spmat_data_graph_prec_2(nx, ny, L, t_min, t_max)
%SPMAT_DATA_PREC Summary of this function goes here
%   Detailed explanation goes here
    delta_t = (t_max - t_min) / (L-1);
    t = (t_min:delta_t:t_max)';
    
    N = nx*ny;
    Q = kron(speye(N), -ones(L-1, 1));
    Q = spdiags(repmat(t(2:end) - t(1:end-1), [N, 1]), 0, N*(L-1)) * Q;
    W = sparse(0,0);
    for i=1:L-1
        W = cat(1, W, [ones(1,i-1), zeros(1,L-1-i+1)]);
    end
    D = spdiags(-t(1:end-1) ./ (t(2:end) - t(1:end-1)), 0, L-1, L-1);
    V = kron(speye(N), W + D);
    V = spdiags(repmat(t(2:end) - t(1:end-1), [N, 1]), 0, N*(L-1)) * V;
    Z = speye(N*(L-1));
    K = cat(2, V, Z);
    K = cat(2, K, Q);
    K=K';
end

