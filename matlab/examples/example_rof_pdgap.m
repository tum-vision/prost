function [is_converged] = example_rof_pdgap(it, x, y, K, f, lmb, ny, nx, nc)

    % compute primal and dual energy for isotropic Frobenius TV ROF
    [m, n] = size(K);
    grad = reshape(K*x, [ny*nx, 2*nc]);
    gradnorms = sqrt(sum(grad.^2, 2)); % 2,1-norm
    en_prim = 0.5 * lmb * sum((x-f).^2) + sum(gradnorms);
    
    div = K' * y;
    en_dual = f' * div - (1 / (2 * lmb)) * sum(div.^2);
    
    en_gap = (en_prim - en_dual) / (nx * ny);
    fprintf('primal_dual_gap=%.2e.\n', en_gap);
    
    is_converged = en_gap < 1e-5;
    
end
