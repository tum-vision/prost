function example_rof_energy_cb(it, x, y, K, f, lmb, ny, nx, nc)

    %% compute energy for isotropic TV
    [m, n] = size(K);
    grad = reshape(K*x, [ny*nx, 2*nc]);
    gradnorms = sqrt(sum(grad.^2, 2)); % 2,1-norm
    en_prim = 0.5 * sum((x-f).^2) + lmb * sum(gradnorms);
    
    div = K' * y;
    en_dual = f' * div - 0.5 * sum(div.^2);
    
    en_gap = (en_prim - en_dual) / (nx * ny);
    fprintf('primal_dual_gap=%.2e.\n', en_gap);
    
end
