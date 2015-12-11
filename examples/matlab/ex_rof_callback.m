function ex_rof_callback(K, f, lmb, it, x, y)
    % isotropic TV
    [m, n] = size(K);
    grad = K * x;
    gradnorms = sqrt(grad(1:n).^2 + grad(n+1:end).^2);
    en_prim = 0.5 * sum((x-f).^2) + lmb * sum(gradnorms);
        
    div = K' * y;
    en_dual = f' * div - 0.5 * sum(div.^2);
        
    en_gap = en_prim - en_dual;
    
    fprintf('it %5d: en_prim=%E, en_dual=%E, en_gap=%E\n', ...
            it, en_prim, en_dual, en_gap);
    
    global plot_primal;
    global plot_dual;
    global plot_iters;
    plot_primal = cat(1, plot_primal, en_prim);
    plot_dual = cat(1, plot_dual, en_dual);
    plot_iters = cat(1, plot_iters, it);
end
