function ex_rof_callback(K, f, lmb, it, x, y)
    % anisotropic TV
    en_prim = lmb * sum(abs(K*x)) + (1/2)*sum((x-f).^2); 
    en_dual = 0;
    en_gap = en_prim - en_dual;
    
    fprintf('it %5d: primal=%E, dual=%E, gap=%E\n', ...
            it, en_prim, en_dual, en_gap);
end
