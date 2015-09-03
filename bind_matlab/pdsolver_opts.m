function [opts] = pdsolver_opts()

    dummy_cb = @(it, x, y) disp('');
    
    opts = struct('backend', 'pdhg', 'max_iters', 5000, 'cb_iters', 10, ...
                  'tolerance', 1e-6, 'pdhg_type', 'alg1', 'gamma', 0, ...
                  'alpha0', 0.5, 'nu', 0.95, 'delta', 1.5, 's', 20, 'precond', 'off', ...
                  'precond_alpha', 1, 'verbose', true, 'callback', dummy_cb);

end
