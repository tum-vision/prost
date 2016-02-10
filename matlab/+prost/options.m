function [opts] = options(varargin)

    dummy_cb = @(it, x, y) disp('');

    p = inputParser;
    addOptional(p, 'tol_rel_primal', 1e-2);
    addOptional(p, 'tol_rel_dual', 1e-2);
    addOptional(p, 'tol_abs_primal', 1e-4);
    addOptional(p, 'tol_abs_dual', 1e-4);
    addOptional(p, 'max_iters', 1000);
    addOptional(p, 'num_cback_calls', 10);
    addOptional(p, 'verbose', true);
    addOptional(p, 'interm_cb', dummy_cb);

    p.parse(varargin{:});
    
    opts = p.Results;

end
