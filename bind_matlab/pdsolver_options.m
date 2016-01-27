function [opts] = pdsolver_options(varargin)

    dummy_cb = @(it, x, y) disp('');

    p = inputParser;
    addOptional(p, 'tol_rel_primal', 0.01);
    addOptional(p, 'tol_rel_dual', 0.01);
    addOptional(p, 'tol_abs_primal', 0.01);
    addOptional(p, 'tol_abs_dual', 0.01);
    addOptional(p, 'max_iters', 1000);
    addOptional(p, 'num_cback_calls', 10);
    addOptional(p, 'verbose', true);
    addOptional(p, 'interm_cb', dummy_cb);

    p.parse(varargin{:});
    
    opts = p.Results;

end
