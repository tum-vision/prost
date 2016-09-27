function [backend] = admm(varargin)

    p = inputParser;
    addOptional(p, 'rho0', 1);
    addOptional(p, 'residual_iter', 1);
    addOptional(p, 'arb_delta', 1.05);
    addOptional(p, 'arb_tau', 0.8);
    addOptional(p, 'arb_gamma', 1.01);
    addOptional(p, 'alpha', 1.7);
    addOptional(p, 'cg_max_iter', 10);
    addOptional(p, 'cg_tol_pow', 1.3);
    addOptional(p, 'cg_tol_min', 1e-5);
    addOptional(p, 'cg_tol_max', 1e-8);
   
    p.parse(varargin{:});
   
    backend = { 'admm', p.Results };

end
