function [backend] = pdhg(varargin)

    dummy_cb = @(it, res_primal, res_dual, tau, sigma) [tau, sigma];
    
    p = inputParser;
    addOptional(p, 'tau0', 1);
    addOptional(p, 'sigma0', 1);
    addOptional(p, 'solve_dual', false);
    addOptional(p, 'residual_iter', 3);
    addOptional(p, 'scale_steps_operator', true);
    addOptional(p, 'alg2_gamma', 0);
    addOptional(p, 'arg_alpha0', 0.5);
    addOptional(p, 'arg_nu', 0.95);
    addOptional(p, 'arg_delta', 1.5);
    addOptional(p, 'arb_delta', 1.05);
    addOptional(p, 'arb_tau', 0.8);
    addOptional(p, 'stepsize', 'boyd');
    addOptional(p, 'stepsize_callback', dummy_cb);
   
    p.parse(varargin{:});
   
    backend = { 'pdhg', p.Results };

end
