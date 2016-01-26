% problem description
prob = pdsolver_problem();
prob.prox_g = {};
prob.prox_fstar = {};
prob.linop = {};

% backend options
backend = pdsolver_backend_pdhg(... 
    'tau0', 1, ...
    'sigma0', 1);

% general solver options
opts = pdsolver_options();
opts.max_iters = 1000;

% solve problem
sol = pdsolver(prob, backend, opts);
