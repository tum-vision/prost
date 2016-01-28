nx = 1024;
ny = 1024;
nc = 3;
N = nx * ny * nc;

% create problem description
prob = pdsolver_problem();
prob.linop = { linop_zero(0, 0, N, N) };
prob.prox_g = { prox_zero(0, N) };
prob.prox_fstar = { prox_zero(0, N) };
prob.scaling = 'identity';

% create backend
backend = pdsolver_backend_pdhg(...
    'residual_iter', 10);

% specify solver options
opts = pdsolver_options();
opts.max_iters = 10000;
opts.tol_abs_primal = -1;
opts.tol_abs_dual = -1;

% solve problem
tic;
solution = pdsolver(prob, backend, opts);
toc;

