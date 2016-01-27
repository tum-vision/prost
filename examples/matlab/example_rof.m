nx = 256;
ny = 256;
nc = 3;
N = nx * ny * nc;

% create problem description
prob = pdsolver_problem();
prob.linop = { linop_zero(0, 0, 2 * N, N) };
prob.prox_g = { prox_zero(0, N) };
prob.prox_fstar = { prox_zero(0, 2 * N) };

% create backend
backend = pdsolver_backend_pdhg();

% specify solver options
opts = pdsolver_options();

% solve problem
solution = pdsolver(prob, backend, opts);

