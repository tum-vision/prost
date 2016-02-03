im = imread('24004.jpg');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
N = nx * ny * nc;
f = double(im(:)) / 255.;
grad = spmat_gradient2d(nx, ny, nc);
lmb = 1;

% create problem description

prob = pdsolver_problem();
prob.linop = { block_sparse(0, 0, grad) };
prob.prox_g = { prox_1d(0, N, 'square', 1, f, 1, 0, 0) };
prob.prox_fstar = { prox_norm2(0, N, 2, false, 'ind_leq0', ...
                               1 / lmb, 1, 1, 0, 0) };
prob.scaling = 'alpha';

% create backend
backend = pdsolver_backend_pdhg(...
    'residual_iter', 10);

% specify solver options
opts = pdsolver_options();
opts.max_iters = 1000;
opts.tol_abs_primal = -1;
opts.tol_abs_dual = -1;

% solve problem
solution = pdsolver(prob, backend, opts);



