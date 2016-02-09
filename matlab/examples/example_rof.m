im = imread('../../images/dog.png');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.;
grad = spmat_gradient2d(nx,ny,nc);
lmb = 1;

% create problem description
prob = prost.pdsolver_problem();
prob.linop = { prost.block_sparse(0, 0, grad) };
prob.prox_g = { prost.prox_1d(0, nx * ny * nc, 'square', 1, f, 1, 0, 0) };
prob.prox_fstar = { prost.prox_norm2(0, nx * ny, 2 * nc, false, 'ind_leq0', ...
                               1 / lmb, 1, 1, 0, 0) };
prob.scaling = 'alpha';

% create backend
backend = prost.pdsolver_backend_pdhg(...
    'residual_iter', 10, ...
    'stepsize', 'boyd', ...
    'scale_steps_operator', true);

% specify solver options
opts = prost.pdsolver_options();
opts.max_iters = 10000;

% solve problem
tic;
solution = prost.prost_(prob, backend, opts);
toc;

figure;
imshow(reshape(solution.x, ny, nx, nc));
