% implementation of the classical ROF model
%
% min_u (1/2) (u-f)^2 + \lambda |\nabla u| 
%

im = imread('../../images/dog.png');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.;
grad = spmat_gradient2d(nx,ny,nc);
lmb = 1;

prost.init();

%% create problem description
prob = prost.problem();
prob.linop = { prost.block.sparse(0, 0, grad) };
prob.prox_g = { prost.prox.sum_1d(0, nx * ny * nc, 'square', 1, f, 1, 0, 0) };
prob.prox_fstar = { prost.prox.sum_norm2(0, nx * ny, 2 * nc, false, 'ind_leq0', ...
                               1 / lmb, 1, 1, 0, 0) };
prob.scaling = 'alpha';

%% create backend
backend = prost.backend.pdhg(...
    'residual_iter', 10, ...
    'stepsize', 'boyd', ...
    'scale_steps_operator', true);

%% specify solver options
opts = prost.options();
opts.max_iters = 500;

%% solve problem
tic;
solution = prost.solve(prob, backend, opts);
toc;

prost.release();

%% show result
figure;
imshow(reshape(solution.x, ny, nx, nc));
