% implementation of the classical ROF model
%
% min_u (1/2) (u-f)^2 + \lambda |\nabla u| 
%

im = imread('../../images/dog.png');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.;
grad = spmat_gradient2d(nx,ny,nc);
lmb = 15;

prost.init();

%% create problem description
prob = prost.problem();
%prob.linop = { prost.block.sparse(0, 0, grad) };
prob.linop = { prost.block.gradient2d(0, 0, nx, ny, nc) };

prob.prox_g = { prost.prox.sum_1d(0, nx * ny * nc, 'square', 1, f, ...
                                   1, 0, 0) };

% Frobenius TV
%prob.prox_f = { prost.prox.sum_norm2(0, nx * ny, 2 * nc, false, 'abs', ...
%                                            1, 0, lmb, 0, 0) };

prob.prox_fstar = { prost.prox.sum_norm2(0, nx * ny, 2 * nc, false, 'ind_leq0', ...
                                            1 / lmb, 1, 1, 0, 0) };

% Nuclear norm TV, assumes nc = 3
% prob.prox_f = { prost.prox.sum_singular_nx2(0, nx * ny, 6, false, 'sum_1d:abs', ...
%                                             1, 0, 1, 0, 0) };

prob.scaling = 'alpha';

%% create backend
backend = prost.backend.pdhg('stepsize', 'goldstein');

%% specify solver options
rof_cb =  @(it, x, y) example_rof_energy_cb(...
    it, x, y, grad, f, lmb, ny, nx, nc);

opts = prost.options('tol_abs_dual', 1e-4, ...
                     'tol_abs_primal', 1e-4, ...
                     'tol_rel_dual', 1e-4, ...
                     'tol_rel_primal', 1e-4, ...
                     'max_iters', 10000);
opts.x0 = f;


%% solve problem
tic;
solution = prost.solve(prob, backend, opts);
toc;

prost.release();

%% show result
figure;
imshow(reshape(solution.x, ny, nx, nc));
