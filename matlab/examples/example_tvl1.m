rng(42);

%%
% load input image
im = imread('../../images/fisch.jpg');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.; % convert to [0, 1]

% add salt&pepper noise
pix = randperm(nx*ny*nc);
num_bad_pix = round(0.25*nx*ny*nc);
f(pix(1:num_bad_pix)) = 1;
f(pix(num_bad_pix+1:2*num_bad_pix)) = 0;

%%
% parameters
grad = spmat_gradient2d(nx,ny,nc);
lmb = 1;

%%
% problem
u = prost.variable(nx*ny*nc);
q = prost.variable(2*nx*ny*nc);

prob = prost.min_max_problem( {u}, {q} );
prob.add_function(u, prost.function.sum_1d('abs', 1, f, lmb));

% Equivalently:
% prob.add_function(u, prost.function.transform(...
%     prost.function.sum_1d('square'), 1, f, lmb));

prob.add_function(q, prost.function.sum_norm2(2 * nc, false, ...
                                              'ind_leq0', 1, 1, 1));

prob.add_dual_pair(u, q, prost.block.gradient2d(nx,ny,nc));
% Equivalently:
% prob.add_dual_pair(u, q, prost.block.sparse(grad));

%%
% specify solver options
backend = prost.backend.pdhg('stepsize', 'boyd', ...
                             'residual_iter', 10);

%backend = prost.backend.admm('rho0', 15);

opts = prost.options('max_iters', 50000, ...
                     'num_cback_calls', 250, ...
                     'verbose', true, ...
                     'tol_rel_primal', 1e-7, ...
                     'tol_rel_dual', 1e-7, ...
                     'tol_abs_dual', 1e-7, ...
                     'tol_abs_primal', 1e-7);

tic;
result = prost.solve(prob, backend, opts);
toc;

%%
% show result
imshow(reshape(u.val, [ny nx nc]));

Du = reshape(grad*u.val, [ny*nx, nc, 2]);
norm_Du = sqrt(sum(sum(Du.^2,2), 3));
energy_pd = lmb * sum(abs(u.val-f)) + sum(norm_Du)

