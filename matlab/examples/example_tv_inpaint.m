rng(42);

%%
% load input image
im = imresize(imread('../../images/lion.png'), 1);
mask = imresize(imread('../../images/maske2.png'), 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.; 
m = 1 - (mask > 0);
m = double(m(:));

%%
% parameters
grad = spmat_gradient2d(nx,ny,nc);
lmb = 7;

%%
% problem
u = prost.variable(nx*ny*nc);
q = prost.variable(2*nx*ny*nc);

prob = prost.min_max_problem( {u}, {q} );
prob.add_function(u, prost.function.sum_1d('square', m, f, lmb));

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
energy_pd = (lmb/2) * sum((m .* (u.val-f)).^2) + sum(norm_Du)

