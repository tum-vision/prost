%%
% load input image
im = imread('../../images/dog.png');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.; % convert to [0, 1]

%%
% parameters
grad = spmat_gradient2d(nx,ny,nc);
lmb = 0.1;

%%
% problem
u = prost.variable(nx*ny*nc);
g = prost.variable(2*nx*ny*nc);

% Example on how to use sub-variables:
u1 = prost.sub_variable(u, 100);
u2 = prost.sub_variable(u, 500);
u3 = prost.sub_variable(u, nx*ny*nc-600);

prob = prost.min_problem( {u}, {g} );
prob.add_function(u1, prost.function.sum_1d('square', 1, f(1:100), lmb, 0, 0));
prob.add_function(u2, prost.function.sum_1d('square', 1, f(101:600), lmb, 0, 0));
prob.add_function(u3, prost.function.sum_1d('square', 1, f(601:end), lmb, 0, 0));
prob.add_function(g, prost.function.sum_norm2(2 * nc, false, 'abs', 1, 0, 1, 0, 0));
prob.add_constraint(u, g, prost.block.sparse(grad));

%%
% specify solver options
backend = prost.backend.pdhg('stepsize', 'boyd', ...
                             'residual_iter', 1, ...
                             'alg2_gamma', 0.05 * lmb, ...
                             'tau0', 1, ...
                             'sigma0', 1);

pd_gap_callback = @(it, x, y) example_rof_pdgap(it, x, y, grad, ...
                                                f, lmb, ny, nx, nc);

opts = prost.options('max_iters', 10000, ...
                     'interm_cb', pd_gap_callback, ...
                     'num_cback_calls', 250, ...
                     'verbose', true);

tic;
result = prost.solve(prob, backend, opts);
toc;

%%
% show result
imshow(reshape(u.val, [ny nx nc]));
