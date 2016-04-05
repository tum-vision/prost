%%
% load input image
im = imread('../../images/dog.png');
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.; % convert to [0, 1]

%%
% parameters
grad = spmat_gradient2d(nx,ny,nc);
lmb = 0.3;

%%
% problem
u = prost.variable(nx*ny*nc);
q = prost.variable(2*nx*ny*nc);

u.fun = prost.function.sum_1d('square', 1, f, lmb, 0, 0);
q.fun = prost.function.sum_norm2(... 
    2 * nc, false, 'ind_leq0', 1, 1, 1, 0, 0);

prost.set_dual_pair(u, q, prost.linop.sparse(grad));

prob = prost.min_max( {u}, {q} );

%%
% specify solver options
%backend = prost.backend.pdhg('stepsize', 'alg2', ...
%                             'residual_iter', -1, ...
%                             'alg2_gamma', 0.05 * lmb);

backend = prost.backend.admm();

pd_gap_callback = @(it, x, y) example_rof_pdgap(it, x, y, grad, ...
                                                f, lmb, ny, nx, nc);

opts = prost.options('max_iters', 100, ...
                     'interm_cb', pd_gap_callback, ...
                     'num_cback_calls', 3, ...
                     'verbose', true, ...
                     'tol_rel_primal', 0, ...
                     'tol_abs_primal', 0, ...
                     'tol_rel_dual', 0, ...
                     'tol_abs_dual', 0);

tic;
result = prost.solve(prob, backend, opts);
toc;

%%
% show result
imshow(reshape(u.val, [ny nx nc]));
