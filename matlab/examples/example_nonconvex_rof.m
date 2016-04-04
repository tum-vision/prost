%%
% load input image
im = imread('../../images/cow.png');
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.; % convert to [0, 1]

%%
% parameters
grad = spmat_gradient2d(nx,ny,nc);
lmb = 3;

%%
% problem
u = prost.variable(nx*ny*nc);
q = prost.variable(2*nx*ny*nc);

u.fun = prost.function.sum_1d('square', 1, f, lmb, 0, 0);
q.fun = prost.function.conjugate(prost.function.sum_norm2(... 
    2 * nc, false, 'l0', 1, 0, 1, 0, 0));

prost.set_dual_pair(u, q, prost.linop.sparse(grad));

prob = prost.min_max( {u}, {q} );

%%
% specify solver options
backend = prost.backend.pdhg('stepsize', 'alg2', ...
                             'residual_iter', -1, ...
                             'alg2_gamma', 0.1 * lmb);

opts = prost.options('max_iters', 2000, ...
                     'num_cback_calls', 10, ...
                     'verbose', true);

tic;
result = prost.solve(prob, backend, opts);
toc;

%%
% show result
imshow([double(im)/255 reshape([u.val], [ny nx nc])]);
