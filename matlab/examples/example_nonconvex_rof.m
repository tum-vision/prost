%%
% load input image
im = imread('../../images/cow.png');
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.; % convert to [0, 1]

%%
% parameters
grad = spmat_gradient2d(nx,ny,nc);


%%
% problem
u = prost.variable(nx*ny*nc);
q = prost.variable(2*nx*ny*nc);

prob = prost.min_max_problem( {u}, {q} );

prob.add_function(u, prost.function.sum_1d('square', 1, f, 1));

% Mumford-Shah (Truncated Quadratic)
% lambda = 0.25;
% alpha = 25;

% prob.add_function(q, ...
%                   prost.function.conjugate(...
%                       prost.function.sum_norm2(... 
%                           2 * nc, false, 'truncquad', ...
%                           1,0,1,0,0, alpha, lambda)));

% \ell^alpha
alpha = 2.5;

prob.add_function(q, ...
                  prost.function.conjugate(...
                      prost.function.sum_norm2(... 
                          2 * nc, false, 'lq', ...
                          1,0,1,0,0, alpha)));

prob.add_dual_pair(u, q, prost.block.sparse(grad));

%%
% specify solver options
backend = prost.backend.pdhg('stepsize', 'alg2', ...
                             'residual_iter', -1, ...
                             'alg2_gamma', 0.1 * lmb);

opts = prost.options('max_iters', 5000, ...
                     'num_cback_calls', 0, ...
                     'verbose', true);

tic;
result = prost.solve(prob, backend, opts);
toc;

%%
% show result
imshow([double(im)/255 reshape([u.val], [ny nx nc])]);
