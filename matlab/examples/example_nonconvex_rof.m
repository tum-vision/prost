%%
% load input image
%im = imread('../../images/cow.png');
im = imread('../../../nonconvexPDHG/talk-SIAM/albuquerque1.jpg');
im = imresize(im, 1);
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
%prob.add_function(q, prost.function.sum_norm2(2 * nc, false, ...
%                                              'ind_leq0', 1, 1, 1));

% Mumford-Shah (Truncated Quadratic)
lambda = 0.01;
alpha = 25;

prob.add_function(q, ...
                  prost.function.conjugate(...
                      prost.function.sum_norm2(... 
                          2 * nc, false, 'truncquad', ...
                          1,0,1,0,0, alpha, lambda)));

% \ell^alpha
% alpha = 2.5;

% prob.add_function(q, ...
%                   prost.function.conjugate(...
%                       prost.function.sum_norm2(... 
%                           2 * nc, false, 'lq', ...
%                           1,0,1,0,0, alpha)));

prob.add_dual_pair(u, q, prost.block.sparse(grad));

%%
% specify solver options
%backend = prost.backend.pdhg('stepsize', 'alg2', ...
%                              'residual_iter', -1, ...
%                              'alg2_gamma', 1);

c = 1;
backend = prost.backend.pdhg('stepsize', 'alg1', ...
                              'residual_iter', -1, ...
                              'alg2_gamma', 1, ...
                             'tau0', c / 2, ...
                             'sigma0', 1 / (4 * c));

pd_gap_callback = @(it, x, y) example_rof_pdgap(it, x, y, grad, ...
                                                f, lmb, ny, nx, nc);

opts = prost.options('max_iters', 1000, ...
                     'num_cback_calls', 100, ...
                     'verbose', true, ...
                     );

tic;
result = prost.solve(prob, backend, opts);
toc;

%%
% show result
imshow([double(im)/255 reshape([u.val], [ny nx nc])]);
