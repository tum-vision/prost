%%
% load input image
im = imread('../../images/lion.png');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.; % convert to [0, 1]

%%
% parameters
grad = spmat_gradient2d(nx,ny,nc);
lmb = 10;

%%
% problem
u = prost.variable(nx*ny*nc);
q = prost.variable(2*nx*ny*nc);

prob = prost.min_max_problem( {u}, {q} );
prob.add_function(u, prost.function.sum_1d('square', 1, f, lmb));

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
backend = prost.backend.pdhg('stepsize', 'alg2', ...
                             'residual_iter', 10, ...
                             'alg2_gamma', 0.05 * lmb);

%backend = prost.backend.admm('rho0', 15);

pd_gap_callback = @(it, x, y) example_rof_pdgap(it, x, y, grad, ...
                                                f, lmb, ny, nx, nc);

opts = prost.options('max_iters', 10000, ...
                     'interm_cb', pd_gap_callback, ...
                     'num_cback_calls', 250, ...
                     'verbose', true);

tic;
result = prost.solve(prob, backend, opts);
toc;

prost.release();

%%
% show result
imshow(reshape(u.val, [ny nx nc]));

%%
% Sanity check: compute residuals "by hand"
% g = prost.variable(2*nx*ny*nc);
% w = prost.variable(nx*ny*nc);
% prost.get_all_variables(result, {u}, {g}, {q}, {w});

% Sigma12 = (sum(abs(grad), 2)+eps).^(-1/2);
% Tau12 = sum(abs(grad), 1)'.^(-1/2);

% norm(Sigma12 .* (grad * u.val - g.val), 'fro')
% norm(Tau12 .* (grad' * q.val + w.val), 'fro')

