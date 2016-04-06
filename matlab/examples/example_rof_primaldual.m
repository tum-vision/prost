%%
% load input image
im = imread('../../images/dog.png');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.; % convert to [0, 1]

%%
% parameters
grad = spmat_gradient2d(nx,ny,nc);
lmb = 1;

%%
% problem
u = prost.variable(nx*ny*nc);
q = prost.variable(2*nx*ny*nc);

u.fun = prost.function.sum_1d('square', 1, f, lmb, 0, 0);
q.fun = prost.function.sum_norm2(... 
    2 * nc, false, 'ind_leq0', 1, 1, 1, 0, 0);

%prost.set_dual_pair(u, q, prost.linop.sparse(grad));
prost.set_dual_pair(u, q, prost.linop.gradient2d(nx,ny,nc));

prob = prost.min_max( {u}, {q} );
%prob.data.scaling = 'identity';

%%
% specify solver options
%backend = prost.backend.pdhg('stepsize', 'alg2', ...
%                             'residual_iter', -1, ...
%                             'alg2_gamma', 0.05 * lmb);

backend = prost.backend.admm('rho0', 1);

pd_gap_callback = @(it, x, y) example_rof_pdgap(it, x, y, grad, ...
                                                f, lmb, ny, nx, nc);

opts = prost.options('max_iters', 1000, ...
                     'interm_cb', pd_gap_callback, ...
                     'num_cback_calls', 25, ...
                     'verbose', true);

tic;
result = prost.solve(prob, backend, opts);
toc;

% g = prost.variable(2*nx*ny*nc);
% w = prost.variable(nx*ny*nc);
% prost.get_all_variables(result, {u}, {g}, {q}, {w});

% Sigma12 = (sum(abs(grad), 2)+1e-6).^(-1/2);
% Tau12 = sum(abs(grad), 1)'.^(-1/2);

% norm(Sigma12 .* (grad * u.val - g.val), 'fro')
% norm(Tau12 .* (grad' * q.val + w.val), 'fro')

%%
% show result
imshow(reshape(u.val, [ny nx nc]));
