%%
% load input image
im = imread('../../images/junction_gray.png');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
im = double(im) / 255;

% compute unary potentials
means = eye(3);
f = zeros(ny, nx, 3);
for i=1:3
    f(:, :, i) = sum((im - repmat(reshape(means(:, i), 1, 1, 3), ny, nx)).^2, 3);
end
f = f(:);

%%
% parameters
lmb = 1;
L = 3;

grad = spmat_gradient2d(nx,ny,L);
sum_op = kron(ones(1, L), speye(ny*nx));

u = prost.variable(nx*ny*L);
q = prost.variable(2*nx*ny*L);
s = prost.variable(nx*ny);

% I(u >= 0) + <u, f>
u.fun = prost.function.sum_1d('ind_geq0', 1, 0, 1, f, 0);

% |q_i| <= lmb

%% Zach et al., VMV '08
q.fun = prost.function.sum_norm2(... 
    2, false, 'ind_leq0', 1 / lmb, 1, 1, 0, 0);

%% Lellmann et al., ICCV '09
%q.fun = prost.function.sum_norm2(... 
%    2 * L, false, 'ind_leq0', 1 / lmb, 1, 1, 0, 0);

% <s, -1>
s.fun = prost.function.sum_1d('zero', 1, 0, 1, 1, 0);

% <grad u, q>
prost.set_dual_pair(u, q, prost.linop.sparse(grad));

% <sum_i u_i, s>
prost.set_dual_pair(u, s, prost.linop.sparse(sum_op));

prob = prost.min_max( {u}, {q, s} );

%%
% options and solve
backend = prost.backend.pdhg('stepsize', 'boyd', ...
                             'residual_iter', 10);

opts = prost.options('max_iters', 5000, ...
                     'tol_rel_primal', 1e-5, ...
                     'tol_abs_primal', 1e-5, ...
                     'tol_rel_dual', 1e-5, ...
                     'tol_abs_dual', 1e-5, ...
                     'num_cback_calls', 250, ...
                     'interm_cb', @(it, x, y) ...
                     example_multilabel_callback(it, x, y, ny, nx, ...
                                                 L, im));

tic;
prost.init();
prost.set_gpu(0);
result = prost.solve(prob, backend, opts);
prost.release();
toc;

u = reshape(u.val, [ny nx L]);
imshow([im, u]);
