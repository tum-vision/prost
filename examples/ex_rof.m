%% read data, build linear operator
im = imread('data/surprised-cat.jpg');
im = imresize(im, 0.25);
[ny, nx] = size(im);

f = double(im(:)) / 255.;
K = grad_forw_2d(nx, ny, 1);
lmb = 0.5; % regularization constant in front of TV

%% setup prox operators
prox_g = { prox_1d(0, nx * ny, 'square', 1, f, 1, 0, 0) };

% anisotropic tv
%prox_hstar = { prox_1d(0, 2 * nx * ny, 'indicator_absleq', 1 / lmb, 0, 1, 0 ,0) }; 
% isotropic tv
prox_hstar = { prox_norm2(0, nx * ny, 2, false, 'indicator_leq', 1 / lmb, 0, 1, 0, 0) }; 

%% run algorithm
% TODO: handle this more nicely, put solver_callback into opts
opts = struct('backend', 'pdhg', 'max_iters', 5000, 'cb_iters', 10, ...
    'tolerance', 1e-6, 'pdhg_type', 'alg1', 'gamma', 0, ...
    'alpha0', 0, 'nu', 0, 'delta', 0, 's', 0, 'precond', 'off', ...
    'precond_alpha', 1, 'verbose', true);

solver_callback = @(it, x, y) ex_rof_callback(K, f, lmb, it, x, y);
[x, y] = pdsolver(K, prox_g, prox_hstar, opts, solver_callback);

%% show result
figure;
subplot(1,2,1);
imshow(reshape(f, ny, nx));
subplot(1,2,2);
imshow(reshape(x, ny, nx));
