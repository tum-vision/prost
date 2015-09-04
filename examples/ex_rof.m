% Solves:
% min_u (1/2) (u-f)^2 + \lambda * ||\nabla u||_{2,1}

%% read data, build linear operator
im = imread('data/surprised-cat.jpg');
im = imresize(im, 0.25);
[ny, nx] = size(im);

f = double(im(:)) / 255.;
K = grad_forw_2d(nx, ny, 1);
lmb = 0.5; 

%% setup prox operators
prox_g = { prox_1d(0, nx * ny, 'square', 1, f, 1, 0, 0) };
prox_hstar = { prox_norm2(0, nx * ny, 2, false, 'indicator_leq', ...
                          1 / lmb, 0, 1, 0, 0) }; 

% anisotropic tv
% prox_hstar = { prox_1d(0, 2 * nx * ny, 'indicator_absleq', 1 / lmb, 0, 1, 0 ,0) }; 

%% set options and run algorithm
opts = pdsolver_opts();
opts.pdhg_type = 'backtrack';
opts.max_iters = 5000;
opts.precond = 'alpha';
opts.precond_alpha = 1.;
opts.callback = @(it, x, y) ex_rof_callback(K, f, lmb, it, x, y);
[x, y] = pdsolver(K, prox_g, prox_hstar, opts);

%% show result
figure;
subplot(1,2,1);
imshow(reshape(f, ny, nx));
subplot(1,2,2);
imshow(reshape(x, ny, nx));
