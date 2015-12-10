% Solves:
% min_u (1/2) (u-f)^2 + \lambda * ||\nabla u||_{2,1}

%% read data, build linear operator
im = imread('data/24004.jpg');
im = imresize(im, 1);
[ny, nx] = size(im);

%f = double(im(:)) / 255.;

K = grad_forw_2d(nx, ny, 1);

linop = { linop_gradient2d(0, 0, nx, ny, 1) };

lmb = 1; 

%% setup prox operators
%prox_g = { prox_1d(0, nx * ny, 'square', 1, f, 1, 0, 0) };


%prox_hstar = { prox_norm2(0, nx * ny, 2, false, 'ind_leq0', ...
%                          1 / lmb, 1, 1, 0, 0) }; 

% anisotropic tv
%prox_hstar = { prox_1d(0, 2 * nx * ny, 'ind_box01', 1 / lmb, -0.5, 1, 0 ,0) }; 

% testing Moreau prox
prox_g = { prox_1d(0, nx * ny, 'square', 1, f, lmb, 0, 0) };
prox_hstar = { prox_moreau(prox_norm2(0, nx * ny, 2, false, 'abs', ...
                                      1, 0, 1, 0, 0)) }; 

global plot_primal;
global plot_dual;
global plot_iters;

plot_primal=[];
plot_dual=[];
plot_iters=[];

%% set options and run algorithm
opts = pdsolver_opts();
opts.adapt = 'converge';
opts.ads_gamma = lmb/2;
opts.verbose = true;
opts.bt_enabled = false;
opts.max_iters = 10000;
opts.cb_iters = 2;
opts.precond = 'alpha';
opts.precond_alpha = 1.;
opts.tol_primal = 0.1;
opts.tol_dual = 0.1;
opts.callback = @(it, x, y) disp(''); % ex_rof_callback(K, f, 1/lmb, it, x, y);

tic;
[x, y] = pdsolver(linop, prox_g, prox_hstar, opts);
toc;

%% show result
figure;
subplot(1,2,1);
imshow(reshape(f, ny, nx));
subplot(1,2,2);
imshow(reshape(x, ny, nx));

u_proj = x;
Kmat = spmat_gradient2d(nx,ny,1);
[m, n] = size(Kmat);
grad = Kmat * u_proj(:);
gradnorms = sqrt(grad(1:n).^2 + grad(n+1:end).^2);
en_prim = 0.5 * sum((u_proj(:)-f2).^2) + lmb * sum(gradnorms);
fprintf('primal_energy=%f\n', en_prim);
