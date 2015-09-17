% Solves lifted total variation
% not the most efficient formulation but only linearly many constraints
%
% Cost function:
% min_{u,w} max_{q,r,s} <u, f> + <u, -Div q> + <u, S^T s> - <s, 1>
% \sum_i <w_i, q_i - q_{i+1} - r_i> + I_>=0(u) - I_C(r)
%
% where C is the set ||r_i||_2 <= 1 and S is a sum operator.
%

% size of primal variables: 
% u: N*L (labeling indicator function) 
% w: 2*N*(L-1) (lagrange multiplier for q_i - q_{i+1})
% size of q: N*L*2 (lagrange for gradient)
% size of r: N*(L-1)*2 (r_i = q_i - q_{i+1})
% size of s: N (sum constraint)
% total size of dual variable: N*L*2 + N*(L-1)*2 + N

%% data term
im = imread('data/surprised-cat.jpg');
im = imresize(im, 0.25);
[ny, nx] = size(im);

L = 32;  % number of labels
t = linspace(0, 1, L); % label space, equidistant
N = nx * ny;

% ROF denoising data term
f2 = double(im(:)) / 255.;
f = zeros(nx * ny * L, 1);
for i=0:(L-1)
    f(i*nx*ny+1:(i+1)*nx*ny) = (1/2) * (t(i + 1) - f2).^2;
end

% regularization in front of total variation
% scale accordingly to label spacing
lmb = 0.5;
lmb_scaled = lmb * (t(2) - t(1)); % assumes equidistant labels

%% linear operator
% divergence
K1 = grad_forw_2d(nx, ny, L)'; 
% zeros
K2 = sparse(N*L, 2*N*(L-1)); 
% constraint for regularizer
K3 = spdiags([ones(2*N*L,1), -ones(2*N*L,1)],[0, 2*N],2*N*(L-1),2*N*L);
% constraint for regularizer
K4 = -speye(2*N*(L-1));
% sum constraint for primal variable
K5 = spdiags(ones(N, L),(0:(L-1))*N,N,N*L);
% zeros
K6 = sparse(N, 2 * N * (L - 1));

% combine submatrices
K = [K1' K3'; K2' K4'; K5 K6];
%K = K5;

%% prox operators
prox_g = { 
     prox_1d(0, N*L,'indicator_geq', 1, 0, 1, f, 0), 
     prox_zero(N*L, 2*N*(L-1)) };

prox_hc = { prox_zero(0, 2*N*L) };

for i=0:(L-2)
    prox_hc{i + 2, 1} = prox_norm2(2*N*L+i*2*N, N, 2, false, 'indicator_leq', ...
                                1 / lmb_scaled, 0, 1, 0, 0);
end

prox_hc{L + 1, 1} = prox_1d(2*N*L+2*N*(L-1),N,'zero',1,0,1,1,0);

%% solve problem
opts = pdsolver_opts();
opts.verbose = true;
opts.adapt = 'converge';
opts.bt_enabled = false;
opts.max_iters = 50;
opts.cb_iters = 10;
opts.precond = 'alpha';
opts.precond_alpha = 1.;
opts.tol_primal = 0.05;
opts.tol_dual = 0.05;
opts.callback = @(it, x, y) ex_lifted_tv_callback(it, x, y, f, nx, ny, L);
[uw, qrs] = pdsolver(K, prox_g, prox_hc, opts);


%% obtain "unlifted" result by computing the barycenter
u = reshape(uw(1:nx*ny*L), nx*ny, L) * t';

% u1 = reshape(uw(1:nx*ny), nx*ny, 1);
% u2 = reshape(uw(nx*ny+1:2*nx*ny), nx*ny, 1);
% u3 = reshape(uw(2*nx*ny+1:3*nx*ny), nx*ny, 1);
% u4 = reshape(uw(3*nx*ny+1:4*nx*ny), nx*ny, 1);

figure;
subplot(1,1,1);
imshow(reshape(u, ny, nx)); % solution


