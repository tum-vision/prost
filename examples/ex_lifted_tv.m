% Solves lifted total variation
% not the most efficient formulation but only linearly many constraints
%
% Cost function:
% min_{u,w} max_{q,r,s} <u, f> + <u, -Div q> + <u, S^T s> - <s, 1>
% \sum_i <w_i, q_i - q_{i+1} - r> + I_>=0(u) - I_C(r)
%
% where C is the set ||r_i||_2 <= 1 and S is a sum operator.
%

% size of primal variables: 
% u: N*L 
% w: 2 * N * (L - 1)
% size of q: N*L*2
% size of r: N*(L-1)*2
% size of s: N
% size of dual variable: N*L*2 + N*(L-1)*2 + N


%% data term
im = imread('data/surprised-cat.jpg');
im = imresize(im, 0.1);
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
lmb = 1;
lmb_scaled = lmb * (t(2) - t(1)); % assumes equidistant labels

%% linear operator
% divergence
K1 = grad_forw_2d(nx, ny, L)'; 
% zeros
K2 = sparse(N*L, 2*N*(L-1)); 
% constraint
K3 = spdiags([ones(2*N*L,1), -ones(2*N*L,1)],[0, 2*N],2*N*(L-1),2*N*L);
% constraint
K4 = -speye(2*N*(L-1));
% sum constraint
K5 = spdiags(ones(N, L),(0:(L-1))*N,N,N*L);
% zeros
K6 = sparse(N, 2 * N * (L - 1));

% combine submatrices
%K = [K1' K3'; K2' K4'; K5 K6];
K = K5;

%% prox operators
% prox_g = { 
%     prox_1d(0, N*L,'indicator_geq', 1, 0, 1, f, 0), 
%     prox_zero(N*L, 2*N*(L-1)) };

prox_g = {
    prox_1d(0, N*L, 'indicator_geq', 1, 0, 1, f, 0) };

% prox_hc = { 
%     prox_zero(0, 2*N*L),
%     prox_norm2(2*N*L, N*(L-1), 2, false, 'indicator_leq', 1 / lmb_scaled, 0, 1, 0, 0),
%     prox_1d(2*N*L+2*N*(L-1),N,'zero',1,0,1,1,0) 
% %    prox_zero(2*N*L+2*N*(L-1), N)
%           };

prox_hc = {
    prox_1d(0, N, 'zero', 1,0,1,1,0) };

%% build ground truth
[val, ind] = min(reshape(f, nx * ny, L)');
result_gt = t(ind);
u_opt = zeros(nx * ny, L);
for i=1:nx*ny
    u_opt(i, ind(i)) = 1;
end

en_opt = u_opt(:)' * f;

%% solve problem
opts = pdsolver_opts();
opts.adapt = 'converge';
opts.bt_enabled = true;
opts.max_iters = 10000;
opts.cb_iters = 10;
opts.precond = 'alpha';
opts.precond_alpha = 1.;
opts.tol_primal = 0.05;
opts.tol_dual = 0.05;
opts.callback = @(it, x, y) ex_lifted_tv_callback(it, x, y, f, nx, ...
                                                  ny, L, en_opt);
%[uw, qrs] = pdsolver(K, prox_g, prox_hc, opts);
tic;
[u, s] = pdsolver(K, prox_g, prox_hc, opts);
toc;

%% obtain "unlifted" result by computing the barycenter
%u = reshape(uw(1:nx*ny*L), nx*ny, L) * t';

%u2 = reshape(u, nx*ny, L);
%u2(1:100,:)

[val, ind] = max(reshape(u, nx * ny, L)');
result = t(ind);


figure;
subplot(1,3,1);
imshow(reshape(f2, ny, nx)); % input
subplot(1,3,2);
imshow(reshape(result, ny, nx)); % solution
subplot(1,3,3);
imshow(reshape(result_gt, ny, nx)); % solution
