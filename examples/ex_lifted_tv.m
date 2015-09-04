% Solves lifted total variation
% not the most efficient formulation but only linearly many constraints
%
% Cost function:
% min_{u,w} max_{q,r,s} <u, f> + <u, -Div q> + <u, S^T s> - <s, 1>
% \sum_i <w_i, q_i - q_{i+1} - r> + I_>=0(u) + I_C(r)
%
% where C is the set ||r_i||_2 <= 1 and S is a sum operator.
%

%% data term
im = imread('data/surprised-cat.jpg');
im = imresize(im, 0.125);
[ny, nx] = size(im);

L = 2;  % number of labels
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
lmb = 0.6;
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
K = [K1' K3'; K2' K4'; K5 K6];

%% prox operators
prox_g = { 
    prox_1d(0, N*L,'indicator_leq', -1, 0, 1, 0, 0), 
    prox_zero(N*L, N*(L-1)) };

prox_hc = { 
    prox_zero(0, 2*N*L),
    prox_norm2(2*N*L, N*(L-1), 2, false, 'indicator_leq', 1 / lmb_scaled, 0, 1, 0, 0),
    prox_1d(2*N*L+2*N*(L-1),N,'zero',0,0,0,1,0) };

%% solve problem
opts = pdsolver_opts();
opts.pdhg_type = 'adapt';
opts.max_iters = 100;
[uw, qrs] = pdsolver(K, prox_g, prox_hc, opts);

%% obtain "unlifted" result by computing the barycenter
u = reshape(uw(1:nx*ny*L), nx*ny, L) * t';

figure;
subplot(1,2,1);
imshow(reshape(f2, ny, nx)); % input
subplot(1,2,2);
imshow(reshape(u, ny, nx)); % solution
