% implementation of the classical ROF model
%
% min_u (1/2) (u-f)^2 + \lambda |\nabla u| 
%

im = imread('../../images/frame10.png');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.;
grad = spmat_gradient2d(nx,ny,nc);
lmb = 1;

% grad_gray = spmat_gradient2d(nx,ny,1);
% f_gray = double(rgb2gray(im)) / 255;
% Gpre=fspecial('gaussian', [13, 13], 0.5);
% f_gray = imfilter(f_gray, Gpre, 'same');
% f_gray = f_gray(:);
% gf = grad_gray * f_gray;
% gf_norm = reshape(gf, [ny, nx, 2]);
% gf_norm = sqrt(sum(gf_norm.^2, 3));
% normal = zeros(ny*nx*2,1);
% gf_norm_rep = repmat(gf_norm(:), 2, 1);
% normal(gf_norm_rep>eps) = gf(gf_norm_rep>eps) ./ gf_norm_rep(gf_norm_rep>eps);
% normal = reshape(normal, [ny nx 2]);

% normal_perp = zeros(ny,nx,2);
% normal_perp(:,:,1) = -normal(:, :, 2); 
% normal_perp(:,:,2) = normal(:, :, 1); 

% normal_outer = zeros(ny, nx, 2, 2);
% normal_perp_outer = zeros(ny, nx, 2, 2);

% for i=1:ny
%     i
%     for j=1:nx
%         normal_outer(i,j,:,:) = squeeze(normal(i, j, :)) * squeeze(normal(i, j, :))';
%         normal_perp_outer(i,j,:,:) = squeeze(normal_perp(i, j, :)) * squeeze(normal_perp(i, j, :))';
%     end
% end

% alpha = 5;
% beta = 0.5;
% D = repmat(reshape(exp(-alpha .* (gf_norm .^beta)), [ny nx]), [1, ...
%                     1, 2, 2]) .* normal_outer + normal_perp_outer;

% d11 = squeeze(D(:,:,1,1));
% d12 = squeeze(D(:,:,1,2));
% d21 = squeeze(D(:,:,2,1));
% d22 = squeeze(D(:,:,2,2));

% Dtensor = [spdiags(d11(:), 0, ny*nx, ny*nx), spdiags(d12(:), 0, ny*nx, ny*nx); 
%            spdiags(d21(:), 0, ny*nx, ny*nx), spdiags(d22(:), 0, ny*nx, ny*nx); ];

% grad=spmat_gradient2d_dtensor(nx,ny,nc,Dtensor);

prost.init();

%% create problem description
prob = prost.problem();
prob.linop = { prost.block.sparse(0, 0, grad) };
%prob.linop = { prost.block.gradient2d(0, 0, nx, ny, nc) };

prob.prox_g = { prost.prox.sum_1d(0, nx * ny * nc, 'square', 1, f, ...
                                   1, 0, 0) };

% Frobenius TV
%prob.prox_f = { prost.prox.sum_norm2(0, nx * ny, 2 * nc, false, 'abs', ...
%                                            1, 0, lmb, 0, 0) };

prob.prox_fstar = { prost.prox.sum_norm2(0, nx * ny, 2 * nc, false, 'ind_leq0', ...
                                            1 / lmb, 1, 1, 0, 0) };

% Nuclear norm TV, assumes nc = 3
% prob.prox_f = { prost.prox.sum_singular_nx2(0, nx * ny, 6, false, 'sum_1d:abs', ...
%                                             1, 0, 1, 0, 0) };

prob.scaling = 'alpha';

%% create backend
backend = prost.backend.pdhg('stepsize', 'goldstein');

%% specify solver options
rof_cb =  @(it, x, y) example_rof_energy_cb(...
    it, x, y, grad, f, lmb, ny, nx, nc);

opts = prost.options('tol_abs_dual', 1e-4, ...
                     'tol_abs_primal', 1e-4, ...
                     'tol_rel_dual', 1e-4, ...
                     'tol_rel_primal', 1e-4, ...
                     'max_iters', 10000);
opts.x0 = f;


%% solve problem
tic;
solution = prost.solve(prob, backend, opts);
toc;

prost.release();

%% show result
figure;
imshow(reshape(solution.x, ny, nx, nc));
