% test-bench for the different linear operators

%%
% Sparse Matrix test

% build big block matrix out of many sparse matrices
K = [];
linop = {};
idx = 1;
row = 0;
nrows = 321;
ncols = 117;

By = 12;
Bx = 14;

for i=1:By
    
    K_row = [];
    col = 0;
    for j=1:Bx
        K_mat = sprand(nrows,ncols,0.01);
        K_row = cat(2, K_row, K_mat);
        linop{idx, 1} = linop_sparse(row, col, K_mat);
        idx = idx + 1;
        col = col + ncols;
    end
    
    row = row + nrows;
    K = cat(1, K, K_row);
end

inp = rand(ncols*Bx, 1);
inp2 = rand(nrows*By, 1);

[x,~,~] = pdsolver_eval_linop(linop, inp, false);
[y,rowsum,colsum] = pdsolver_eval_linop(linop, inp2, true);

x_ml = K*inp;
y_ml = K'*inp2;

rowsum_ml = sum(abs(K), 2);
colsum_ml = sum(abs(K), 1)';

fprintf('norm_diff_forward: %f\n', norm(x-x_ml));
fprintf('norm_diff_adjoint: %f\n', norm(y-y_ml));
fprintf('norm_diff_rowsum: %f\n', norm(rowsum-rowsum_ml));
fprintf('norm_diff_colsum: %f\n', norm(colsum-colsum_ml));

%%
% Data term prec relax
nx = 123;
ny = 31;
L = 20;
t_min=0;
t_max=255;

linop = { linop_data_prec(0, 0, nx, ny, L, t_min, t_max) };

inp = rand(nx*ny*L+2*nx*ny*(L-1), 1);
inp2 = rand(nx*ny*L, 1);

[x,~,~] = pdsolver_eval_linop(linop, inp, false);
[y,rowsum,colsum] = pdsolver_eval_linop(linop, inp2, true);

K = spmat_data_prec(nx, ny, L, t_min, t_max);

tic;
x_ml = K * inp;
y_ml = K' * inp2;
toc;

rowsum_ml = sum(abs(K), 2);
colsum_ml = sum(abs(K), 1)';

fprintf('norm_diff_forward: %f\n', norm(x-x_ml));
fprintf('norm_diff_adjoint: %f\n', norm(y-y_ml));
fprintf('norm_diff_rowsum: %f\n', norm(rowsum-rowsum_ml));
fprintf('norm_diff_colsum: %f\n', norm(colsum-colsum_ml));

%%
% Linop data prec 2
nx = 211;
ny = 103;
L = 33;
t_min=0.98;
t_max=4.68;
N=nx*ny;
Q = kron(speye(N), -ones(L-1, 1))';
linop = { linop_data_prec(0, 0, nx, ny, L, t_min, t_max);
          linop_zero(N*L, 0, N, N*L+N*(L-1)); linop_sparse(N*L, N*L+N*(L-1), Q);...
          linop_zero(N*L+N, 0, N*2*(L-1), N*L); linop_identity(N*L+N, N*L, N*2*(L-1))
};
inp2 = rand(N*L + N + 2*N*(L-1), 1);
inp = rand(N*L + 2*N*(L-1), 1);

[x,~,~] = pdsolver_eval_linop(linop, inp, false);
[y,rowsum,colsum] = pdsolver_eval_linop(linop, inp2, true);

K = spmat_data_prec_2(nx, ny, L, t_min, t_max);

tic;
x_ml = K * inp;
y_ml = K' * inp2;
toc;

rowsum_ml = sum(abs(K), 2);
colsum_ml = sum(abs(K), 1)';

fprintf('norm_diff_forward: %f\n', norm(x-x_ml));
fprintf('norm_diff_adjoint: %f\n', norm(y-y_ml));
fprintf('norm_diff_rowsum: %f\n', norm(rowsum-rowsum_ml));
fprintf('norm_diff_colsum: %f\n', norm(colsum-colsum_ml));

return;


%%
% Data term prec graph relax
nx = 113;
ny = 200;
L = 50;
t_min=0;
t_max=1;

linop = { linop_data_graph_prec(0, 0, nx, ny, L, t_min, t_max) };

inp = rand(nx*ny*(L-1), 1);
inp2 = rand(nx*ny*(L-1), 1);

[x,~,~] = pdsolver_eval_linop(linop, inp, false);
[y,rowsum,colsum] = pdsolver_eval_linop(linop, inp2, true);

K = spmat_data_graph_prec(nx, ny, L, t_min, t_max);

tic;
x_ml = K * inp;
y_ml = K' * inp2;
toc;

rowsum_ml = sum(abs(K), 2);
colsum_ml = sum(abs(K), 1)';

fprintf('norm_diff_forward: %f\n', norm(x-x_ml));
fprintf('norm_diff_adjoint: %f\n', norm(y-y_ml));
fprintf('norm_diff_rowsum: %f\n', norm(rowsum-rowsum_ml));
fprintf('norm_diff_colsum: %f\n', norm(colsum-colsum_ml));


%% Linop graph data prec 2
nx = 100;
ny = 55;
L = 30;
t_min=0;
t_max=1;

delta_t = (t_max - t_min) / (L-1);
t = t_min:delta_t:t_max;
linop = { linop_data_graph_prec(0, 0, nx, ny, L, t_min, t_max);...
          linop_identity(nx*ny*(L-1), 0, nx*ny*(L-1));...
          linop_sparse(2*nx*ny*(L-1), 0, kron(speye(nx*ny), t(1:end-1)-t(2:end)))...
        };

inp = rand(nx*ny*(L-1), 1);
inp2 = rand(2*nx*ny*(L-1)+nx*ny, 1);

[x,~,~] = pdsolver_eval_linop(linop, inp, false);
[y,rowsum,colsum] = pdsolver_eval_linop(linop, inp2, true);

K = spmat_data_graph_prec_2(nx, ny, L, t_min, t_max);

tic;
x_ml = K * inp;
y_ml = K' * inp2;
toc;

rowsum_ml = sum(abs(K), 2);
colsum_ml = sum(abs(K), 1)';

fprintf('norm_diff_forward: %f\n', norm(x-x_ml));
fprintf('norm_diff_adjoint: %f\n', norm(y-y_ml));
fprintf('norm_diff_rowsum: %f\n', norm(rowsum-rowsum_ml));
fprintf('norm_diff_colsum: %f\n', norm(colsum-colsum_ml));


%%
% Gradient
nx = 300;
ny = 220;
L = 8;
%linop = { linop_gradient2d(0, 0, nx, ny, L) };
linop = { linop_gradient3d(0, 0, nx, ny, L) };

inp = rand(nx*ny*L, 1);
inp2 = rand(nx*ny*L*3, 1);

[x,~,~] = pdsolver_eval_linop(linop, inp, false);
[y,rowsum,colsum] = pdsolver_eval_linop(linop, inp2, true);

%K = spmat_gradient2d(nx, ny, L);
K = spmat_gradient3d(nx, ny, L);

tic;
x_ml = K * inp;
y_ml = K' * inp2;
toc;

rowsum_ml = sum(abs(K), 2);
colsum_ml = sum(abs(K), 1)';

fprintf('norm_diff_forward: %f\n', norm(x-x_ml));
fprintf('norm_diff_adjoint: %f\n', norm(y-y_ml));
fprintf('sum_gt_rowsum: %f\n', full(sum(rowsum<rowsum_ml)));
fprintf('sum_gt_colsum: %f\n', full(sum(colsum<colsum_ml)));

%%
% diags test
Ndiags = 29;
nrows = 5912;
ncols = 1131;
By=3;
Bx=9;

% build big block matrix
K = [];
linop = {};
idx = 1;
row = 0;
for i=1:By
    
    K_row = [];
    col = 0;
    for j=1:Bx
        factors = rand(Ndiags, 1);
        perm = randperm(nrows + ncols - 2);
        offsets = perm(1:Ndiags)' - nrows + 1;
        K_row = cat(2, K_row, spdiags(ones(nrows, 1) * factors', offsets, ...
                              nrows, ncols));
        
        linop{idx, 1} = linop_diags(row, col, nrows, ncols, factors, ...
                                    offsets);
        idx = idx + 1;
        col = col + ncols;
    end
    
    row = row + nrows;
    K = cat(1, K, K_row);
end

inp = randn(ncols * Bx, 1);
inp2 = randn(nrows * By, 1);

[x,~,~] = pdsolver_eval_linop(linop, inp, false);
[y,rowsum,colsum] = pdsolver_eval_linop(linop, inp2, true);

tic;
x_ml = K * inp;
y_ml = K' * inp2;
toc;

rowsum_ml = sum(abs(K), 2);
colsum_ml = sum(abs(K), 1)';

fprintf('norm_diff_forward: %f\n', norm(x-x_ml));
fprintf('norm_diff_adjoint: %f\n', norm(y-y_ml));
fprintf('norm_diff_rowsum: %f\n', norm(rowsum-rowsum_ml));
fprintf('norm_diff_colsum: %f\n', norm(colsum-colsum_ml));
