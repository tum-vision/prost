% test-bench for the different linear operators


% gradient

left=0;
right=1;

data_linop = { linop_data_prec(0, 0, nx, ny, L, left, right) };

%grad_linop = { linop_gradient2d(0, 0, nx, ny, L) };
%grad_linop = { linop_sparse(0, 0, grad) };

inp = rand(nx*ny*L+2*nx*ny*(L-1), 1);

% compute laplacian using CUDA
y = pdsolver_eval_linop(data_linop, inp, false);
x = pdsolver_eval_linop(data_linop, y, true);
%%
% compute laplacian using MATLAB
%grad = spmat_gradient2d(nx, ny, L);
data = spmat_data_prec(nx, ny, L, left, right);

tic;
y_ml = data * inp;
x_ml = data' * y_ml;
toc;

norm(x_ml - x)
%norm(y_ml - y);

%nx = 300;
%ny = 220;
%L = 8;
% grad_linop = { linop_gradient3d(0, 0, nx, ny, L) };
% %grad_linop = { linop_sparse(0, 0, grad) };

% inp = rand(nx*ny*L, 1);

% % compute laplacian using CUDA
% y = pdsolver_eval_linop(grad_linop, inp, false);
% x = pdsolver_eval_linop(grad_linop, y, true);

% % compute laplacian using MATLAB
% tic;
% y_ml = grad * inp;
% x_ml = grad' * y_ml;
% toc;

% norm(x_ml - x)

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

>>>>>>> 4e7b65e559ce94aeff3eda474c5ec97469b6c829
