% test-bench for the different linear operators


nx = 300;
ny = 220;
L = 8;

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
