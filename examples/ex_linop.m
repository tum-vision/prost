% test-bench for the different linear operators

nx = 512;
ny = 512;
L = 64;

tic;
grad = spmat_gradient3d(nx, ny, L);
toc;

grad_linop = { linop_gradient3d(0, 0, nx, ny, L) };
%grad_linop = { linop_sparse(0, 0, grad) };

inp = rand(nx*ny*L, 1);

% compute laplacian using CUDA
y = pdsolver_eval_linop(grad_linop, inp, false);
x = pdsolver_eval_linop(grad_linop, y, true);

% compute laplacian using MATLAB
tic;
y_ml = grad * inp;
x_ml = grad' * y_ml;
toc;

norm(x_ml - x)
