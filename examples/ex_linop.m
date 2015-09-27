% test-bench for the different linear operators

nx = 357;
ny = 291;
L = 24;

grad = spmat_gradient3d(nx, ny, L);
grad_linop = { linop_gradient3d(0, 0, nx, ny, L) };

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
