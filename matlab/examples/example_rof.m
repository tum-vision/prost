%%
% load input image
im = imread('../../images/frame10.png');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.;

%%
% setup parameters
grad = spmat_gradient2d(nx,ny,nc);
lmb = 1;

%%
% setup min-max problem
u = prost.primal_variable(nx*ny*nc);
q = prost.dual_variable(2*nx*ny*nc);

u.fun = prost.function.sum_1d('square', 1, f, 1, 0, 0);
q.fun = prost.function.sum_1d(... 
    'ind_box01', 1 / (2 * lmb), -0.5, 1, 0, 0);

prost.set_dual_pair(u, q, prost.linop.sparse(grad));

prob = prost.min_max( {u}, {q} );

%%
% setup min problem
% u = prost.primal_variable(nx*ny*nc);
% g = prost.primal_variable(2*nx*ny*nc);

% u.fun = prost.function.sum_1d('square', 1, f, 1, 0, 0);
% g.fun = prost.function.sum_1d('abs', lmb, 0, 1, 0, 0);

% prost.set_constraint(u, g, prost.linop.sparse(grad));

% prob = prost.min( {u}, {g} );

%%
% options and solve
backend = prost.backend.pdhg();
opts = prost.options('max_iters', 1000);

result = prost.solve(prob, backend, opts);

Kx = grad * result.x;
norm(Kx - result.z)

Kty = -grad' * result.y;
norm(Kty - result.w)

%%
% show result
imshow(reshape(u.val, [ny nx nc]));
