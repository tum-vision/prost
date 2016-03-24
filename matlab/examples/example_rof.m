im = imread('../../images/frame10.png');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.;
grad = spmat_gradient2d(nx,ny,nc);
lmb = 1;

u = prost.primal_variable(nx*ny*nc);
q = prost.dual_variable(2*nx*ny*nc);

u.fun = prost.function.sum_1d('square', 1, f, 1, 0, 0);
q.fun = prost.function.sum_1d(... 
    'ind_box01', 1 / (2 * lmb), -0.5, 1, 0, 0);

prost.set_dual_pair(u, q, prost.linop.sparse(grad));

prob = prost.min_max( {u}, {q} );

backend = prost.backend.pdhg();
opts = prost.options('max_iters', 10000);

prost.solve(prob, backend, opts);

imshow(reshape(u.val, [ny nx nc]));
