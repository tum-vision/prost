%%
% load input image
im = imread('../../images/flowers.png');
im = imresize(im, 1);
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.; % convert to [0, 1]

%%
% parameters
grad = spmat_gradient2d(nx,ny,nc);
lmb = 100;

%kernel = fspecial('motion', 15, 45);

%B = convmtx2(kernel, ny, nx);
%B = kron(speye(nc), B);

%kx = size(kernel, 2);
%ky = size(kernel, 1);

%nx2 = nx + kx - 1;
%ny2 = ny + ky - 1;

%f_blurred = B * f;
%f_blurred = f_blurred + 0.05 * randn(ny2*nx2*nc,1);

%%
% problem
u = prost.variable(nx*ny*nc);
v = prost.variable(nx2*ny2*nc);
g = prost.variable(2*nx*ny*nc);

prob = prost.min_problem( {u}, {v, g} );
prob.add_function(v, prost.function.sum_1d('square', 1, f_blurred, lmb, 0, 0));
prob.add_function(g, prost.function.sum_norm2(2 * nc, false, 'abs', 1, 0, 1, 0, 0));
prob.add_constraint(u, v, prost.block.sparse(B));
prob.add_constraint(u, g, prost.block.sparse(grad));


backend = prost.backend.pdhg('stepsize', 'boyd', ...
                             'residual_iter', 1);


opts = prost.options('max_iters', 25000, ...
                     'num_cback_calls', 250, ...
                     'verbose', true, ...
                     'tol_rel_primal', 1e-4, ...
                     'tol_abs_primal', 1e-4, ...
                     'tol_rel_dual', 1e-4, ...
                     'tol_abs_dual', 1e-4);

tic;
result = prost.solve(prob, backend, opts);
toc;

blurry_img = reshape(f_blurred, [ny2 nx2 nc]);
imshow([blurry_img(floor(ky/2)+1:ny+floor(ky/2), floor(kx/2)+1:nx+ ...
                   floor(kx/2), :) reshape(u.val, ...
                                           [ny nx nc])]);
