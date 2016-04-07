%%
% load input image
im = imread('../../images/dog.png');
[ny, nx, nc] = size(im);
f = double(im(:)) / 255.; % convert to [0, 1]

%%
% parameters
grad = spmat_gradient2d(nx,ny,nc);
lmb = 0.3;

%%
% problem
q = prost.variable(2*nx*ny*nc);
w = prost.variable(nx*ny*nc);

prob = prost.min( {q}, {w} );
prob.add_function(q, prost.function.sum_norm2(... 
    2 * nc, false, 'ind_leq0', 1, 1, 1, 0, 0));
prob.add_function(w, prost.function.sum_1d(...
    'square', 1, -f * lmb, 1 / lmb, 0, 0));
prob.add_constraint(q, w, prost.linop.sparse(-grad'));

%%
% specify solver options
backend = prost.backend.pdhg('stepsize', 'goldstein', ...
                             'residual_iter', 100);

pd_gap_callback = @(it, x, y) example_rof_pdgap(it, y, x, grad, ...
                                                f, lmb, ny, nx, nc);

opts = prost.options('max_iters', 20000, ...
                     'interm_cb', pd_gap_callback, ...
                     'num_cback_calls', 100, ...
                     'verbose', false);

tic;
result = prost.solve(prob, backend, opts);
toc;

% We solved the dual problem, but are ultimately interested in the 
% primal solution. The get_all_variables function allows to read
% out the full primal and dual variables. This is enough, since the 
% dual variables of the dual problem are the primal variables we
% are interested in.
u = prost.variable(nx*ny*nc);
prost.get_all_variables(result, {}, {}, {u}, {});

%%
% show result
imshow(reshape(u.val, [ny nx nc]));
