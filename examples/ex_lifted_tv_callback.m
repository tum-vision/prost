function ex_lifted_tv_callback(it, x, y, f, nx, ny, L, t, Kgrad, f2, ...
                               lmb)

%primal_en = f' * x(1:nx*ny*L);
%primal_feas = sum((sum(reshape(x(1:nx*ny*L), nx * ny, L)')-1).^2);
%fprintf('it %5d: en_prim=%E, feas_prim=%E \n', it, primal_en,
%primal_feas);
    

u = reshape(x(1:nx*ny*L), nx*ny, L) * t';
n=nx*ny;

grad = Kgrad * u;
gradnorms = sqrt(grad(1:n).^2 + grad(n+1:end).^2);
en_prim = 0.5 * sum((u-f2).^2) + lmb * sum(gradnorms);

global plot_primal;
global plot_iters;
plot_primal = cat(1, plot_primal, en_prim);
plot_iters = cat(1, plot_iters, it);
    
end
