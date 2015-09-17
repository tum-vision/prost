function ex_lifted_tv_callback(it, x, y, f, nx, ny, L)

    primal_en = f' * x(1:nx*ny*L);
    primal_feas = sum((sum(reshape(x(1:nx*ny*L), nx * ny, L)')-1).^2);
    
    fprintf('it %5d: en_prim=%E, feas_prim=%E \n', it, primal_en, primal_feas);
    
end
