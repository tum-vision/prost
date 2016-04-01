function [passed] = test_linop_gradient2d()
    
    nx = 307;
    ny = 229;
    L = 8;

    linop_2d = { prost.block.gradient2d(0, 0, nx*ny*2*L, nx*ny*L, ...
                                        nx, ny, L, false) };
    K_2d = spmat_gradient2d(nx, ny, L);

    inp = rand(nx*ny*L, 1);
    inp2 = rand(nx*ny*L*2, 1);

    [x,~,~] = prost.eval_linop(linop_2d, inp, false);
    [y,rowsum,colsum] = prost.eval_linop(linop_2d, inp2, true);

    x_ml = K_2d * inp;
    y_ml = K_2d' * inp2;

    rowsum_ml = sum(abs(K_2d), 2);
    colsum_ml = sum(abs(K_2d), 1)';

    if(norm(x - x_ml) > 1e-3)
        fprintf('failed! Reason: gradient2d_norm_diff_forward: %f\n', ...
                norm(x - x_ml));
        passed = false;
        return;
    end

    if(norm(y - y_ml) > 1e-3)
        fprintf('failed! Reason: gradient2d_norm_diff_adjoint: %f\n', ...
                norm(y - y_ml));
        passed = false;
        return;
    end

    if(full(sum(rowsum<rowsum_ml)) > 1e-3)
        fprintf('failed! Reason: gradient2d_diff_rowsum: %f\n', ...
                full(sum(rowsum<rowsum_ml)));
        passed = false;
    end

    if(full(sum(colsum<colsum_ml)) > 1e-3)
        fprintf('failed! Reason: gradient2d_diff_rowsum: %f\n', ...
                full(sum(colsum<colsum_ml)));
        passed = false;
    end

    passed = true;
    
end
