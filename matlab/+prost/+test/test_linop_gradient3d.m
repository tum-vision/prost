function [passed] = test_linop_gradient3d()
    
    nx = 151;
    ny = 291;
    L = 7;

    linop_3d = { prost.block.gradient3d(0, 0, nx, ny, L) };
    K_3d = spmat_gradient3d(nx, ny, L);

    inp = rand(nx*ny*L, 1);
    inp2 = rand(nx*ny*L*3, 1);

    [x,~,~] = prost.eval_linop(linop_3d, inp, false);
    [y,rowsum,colsum] = prost.eval_linop(linop_3d, inp2, true);

    x_ml = K_3d * inp;
    y_ml = K_3d' * inp2;

    rowsum_ml = sum(abs(K_3d), 2);
    colsum_ml = sum(abs(K_3d), 1)';

    if(norm(x - x_ml) > 1e-3)
        fprintf('failed! Reason: gradient3d_norm_diff_forward: %f\n', ...
                norm(x - x_ml));
        passed = false;
        return;
    end

    if(norm(y - y_ml) > 1e-3)
        fprintf('failed! Reason: gradient3d_norm_diff_adjoint: %f\n', ...
                norm(y - y_ml));
        passed = false;
        return;
    end

    if(full(sum(rowsum<rowsum_ml)) > 1e-3)
        fprintf('failed! Reason: gradient3d_diff_rowsum: %f\n', ...
                full(sum(rowsum<rowsum_ml)));
        passed = false;
    end

    if(full(sum(colsum<colsum_ml)) > 1e-3)
        fprintf('failed! Reason: gradient3d_diff_rowsum: %f\n', ...
                full(sum(colsum<colsum_ml)));
        passed = false;
    end

    passed = true;
    
end
