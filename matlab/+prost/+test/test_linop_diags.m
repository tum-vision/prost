function [passed] = test_linop_diags()
    
    Ndiags = 29;
    nrows = 5912;
    ncols = 1131;
    By=3;
    Bx=9;

    % build big block matrix
    K = [];
    linop = {};
    idx = 1;
    row = 0;
   
    for i=1:By
        K_row = [];
        col = 0;

        for j=1:Bx
            factors = rand(Ndiags, 1);
            perm = randperm(nrows + ncols - 2);
            offsets = perm(1:Ndiags)' - nrows + 1;
            K_row = cat(2, K_row, spdiags(ones(nrows, 1) * factors', offsets, ...
                                          nrows, ncols));

            block_fun = prost.block.diags(nrows, ncols, factors, offsets);
            make_block_diags = block_fun(row, col, nrows, ncols);
            
            linop{idx, 1} = make_block_diags{1};

            idx = idx + 1;
            col = col + ncols;
        end

        row = row + nrows;
        K = cat(1, K, K_row);
    end

    inp = randn(ncols * Bx, 1);
    inp2 = randn(nrows * By, 1);

    [x,~,~] = prost.eval_linop(linop, inp, false);
    [y,rowsum,colsum] = prost.eval_linop(linop, inp2, true);

    x_ml = K * inp;
    y_ml = K' * inp2;

    rowsum_ml = sum(abs(K), 2);
    colsum_ml = sum(abs(K), 1)';

    if norm(y-y_ml) > 1e-3
        fprintf('failed! Reason: norm_diff_adjoint > 1e-3: %f\n', norm(y-y_ml));
        passed = false;
            
        return;
    end
    
    if norm(x-x_ml) > 1e-3
        fprintf('failed! Reason: norm_diff_forward > 1e-3: %f\n', norm(x-x_ml));
        passed = false;
        
        return;
    end
    
    if norm(rowsum-rowsum_ml) > 1e-3
        fprintf('failed! Reason: norm_diff_rowsum > 1e-3: %f\n', norm(rowsum-rowsum_ml));
        passed = false;
        return;
    end

    if norm(colsum-colsum_ml) > 1e-3
        fprintf('failed! Reason: norm_diff_colsum > 1e-3: %f\n', norm(colsum-colsum_ml));
        passed = false;
        return;
    end

    passed = true;
    
end
