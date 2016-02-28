function [passed] = test_linop_sparse_kron_id()

    diaglength = 64 * 64;
    nrows = 86;
    ncols = 92;
    K_mat = sprand(nrows, ncols, 0.1);
    
    linop = { ...
        prost.block.sparse_kron_id(0, 0, K_mat, diaglength) };
    
    K = kron(K_mat, speye(diaglength));
    
    inp = randn(diaglength * ncols, 1);
    inp_trans = randn(diaglength * nrows, 1);
    
    [x, rowsum, colsum, time] = prost.eval_linop(linop, inp, false);
    [x_t, ~, ~, time_t] = prost.eval_linop(linop, inp_trans, true);
    
    tic;
    x_matlab = K * inp;
    time_matlab = toc;

    tic;
    x_matlab_t = K' * inp_trans;
    time_matlab_t = toc;
    
    norm_diff = max(abs(x - x_matlab));
    norm_diff_t = max(abs(x_t - x_matlab_t));
    rowsum_diff = max(abs(rowsum - sum(abs(K),2)));
    colsum_diff = max(abs(colsum - sum(abs(K),1)'));
    if norm_diff > 1e-3
        fprintf('failed! Reason: norm_diff > 1e-3: %f\n', norm_diff);
        passed = false;
        
        return;
    end

    if norm_diff_t > 1e-3
        fprintf('failed! Reason: norm_diff_t > 1e-3: %f\n', norm_diff_t);
        passed = false;

        return;
    end
    
    if rowsum_diff > 1e-3
        fprintf('failed! Reason: rowsum_diff > 1e-3: %f\n', rowsum_diff);

        passed = false;
        return;
    end
    
    if colsum_diff > 1e-3
        fprintf('failed! Reason: colsum_diff > 1e-3: %f\n', colsum_diff);

        passed = false;
        return;
    end
    
    passed = true;
    
end