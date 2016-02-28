function [passed] = test_linop_sparse_kron_id()

    diaglength = 50 * 50;
    nrows = 81;
    ncols = 64;
    K_mat = sprand(nrows, ncols, 0.05);
    % sum(sum(abs(K_mat)>0))
    
    linop = { ...
        prost.block.sparse_kron_id(0, 0, K_mat, diaglength); ...
        prost.block.sparse_kron_id(nrows * diaglength, 0, K_mat, diaglength);
        prost.block.sparse_kron_id(nrows * diaglength, ncols * diaglength, K_mat, diaglength);
        prost.block.sparse_kron_id(0, ncols * diaglength, K_mat, diaglength);
            };
    
    K = kron(K_mat, speye(diaglength));
    K = [K K; K K];
    
    inp = randn(diaglength * ncols * 2, 1);
    inp_trans = randn(diaglength * nrows * 2, 1);
    
    [x, rowsum, colsum, time] = prost.eval_linop(linop, inp, false);
    [x_t, ~, ~, time_t] = prost.eval_linop(linop, inp_trans, true);
    
    tic;
    x_matlab_t = K' * inp_trans;
    time_matlab_t = toc * 1000;

    tic;
    x_matlab = K * inp;
    time_matlab = toc * 1000;
    
    %fprintf('CUDA: %f %f, MATLAB: %f %f\n', time, time_t, time_matlab, ...
    %        time_matlab_t);
    
    norm_diff = max(abs(x - x_matlab));
    norm_diff_t = max(abs(x_t - x_matlab_t));
    rowsum_diff = max(abs(rowsum - sum(abs(K),2)));
    colsum_diff = max(abs(colsum - sum(abs(K),1)'));
    if norm_diff > 1e-3
        fprintf('failed! Reason: norm_diff > 1e-4: %f\n', norm_diff);
        passed = false;
        
        return;
    end

    if norm_diff_t > 1e-4
        fprintf('failed! Reason: norm_diff_t > 1e-4: %f\n', norm_diff_t);
        passed = false;

        return;
    end
    
    if rowsum_diff > 1e-4
        fprintf('failed! Reason: rowsum_diff > 1e-4: %f\n', rowsum_diff);

        passed = false;
        return;
    end
    
    if colsum_diff > 1e-4
        fprintf('failed! Reason: colsum_diff > 1e-4: %f\n', colsum_diff);

        passed = false;
        return;
    end
    
    passed = true;
    
end