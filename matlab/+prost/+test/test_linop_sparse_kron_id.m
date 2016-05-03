function [passed] = test_linop_sparse_kron_id()

    diaglength = 64 * 48;
    nrows = 81;
    ncols = 64;
    K_mat = sprand(nrows, ncols, 0.001);
    % sum(sum(abs(K_mat)>0))
    
    block_fun = prost.block.sparse_kron_id(K_mat, diaglength);
    make_block_sparse_kron_id_1 = ...
        block_fun(0, ...
                  0, ...
                  size(K_mat, 1) * diaglength, ...
                  size(K_mat, 2) * diaglength);

    make_block_sparse_kron_id_2 = ...
        block_fun(size(K_mat, 1) * diaglength, ...
                  0, ...
                  size(K_mat, 1) * diaglength, ...
                  size(K_mat, 2) * diaglength);

    make_block_sparse_kron_id_3 = ...
        block_fun(size(K_mat, 1) * diaglength, ...
                  size(K_mat, 2) * diaglength, ...
                  size(K_mat, 1) * diaglength, ...
                  size(K_mat, 2) * diaglength);

    make_block_sparse_kron_id_4 = ...
        block_fun(0, ...
                  size(K_mat, 2) * diaglength, ...
                  size(K_mat, 1) * diaglength, ...
                  size(K_mat, 2) * diaglength);
    
    linop = { ...
        make_block_sparse_kron_id_1{1}; ...
        make_block_sparse_kron_id_2{1}; ...
        make_block_sparse_kron_id_3{1}; ...
        make_block_sparse_kron_id_4{1}; ...
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
    if norm_diff > 1e-4
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