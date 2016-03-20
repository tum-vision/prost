function [passed] = test_linop_dense()

    N = 534;
    M = 718;

    A = randn(M, N);
    x = randn(N, 1);
    y = randn(M, 1);

    tic;
    fw_mat = A * x;
    t_fw_mat = toc * 1000;

    tic;
    ad_mat = A' * y;
    t_ad_mat = toc * 1000;

    rs_mat = sum(abs(A),2);
    cs_mat = sum(abs(A),1)';

    linop = { prost.block.dense(0, 0, A); };

    [fw_cuda, rs_cuda, cs_cuda, t_fw_cuda] = ...
        prost.eval_linop( linop, x, false );

    [ad_cuda, ~, ~, t_ad_cuda] = ...
        prost.eval_linop( linop, y, true );

    % fprintf(['time_fw_cuda=%f, time_ad_cuda=%f, time_fw_mat=%f, ' ...
    %          'time_ad_mat=%f\n'], t_fw_cuda, t_ad_cuda, t_fw_mat, ...
    %         t_ad_mat);

    if norm(fw_cuda - fw_mat, 'inf') > 1e-3
        fprintf('failed! Reason: norm_diff_forward > 1e-3: %f\n', norm(fw_cuda-fw_mat));
        passed = false;
            
        return;
    end

    if norm(ad_cuda - ad_mat, 'inf') > 1e-3
        fprintf('failed! Reason: norm_diff_adjoint > 1e-3: %f\n', norm(ad_cuda-ad_mat));
        passed = false;
            
        return;
    end

    if norm(rs_mat - rs_cuda, 'inf') > 1e-3
        fprintf('failed! Reason: norm_diff_rowsum > 1e-3: %f\n', norm(rs_cuda-rs_mat));
        passed = false;
            
        return;
    end
    
    if norm(cs_mat - cs_cuda, 'inf') > 1e-3
        fprintf('failed! Reason: norm_diff_colsum > 1e-3: %f\n', norm(cs_cuda-cs_mat));
        passed = false;
            
        return;
    end

    passed = true;
    
end