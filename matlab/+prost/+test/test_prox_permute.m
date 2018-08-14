function [passed] = test_prox_permute()

    n = 34;

    y = 10 * randn(n, 1);
    tau = 0.1;
    Tau = ones(n, 1);
    perm = randperm(n) - 1;
    inv_perm(perm+1) = 1:length(perm);

    x1 = prost.eval_prox( prost.function.permute(...
        prost.function.sum_norm2(2, false, 'ind_leq0', 1, 1, 1), perm), ...
                          y, tau, Tau );

    x2 = prost.eval_prox( prost.function.sum_norm2(2, false, 'ind_leq0', ...
                                                   1, 1, 1), y(perm+1), tau, Tau );

    diff = x1 - x2(inv_perm);
    if(norm(diff, Inf) > 1e-5)
        passed = false;
        fprintf(' failed! norm_diff = %f\n', norm(diff, Inf));
        return;
    end

    passed = true;
end
