function [passed] = test_prox_sum_ind_sum()

    N = 21;
    tau = 1;
    Tau = ones(N, 1);
    d = 3;
    y = randn(N, 1);

    [x, time] = prost.eval_prox( prost.function.sum_ind_sum(d, false), y, tau, Tau);
    x = reshape(x, [N/d, d]);

    diff = sum(x, 2) - ones(N/d, 1);

    if norm(diff, Inf) > 1e-5
        passed = false;
        return;
    end

    passed = true;
end
