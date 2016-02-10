function [passed] = test_prox_sum_norm2()

    N=6000;
    d=7;

    P = -2 + 4 * rand(N, d);
    P = P(:);

    tau = 1;
    Tau = ones(N * d, 1);

    % h(x) = c f(ax - b) + dx + 0.5ex^2

    Q = prost.eval_prox( prost.prox.sum_norm2(0, N, d, false, 'ind_leq0', ...
                                              ones(N,1), 1, ones(N,1), 0, 0, 0, 0), P, tau, Tau);

    P = reshape(P, N, d);
    Pnorm = repmat(sqrt(sum(P.^2, 2)), 1, d);

    Q2 = zeros(N, d);
    Q2(Pnorm <= 1) = P(Pnorm <= 1);
    Q2(Pnorm > 1) = P(Pnorm > 1) ./ Pnorm(Pnorm > 1);

    Q = reshape(Q, N, d);
    %Q2 = reshape(Q2, N, d);

    diff = Q-Q2;
    passed = (norm(diff(:), Inf) < 1e-5);

end
