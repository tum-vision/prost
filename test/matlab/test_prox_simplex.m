function [passed] = test_prox_simplex()

    % generate random points and project them onto the simplex
    N=6000;
    d=15;

    P = -2 + 4 * rand(N, d);
    P = P(:);

    tau = 1;
    Tau = ones(N * d, 1);

    Q = pdsolver_eval_prox( prox_simplex(0, N, d, false), P, tau, Tau);
    Q2 = zeros(size(P));
    for i=0:(N-1)
        ind = 1 + i+(0:(d-1))*N;
        Q2(ind) = projsplx(P(ind));
    end

    Q = reshape(Q, N, d);
    Q2 = reshape(Q2, N, d);
    P = reshape(P, N, d);

    if norm(Q-Q2, Inf) > 1e-5
        passed = false;
    end
    
    passed = true;

end
