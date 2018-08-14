function [passed] = test_prox_sum_ind_simplex()

    % generate random points and project them onto the simplex
    N=1000;
    d=17*17;

    P = -2 + 4 * rand(N, d);
    P = P(:);

    tau = 1;
    Tau = ones(N * d, 1);

    [Q, time] = prost.eval_prox( prost.function.sum_ind_simplex(d, false), P, tau, Tau);
    %fprintf('CUDA took %f ms\n', time);
    Q2 = zeros(size(P));
    tic;
    for i=0:(N-1)
        ind = 1 + i+(0:(d-1))*N;
        Q2(ind) = projsplx(P(ind));
    end
    t2=toc;
    %fprintf('MATLAB took %f ms\n', t2 * 1000);
    

    Q = reshape(Q, N, d);
    Q2 = reshape(Q2, N, d);
    P = reshape(P, N, d);

    if norm(Q-Q2, Inf) > 1e-5
        passed = false;
        return;
    end
    
    passed = true;

end
