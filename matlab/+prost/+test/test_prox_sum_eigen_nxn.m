function [passed] = test_prox_sum_eigen_nxn()

    % generate random points and project them onto the PSD cone
    N=30000;
    n = 13;
    dim = n*(n - 1) / 2 + n;
    
    P = randn(dim,N)*10;

    R = zeros(dim,N);
    
    for t=1:N
        M = zeros(n,n);
        
        k = 1;
        for i=0:n-1
            for j=1:n-i
                M(j, j+i) = P(k,t);
                M(j+i, j) = P(k,t);
                k = k + 1;
            end
        end

        [V, S, Y] = eig(M);
       
        J = V*max(0,S)*Y';
        
        k = 1;
        for i=0:n-1
            for j=1:n-i
                R(k,t) = J(j, j+i);
                k = k + 1;
            end
        end
    end

    tau = 1;
    Tau = ones(N*dim, 1);

    % c * f_{alpha,beta}(ax - b) + dx + 0.5 ex^2
    
    [Q, time] = prost.eval_prox( prost.function.sum_eigen_nxn(n, true, ...
        'ind_leq0', -ones(N,1), 0, ones(N,1), 0, 0, 0, 0), P(:), tau, Tau);
   
    Q = reshape(Q,dim,N);
    

    passed = true;
    
    norm_diff = norm(Q(:)-R(:), Inf)
    time
    
    if norm_diff > 1e-4
        fprintf('failed! Reason: norm_diff > 1e-4: %f\n', norm_diff);
        passed = false;
        
        return;
    end
 end

