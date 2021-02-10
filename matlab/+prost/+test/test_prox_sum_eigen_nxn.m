function [passed] = test_prox_sum_eigen_nxn()

    % generate random points and project them onto the PSD cone
    N=30000;
    n = 5;
    dim = n*n;
    
    P = randn(dim,N)*10;

    R = zeros(dim,N);
    
    for t=1:N
        M = zeros(n,n);
        
        for i=1:n
            for j=i:n
                %[i, j, (i-1)*n+j]
                M(i,j) = (P((i-1)*n+j, t) + P((j-1)*n+i, t)) / 2;
                M(j,i) = M(i,j);
            end
        end

        [V, S, Y] = eig(M);
       
        J = V*max(0,S)*Y';
        
        for i=1:n
            for j=i:n
                R((i-1)*n+j,t) = J(i, j);
                R((j-1)*n+i,t) = J(i, j);
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


