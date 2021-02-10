function [passed] = test_prox_sum_eigen_2x2()

    % generate random points and project them onto the PSD cone
    N=300000;
    P = randn(4,N)*10;
    R = zeros(4,N);
    
    for i=1:N
        M = zeros(2,2);
        M(1,1)=P(1,i);
        M(2,1)=(P(2,i)+P(3,i))/2;
        
        M(1,2)=M(2,1);
        M(2,2)=P(4,i);

        [V, S, Y] = eig(M);
       
        J = V*max(0,S)*Y';
        
        R(1,i)=J(1,1);
        R(2,i)=J(2,1);
        R(3,i)=J(1,2);
        R(4,i)=J(2,2);
    end

    tau = 1;
    Tau = ones(N*4, 1);

    % c * f_{alpha,beta}(ax - b) + dx + 0.5 ex^2
    
    [Q, time] = prost.eval_prox( prost.function.sum_eigen_2x2(true, ...
        'ind_leq0', -ones(N,1), 0, ones(N,1), 0, 0, 0, 0), P(:), tau, Tau);
   
    Q = reshape(Q,4,N);
    

    passed = true;
    
    norm_diff = norm(Q(:)-R(:), Inf)
    time
    
    if norm_diff > 1e-4
        fprintf('failed! Reason: norm_diff > 1e-4: %f\n', norm_diff);
        passed = false;
        
        return;
    end
 end

