function [passed] = test_prox_sum_eigen_3x3()

    % generate random points and project them onto the PSD cone
    N=300000;
    P = randn(6,N)*10;

    R = zeros(6,N);
    
    for i=1:N
        M = zeros(3,3);
        M(1,1)=P(1,i);
        M(2,2)=P(2,i);    
        M(3,3)=P(3,i);
        
        M(1,2)=P(4,i);
        M(2,1)=P(4,i);
        
        M(2,3)=P(5,i);
        M(3,2)=P(5,i);
        
        M(1,3)=P(6,i);
        M(3,1)=P(6,i);

        [V, S, Y] = eig(M);
       
        J = V*max(0,S)*Y';
        
        R(1,i)=J(1,1);
        R(2,i)=J(2,2);    
        R(3,i)=J(3,3);
        R(4,i)=J(1,2);
        R(5,i)=J(2,3);     
        R(6,i)=J(1,3);
    end

    tau = 1;
    Tau = ones(N*6, 1);

    % c * f_{alpha,beta}(ax - b) + dx + 0.5 ex^2
    
    [Q, time] = prost.eval_prox( prost.function.sum_eigen_3x3(true, ...
        'ind_leq0', -ones(N,1), 0, ones(N,1), 0, 0, 0, 0), P(:), tau, Tau);
   
    Q = reshape(Q,6,N);
    

    passed = true;
    
    norm_diff = norm(Q(:)-R(:), Inf)
    time
    
    if norm_diff > 1e-4
        fprintf('failed! Reason: norm_diff > 1e-4: %f\n', norm_diff);
        passed = false;
        
        return;
    end
 end
