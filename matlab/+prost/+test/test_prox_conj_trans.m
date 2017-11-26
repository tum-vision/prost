function [passed] = test_prox_conj_trans()

    for i=1:10
        N=500;
        b = rand(N, 1);

        y = rand(N, 1);
        tau = rand(1, 1);
        Tau = rand(N, 1);

        %% evaluate shifted prox directly
        x = prost.eval_prox( prost.function.conjugate(prost.function.sum_1d('abs', ...
                                                          1, b, 1, 0, 0)), ...
                             y, tau, Tau);
    
        %% evaluate using the conjugate shifting formula
        prox_1d = prost.function.sum_1d('abs', 1, 0, 1, 0, 0);
        
        x2 = prost.eval_prox( prost.function.transform(...
            prost.function.conjugate(prox_1d), 1, 0, 1, b, 0), y, tau, Tau );

        %% check if result is the same
        diff = x - x2;
    
        if norm(diff, Inf) > 1e-5
            fprintf(' failed! norm_diff = %f\n', norm(diff, Inf));
            passed = false;
            return;
        end
    end

    passed = true;
   
end
