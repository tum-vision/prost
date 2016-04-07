function [passed] = test_prox_transform()

    for i=1:10
        N=5000;
        a = rand(N, 1);
        b = rand(N, 1);
        c = rand(N, 1);
        d = rand(N, 1);
        e = rand(N, 1);
        y = rand(N, 1);
        tau = rand(1, 1);
        Tau = rand(N, 1);

        %% evaluate prox directly
        x = prost.eval_prox( prost.function.conjugate(prost.function.sum_1d('abs', ...
                                           a, b, c, d, e)), y, ...
                             tau, Tau);
    
        %% evaluate prox via transformation
        prox_1d = prost.function.sum_1d('abs', 1, 0, 1, 0, 0);
        
        x2 = prost.eval_prox( prost.function.conjugate(prost.function.transform( ...
            prox_1d, a, b, c, d, e)), y, tau, Tau );

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

