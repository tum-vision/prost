function [passed] = test_prox_sum_ind_epi_polyhedral()

    for d=1:2
        m = 25; % number of linear "maxlin" functions

        seed = randi([0, 10000]);
        %rng(8954);
        rng(seed);

        coeff_a = randn(m, d);
        coeff_b = randn(m, 1);

        N = 250; % number of points to project
        x0 = 1000 * randn(N, d);
        y0 = 1000 * randn(N, 1);

        tmp = coeff_a';
        rep_a = repmat(tmp(:), N, 1);
        rep_b = repmat(coeff_b, N, 1);
        count_vec = repmat(m, N, 1);
        index_vec = zeros(N, 1);
        for i=0:(N-2)
            index_vec(N-i) = sum(count_vec(i+2:end));
        end
        index_vec = cumsum(count_vec) - m;

        prox = prost.function.sum_ind_epi_polyhedral(d + 1, false, rep_a, rep_b, ...
                                                 count_vec, index_vec);

        arg = [x0(:); y0];
        [res, t] = prost.eval_prox(prox, arg, 1, ones(N * (d + 1), 1), true);
        %fprintf('Prox took %f ms.\n', t);

        x_proj = res(1:N*d);
        y_proj = res(N*d+1:end);

        opts = optimoptions('quadprog',...
                            'Algorithm','interior-point-convex','Display','off');

        for i=1:N
            H = eye(d + 1);
            f = [-x0(i,:)'; -y0(i)];
            A = [coeff_a -ones(m, 1)];
            b = coeff_b;
        
            x = quadprog(H, f, A, b, [], [], [], [], [], opts);
        
            norm_diff = norm(x(1:d)-x_proj(i:N:end)) + ...
                norm(x(end) - y_proj(i));
        
            if norm_diff > 1e-3
                fprintf('failure! Point %3d has norm_diff = %f.\n', i, ...
                        norm_diff);
                passed = false;
                return
            end
        end
    end

    passed = true;
end
