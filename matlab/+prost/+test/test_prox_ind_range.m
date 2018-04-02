function [passed] = test_prox_ind_range()
    
  A = sparse(sprandn(500, 250, 0.1));
  AA = full(A' * A);
  L = sparse(chol(AA));
  N = size(A, 1);

  prox = prost.function.ind_range(A, AA);
  arg = randn(N, 1);
    
  [res, t] = prost.eval_prox(prox, arg, 1, ones(N, 1), true);
  tic; res_matlab = A * (L \ (L' \ (A' * arg))); time_matlab = toc;
  t = t / 1000;
  
  if norm(res - res_matlab) > 1e-4
      fprintf('failure! (time=%f, time_matlab=%f, diff=%f, factor=%f)\n', ...
              t, time_matlab, norm(res-res_matlab), t/time_matlab);
      passed = false;
  else
      passed = true;
  end
  
end
