n = 5;

y = 10 * randn(n, 1);
tau = 0.1;
Tau = ones(n, 1);
perm = randperm(n) - 1;

% TODO: need to test with non-separable function, i.e. sum_norm2
x1 = prost.eval_prox( prost.function.permute(prost.function.sum_1d('abs'), perm), y, tau, Tau );
x2 = prost.eval_prox( prost.function.sum_1d('abs'), y, tau, Tau );
