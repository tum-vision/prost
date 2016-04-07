% TODO: integrate into unit test framework

N=10;
d=6;

P = -2 + 4 * rand(N, d);
P = P(:);

tau = 1;
Tau = ones(N * d, 1);

% h(x) = c f(ax - b) + dx + 0.5ex^2

Q = prost.eval_prox( prost.prox.sum_singular_nx2(false, 'sum_1d:abs', ...
                                          ones(N,1), 1, ones(N,1), 0, 0), P, tau, Tau);

P = reshape(P, N, d);


