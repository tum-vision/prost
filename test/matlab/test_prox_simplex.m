% generate random points and project them onto the simplex

N=10000;
d=8;

P = -2 + 4 * rand(N, d);
P = P(:);

tau = 1;
Tau = ones(N * d, 1);

tic;
Q = pdsolver_eval_prox( prox_simplex(0, N, d, false), P, tau, Tau );
toc;

tic;
Q2 = zeros(size(P));
for i=0:(N-1)
    ind = 1 + i+(0:(d-1))*N;
    Q2(ind) = projsplx(P(ind));
end
toc;

Q = reshape(Q, N, d);
Q2 = reshape(Q2, N, d);
P = reshape(P, N, d);

norm(Q-Q2)

% figure;
% hold on;
% scatter(P(:, 1), P(:, 2), 25, 'red', 'filled');
% scatter(Q2(:, 1), Q2(:, 2), 50, 'yellow', 'filled');
% scatter(Q(:, 1), Q(:, 2), 25, 'blue', 'filled');
