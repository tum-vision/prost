% TODO: integrate in test framework
% broken with new matlab interface
N=1;
d=3;

P = [2;-5;9];


tau = 1;
Tau = ones(N * d, 1);

a =1;
b = [3; 1];
c = 2;

Q = prost.eval_prox( prost.prox.sum_ind_epi_quad(0, N, d, false, repmat(a, N, 1), repmat(b, N, 1), repmat(c, N, 1)), P, tau, Tau);
figure;
[X1,X2] = meshgrid(-5:.2:5, -5:.2:5);
X3 = 0.5 * a * (X1 .^2 + X2 .^2) + b(1) * X1 + b(2) * X2 + c;
surf(X1,X2,X3);
hold on;
scatter3(P(1), P(2), P(3));
scatter3(Q(1), Q(2), Q(3));

der = zeros(3,1);
der(1, 1) = a * Q(1) + b(1);
der(2, 1) = a * Q(2) + b(2);
der(3, 1) = -1;

quiver3(Q(1), Q(2), Q(3), der(1), der(2), der(3));

