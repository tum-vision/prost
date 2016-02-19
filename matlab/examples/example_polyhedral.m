% Project onto epigraph of polyhedron given as maximum over linear
% functions

%seed = randi([1, 10000]);
rng(1589);
m = 5; % number of linear functions
coeff_a = linspace(-0.5, 0.5, m)' + 0.05*randn(m,1);
coeff_b = -linspace(-3, 3, m)';

N = 700 * 500; % number of points to project
x0 = 6 + 5 * randn(N, 1);
y0 = 2 + 5 * randn(N, 1);

rep_a = repmat(coeff_a, N, 1);
rep_b = repmat(coeff_b, N, 1);
count_vec = repmat(m, N, 1);
% index_vec = zeros(N, 1);
% for i=0:(N-2)
%     index_vec(N-i) = sum(count_vec(i+2:end));
% end
index_vec = cumsum(count_vec) - m;

prox = prost.prox.sum_ind_epi_polyhedral(0, N, 2, rep_a, rep_b, ...
                                         count_vec, index_vec);

arg = [x0; y0];
res = prost.eval_prox(prox, arg, 1, ones(2 * N, 1), true);

x_proj = res(1:N);
y_proj = res(N+1:end);

figure;
hold on;
for i=1:m
    p1_x = 0; 
    p1_y = coeff_b(i);
    
    p2_x = 1;
    p2_y = coeff_a(i) + coeff_b(i);
    
    dx = p2_x - p1_x;
    dy = p2_y - p1_y;
    
    slope = dy / dx;
    
    p1_x = p1_x - 15;
    p1_y = p1_y - 15 * slope;
    p2_x = p2_x + 15;
    p2_y = p2_y + 15 * slope;

    plot([p1_x p2_x], [p1_y p2_y], 'k--')
end

for i=1:min(N,100)
    plot(x0(i), y0(i), 'ro');
    plot([x0(i) x_proj(i)], [y0(i) y_proj(i)], 'g--');
    plot(x_proj(i), y_proj(i), 'bo');
end

axis equal;
