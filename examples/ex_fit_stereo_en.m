
%%
IL = imread('data/motorcycle_im0.png');
IR = imread('data/motorcycle_im1.png');
%IL = IL(400:1400, 600:2200, :);
%IR = IR(400:1400, 600:2200, :);
scale = 0.2;
maxdisp = 56;
IL = double(imresize(IL, scale));
IR = double(imresize(IR, scale));

dimpatch = [1 0 0 1];
% Create patches
[C, V] = patches_from_im(IL, IR, dimpatch);
[~, c] = size(C);

for i=1:c
    C{i} = C{i} ./ 255;
    V{i} = V{i} ./ 255;
end

[m, n, ~] = size(IL);
n = n-maxdisp;
%%

% create the nonconvex discrete matching energie U
U = zeros(m*n*(maxdisp+1), 1);
for j=1:n
    j
    for i=1:m
        for l=1:(maxdisp+1)
            U((j-1)*m*(maxdisp+1) + (i-1)*(maxdisp+1) + l) = 0;
            for u=1:c
                U((j-1)*m*(maxdisp+1) + (i-1)*(maxdisp+1) + l) = U((j-1)*m*(maxdisp+1) + (i-1)*(maxdisp+1) + l) +...
                abs(C{u}(i, j+l-1) - V{u}(i, j));
            end
        end
    end
end
%%
i = randi([1 m],1, 1);
j = randi([1 n],1, 1);
%%
L = 15;
deg=2;
gamma = 0.9;
delta = 0.0;

%%
%A = fitDiscreteEnergy(U((j-1)*m*(maxdisp+1) + (i-1)*(maxdisp+1)+(1:(maxdisp+1))), 1, 1, maxdisp+1, L, deg, gamma, delta, iter);
range = maxdisp / (L-1);

[K, y] = linop_fit_en(1, 1, L, maxdisp+1, deg, U((j-1)*m*(maxdisp+1) + (i-1)*(maxdisp+1)+(1:(maxdisp+1))));

%% prox operators
prox_hc = { 
     prox_moreau(prox_1d(0, (L-1)*(range+1),'max_pos0', 1, y, gamma, -y, 1)); 
     prox_zero((L-1)*(range+1), (L-2)) };

prox_g = { prox_1d(0, (deg+1)*(L-1), 'abs', 1, 0, delta, 0, 0) };

%% solve problem
opts = pdsolver_opts();
opts.verbose = true;
opts.adapt = 'converge';
opts.bt_enabled = false;

opts.max_iters = 100000;
opts.cb_iters = 100;

opts.precond = 'alpha';
opts.precond_alpha = 1.;
opts.tol_primal = 0.025;
opts.tol_dual = 0.025;
opts.callback = @(it, x, y) disp('');
[A, qrs] = pdsolver(K, prox_g, prox_hc, opts);
A = reshape(A, [deg+1, L-1])';
%%
hold on;
x = (0:maxdisp);
scatter(x, U((j-1)*m*(maxdisp+1) + (i-1)*(maxdisp+1)+(1:(maxdisp+1))));

range = maxdisp / (L-1);
for k=1:L-1
    x = ((k-1)*range:0.01:k*range)';
    y = zeros(size(x));
    for l=1:deg+1
        y = y + A(k, l) * x.^(deg+1-l);
    end
    plot(x, y);
end
