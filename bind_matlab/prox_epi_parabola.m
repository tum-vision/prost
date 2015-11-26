% projection of (phi^x, phi^t) onto epi of parabola phi^t + g <=
% alpha * (|phi^x|^2)
% dim: dimension of phi^x
% g: offset
% assumes pixel-first ordering of the variables, i.e.
% - phi^x_1 (size: count)
% - ...
% - phi^x_dim (size: count)
% - phi^t (size: count)
function [prox] = prox_epi_parabola(idx, count, dim, g, alpha)

data = { g, alpha };
prox = { 'epi_parabola', idx, count, dim + 1, false, false, data };

end
