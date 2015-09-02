function [prox] = prox_norm2(idx, count, dim, interleaved, fn, a, b, c, d, e)
% computes prox of general class of functions
% sum_{i=1}^{count} h(||x||_2), where ||.||_2 denotes the dim-dimensional
% euclidean norm. h is given as h(x) = c f(ax - b) + dx + 0.5ex^2 
% fn is a string describing the 1d function f

data = { fn, a, b, c, d, e };
prox = { 'norm2', idx, count, dim, interleaved, false, data };

end

