function [prox] = prox_halfspace(idx, count, dim, interleaved, a)

data = { a };
prox = { 'halfspace', idx, count, dim, interleaved, false, data };
end
