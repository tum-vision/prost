function [prox] = prox_simplex( idx, count, dim, interleaved, a )

data = { a };
prox = { 'simplex', idx, count, dim, interleaved, false, data };

end