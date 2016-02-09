function [prox] = prox_simplex(idx, count, dim, interleaved)

prox = { 'elem_operation:simplex', idx, count*dim, false, { count, dim, interleaved } };

end