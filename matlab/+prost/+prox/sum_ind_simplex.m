function [prox] = sum_ind_simplex(idx, count, dim, interleaved)

prox = { 'elem_operation:ind_simplex', idx, count*dim, false, { count, dim, interleaved } };

end
