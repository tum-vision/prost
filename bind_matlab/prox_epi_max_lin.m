function [prox] = prox_epi_max_lin(idx, count, dim, interleaved, ...
                                      t, b, index, count_array)

data = { t, b, index, count_array };

prox = { 'epi_max_lin', idx, count, dim, interleaved, false, data };
end
