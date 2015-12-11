function [prox] = prox_epi_piecew_lin_scaled( idx, count, interleaved, ...
                                              x, y, alpha, beta, ...
                                              index, count_array, scaling )

data = { x, y, alpha, beta, index, count_array, scaling };
prox = { 'epi_piecew_lin', idx, count, 2, interleaved, false, data };

end
