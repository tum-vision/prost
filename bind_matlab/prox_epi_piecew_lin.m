function [prox] = prox_epi_piecew_lin( idx, count, interleaved, ...
                                      x, y, alpha, beta, index, count_array )

data = { x, y, alpha, beta, index, count_array };
prox = { 'epi_piecew_lin', idx, count, 2, interleaved, false, data };

end
