function [prox] = prox_epi_conjquadr_scaled( idx, count, interleaved, ...
                                      a, b, c, alpha, beta, scaling )

data = { a, b, c, alpha, beta, scaling };
prox = { 'epi_conjquadr_scaled', idx, count, 2, interleaved, false, data };

end
