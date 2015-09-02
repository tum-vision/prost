function [prox] = prox_epi_conjquadr( idx, count, interleaved, ...
                                      a, b, c, alpha, beta )

data = { a, b, c, alpha, beta };
prox = { 'epi_conjquadr', idx, count, 2, interleaved, false, data };

end
