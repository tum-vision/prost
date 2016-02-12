function [prox] = sum_ind_epi_polyhedral_1d(idx, count, dim, interleaved, ...
                                            x, y, alpha, beta, ...
                                            ind_vec, cnt_vec)

    coeffs = { x, y, alpha, beta, ind_vec, cnt_vec };
    prox = { 'ind_epi_polyhedral_1d', idx, count*dim, false, ...
             { count, dim, interleaved, coeffs } };

end

