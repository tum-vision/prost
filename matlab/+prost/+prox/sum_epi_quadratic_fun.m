function [prox] = proj_epi_quadratic_fun(idx, count, dim, interleaved, a, b, c)

coeffs = { a, b, c };
prox = { 'epi_quadratic_fun', idx, count*dim, true, { ...
    count, dim, interleaved, coeffs } };

end
