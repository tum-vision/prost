function [func] = sum_singular_nx2(dim, interleaved, fun, a, b, c, d, e, alpha, beta)
% SUM_SINGULAR_NX2  func = sum_singular_nx2(dim, interleaved, fun,
%                                           a, b, c, d, e, alpha, beta)
%  
%  Function on the singular values of a Nx2-sized matrix M. Assumes
%  row-first storage of the matrix M. The function is parametrized as
%
%    f(M) = h(sigma_1) + h(sigma_2),
%
%  where h is parametrized by a, b, c, d, e, alpha, beta as in
%  sum_1d. (See 'help prost.function.sum_1d').


    
    switch nargin
    case 9
        beta = 0;
    
    case 8
        alpha = 0;
        beta = 0;
    end

    coeffs = { a, b, c, d, e, alpha, beta };


    
    func = @(idx, count) { strcat('elem_operation:singular_nx2:', fun), idx, count, false, { count / dim, dim, interleaved, coeffs } };
    
end
