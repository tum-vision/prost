function [func] = sum_eigen_nxn(n, interleaved, fun, a, b, c, d, e, alpha, beta)
% SUM_IND_PSD_CONE_3x3  func = sum_ind_psd_cone_3x3(interleaved)
%

    if n > 32
        error('n must be less than 32');
    end
    
    dim = n*(n - 1) / 2 + n;
    
    switch nargin
    case 9
        beta = 0;
    
    case 8
        alpha = 0;
        beta = 0;
    end

    coeffs = { a, b, c, d, e, alpha, beta };


    func = @(idx, count) { strcat('elem_operation:eigen_nxn:', fun), ...
        idx, count, false, ...
        { count / dim, dim, interleaved, coeffs } };
end

