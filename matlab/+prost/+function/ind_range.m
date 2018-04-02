function [func] = ind_range(A, AA)
% SUM_IND_RANGE  func = ind_range(A, AA)
%
%   Computes projection onto the range of A
%   x = A (A'*A)^{-1} A' * y
%
% A must be a sparse matrix and AA = A' * A must be dense!
%
    func = @(idx, count) { 'ind_range', idx, count, false, { A, AA } };

end
