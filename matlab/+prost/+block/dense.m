function [func] = dense(K)    
% DENSE  func = dense(K)
%
% Implements a linear operator built of a full matrix K.
    
    sz = { size(K, 1), size(K, 2) };
    data = { K };
    func = @(row, col, nrows, ncols) { { 'dense', row, col, data }, sz };
end
