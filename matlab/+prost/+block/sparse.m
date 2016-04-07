function [func] = sparse(K)    
% SPARSE  func = sparse(K)
%
% Implements a linear operator built of a sparse matrix K.
    
    sz = { size(K, 1), size(K, 2) };
    data = { K };
    
    func = @(row, col, nrows, ncols) { { 'sparse', row, col, data }, sz };
end
