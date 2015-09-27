function [linop] = linop_sparse(row, col, K)    
    data = { K };
    linop = { 'sparse', row, col, data };
end
