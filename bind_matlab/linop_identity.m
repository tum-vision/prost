function [linop] = linop_identity(row, col, n)  
    data = { n, n, 1, 0 };
    linop = { 'diags', row, col, data };
end
