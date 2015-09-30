function [linop] = linop_identity(row, col, nrows, ncols)  
    data = { nrows, ncols, [1], [0] };
    linop = { 'diags', row, col, data };
end
