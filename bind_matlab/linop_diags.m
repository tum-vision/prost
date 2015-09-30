function [linop] = linop_diags(row, col, nrows, ncols, factors, offsets)  
    data = { nrows, ncols, factors, offsets };
    linop = { 'diags', row, col, data };
end
