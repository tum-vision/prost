function [linop] = linop_identity(row, col, nrows, ncols, factors, offsets)  
    data = { nrows, ncols, factors, offsets };
    linop = { 'identity', row, col, data };
end
