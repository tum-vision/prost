function [linop] = linop_zero(row, col, nrows, ncols)    
    data = { nrows, ncols };
    linop = { 'zero', row, col, data };
end
