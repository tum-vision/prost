function [block] = zero(row, col, nrows, ncols)    
    data = { nrows, ncols };
    block = { 'zero', row, col, data };
end
