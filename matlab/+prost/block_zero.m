function [block] = block_zero(row, col, nrows, ncols)    
    data = { nrows, ncols };
    block = { 'zero', row, col, data };
end
