function [block] = diags(row, col, nrows, ncols, factors, offsets)  
    data = { nrows, ncols, factors, offsets };
    block = { 'diags', row, col, data };
end
