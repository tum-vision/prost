function [block] = identity(row, col, n)  
    data = { n, n, 1, 0 };
    block = { 'diags', row, col, data };
end
