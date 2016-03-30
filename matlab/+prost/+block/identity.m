function [block] = identity(row, col, nrows, ncols, scal)  
    if nrows ~= ncols
        error('Identity block is not rectangular!');
    end
    
    n = nrows;
    
    data = { n, n, scal, 0 };
    block = { 'diags', row, col, data };
end
