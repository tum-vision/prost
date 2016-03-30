function [block] = dense(row, col, nrows, ncols, K)    
    
    if (nrows ~= size(K, 1)) || (ncols ~= size(K,2))
        error('Dense block does not fit size of variables.');
    end
    
    data = { K };
    block = { 'dense', row, col, data };
end

