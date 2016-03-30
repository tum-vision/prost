function [block] = sparse(row, col, nrows, ncols, K)    
    
    if (nrows ~= size(K, 1)) || (ncols ~= size(K, 2))
        error('Block sparse does not fit size of variable');
    end
    
    data = { K };
    block = { 'sparse', row, col, data };
end

