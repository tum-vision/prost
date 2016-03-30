function [block] = sparse_kron_id(row, col, nrows, ncols, K, diaglength) 
        
    if (nrows ~= size(K, 1) * diaglength) || (ncols ~= size(K,2) * ...
                                              diaglength)
        
        error(['Block sparse_kron_id: size of variable doesnt fit ' ...
               'size of block!']);        
    end
        
    data = { K, diaglength };
    block = { 'sparse_kron_id', row, col, data };
end
