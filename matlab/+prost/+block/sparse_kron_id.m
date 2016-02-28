function [block] = sparse_kron_id(row, col, K, diaglength)    
    data = { K, diaglength };
    block = { 'sparse_kron_id', row, col, data };
end
