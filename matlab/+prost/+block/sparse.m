function [block] = sparse(row, col, K)    
    data = { K };
    block = { 'sparse', row, col, data };
end

