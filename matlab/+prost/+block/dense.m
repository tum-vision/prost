function [block] = dense(row, col, K)    
    data = { K };
    block = { 'dense', row, col, data };
end

