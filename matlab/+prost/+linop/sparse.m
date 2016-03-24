function [linop] = sparse(K)    
    linop = @(row, col) prost.block.sparse(row, col, K);
end
