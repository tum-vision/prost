function [result, rowsum, colsum] = eval_linop(linop, input, transpose)
    
    [result, rowsum, colsum] = prost_('eval_linop', linop, input, transpose);
    
end
