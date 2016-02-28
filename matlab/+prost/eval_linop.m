function [result, rowsum, colsum, time] = eval_linop(linop, input, transpose)
    
    [result, rowsum, colsum, time] = prost_('eval_linop', linop, input, transpose);
    
end
