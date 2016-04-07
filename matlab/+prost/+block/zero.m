function [func] = zero()
% ZERO [func] = zero()
%
% Implements zero linear operator.
    
    func = @(row, col, nrows, ncols) ...
           { { 'zero', row, col, {nrows, ncols} }, {nrows, ncols} };
    
end
