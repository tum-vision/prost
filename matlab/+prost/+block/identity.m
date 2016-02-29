function [block] = identity(row, col, n, scal)  
% scaled identity matrix scal * I of size n*n starting at row,col.
    if nargin < 4
        scal = 1;
    end
    
    data = { n, n, scal, 0 };
    block = { 'diags', row, col, data };
end
