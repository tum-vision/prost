function [func] = identity(scal)  
% IDENTITY func = identity(scal)
%
% Implements identity matrix, optionally scaled by factor scal
    
    switch nargin
      case 0
        scal = 1;
    end
    
    func = @(row, col, nrows, ncols) { { 'diags', row, col, { nrows, ...
                        ncols, scal, 0 } }, { nrows, ncols } };
end
