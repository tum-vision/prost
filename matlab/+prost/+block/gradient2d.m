function [func] = gradient2d(nx, ny, L, label_first)  
% GRADIENT2D [func] = gradient2d(nx, ny, L, label_first)  
%
% Implements gradient operator with forward differences and
% Neumann boundary conditions.
    
    switch nargin
      case 3
        label_first = false;
    end
    
    sz = { nx*ny*L*2, nx*ny*L };
    data = { nx, ny, L, label_first  };
    func = @(row, col, nrows, ncols) { { 'gradient2d', row, col, data }, sz };
end
