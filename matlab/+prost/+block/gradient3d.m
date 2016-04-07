function [func] = gradient3d(nx, ny, L, label_first)  
% GRADIENT3D [func] = gradient3d(nx, ny, L, label_first)  
%
% Implements gradient operator with forward differences and
% Neumann boundary conditions.
    
    switch nargin
      case 3
        label_first = false;
    end
   
    sz = { nx*ny*L*3, nx*ny*L };
    data = { nx, ny, L, label_first  };
    func = @(row, col, nrows, ncols) { { 'gradient3d', row, col, data }, sz };

end
