function [linop] = linop_gradient3d(row, col, nx, ny, L, ...
                                    label_first)
    
    switch nargin
      case 5
        label_first = false;
    end
    
    data = { nx, ny, L, label_first };
    linop = { 'gradient_3d', row, col, data };
end
