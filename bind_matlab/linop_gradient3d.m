function [linop] = linop_gradient3d(row, col, nx, ny, L, ...
                                    label_first, hx, hy, ht)
    
    switch nargin
      case 5
        label_first = false;
        hx = 1;
        hy = 1;
        ht = 1;
        
      case 6
        hx = 1;
        hy = 1;
        ht = 1;
    end
    
    data = { nx, ny, L, label_first, hx, hy, ht };
    linop = { 'gradient_3d', row, col, data };
end
