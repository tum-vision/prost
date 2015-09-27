function [linop] = linop_gradient3d(row, col, nx, ny, L)    
    data = { nx, ny, L };
    linop = { 'gradient_3d', row, col, data };
end
