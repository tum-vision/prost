function [linop] = linop_gradient2d(row, col, nx, ny, L)    
    data = { nx, ny, L };
    linop = { 'gradient_2d', row, col, data };
end

