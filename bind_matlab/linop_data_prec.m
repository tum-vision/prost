function [linop] = linop_data_prec(row, col, nx, ny, L, left, right)  
    data = { nx, ny, L, left, right };
    linop = { 'data_prec', row, col, data };
end
