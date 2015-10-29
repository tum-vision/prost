function [linop] = linop_data_graph_prec(row, col, nx, ny, L, left, right)  
    data = { nx, ny, L, left, right };
    linop = { 'data_graph_prec', row, col, data };
end
