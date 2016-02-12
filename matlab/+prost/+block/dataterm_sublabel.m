function [block] = dataterm_sublabel(row, col, nx, ny, L, left, right)
    
    data = { nx, ny, L, left, right };
    block = { 'dataterm_sublabel', row, col, data }; 
    
end
