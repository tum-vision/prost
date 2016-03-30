function [block] = gradient2d(row, col, nrows, ncols, nx, ny, L, label_first)  
    if (nrows ~= nx*ny*L*2) || (ncols ~= nx*ny*L)
        error('Block gradient2d does not fit size of variable.');
    end
    
    data = { nx, ny, L, label_first  };
    block = { 'gradient2d', row, col, data };
end
