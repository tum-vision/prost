function [block] = gradient3d(row, col, nrows, ncols, nx, ny, L, label_first)  
    if (nrows ~= nx*ny*L*3) || (ncols ~= nx*ny*L)
        error('Block gradient3d does not fit size of variable.');
    end

    data = { nx, ny, L, label_first  };
    block = { 'gradient3d', row, col, data };
end
