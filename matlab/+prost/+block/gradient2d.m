function [block] = gradient2d(row, col, nx, ny, L, label_first)  
    switch nargin
      case 5
        label_first = false;
    end

    data = { nx, ny, L, label_first  };
    block = { 'gradient2d', row, col, data };
end
