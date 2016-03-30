function [linop] = gradient3d(nx, ny, L, label_first)
% GRADIENT3D  linop = gradient3d(nx, ny, L, label_first)
%
%   Implements a gradient linear operator for a 3d image with 
%   L channels/labels. If label_first is set to true, then the
%   image is stored in an "interleaved" fashion, i.e. for RGB 
%   images r,g,b,r,g,b,... The image of the linear operator is
%   always non-interleaved.
        
    switch nargin
      case 3
        label_first = false;
    end
    
    linop = @(row, col, nrows, ncols) prost.block.gradient3d(row, col, ...
                                                      nrows, ncols, nx, ny, L, ...
                                                      label_first);
    
end
