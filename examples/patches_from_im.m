function [C, V] = patches_from_im(IL, IR, dim)
%PATCHES_FROM_IM Creates patches and returns for each pixel in the patch a
%separate channel that is stored in a cell array.
%   Returns C, V the cell arrays of channels corresponding to IL and IR
%   respectively. Each cell in the arrays corresponds to one pixel in the
%   patch.
%   Input arguments are IL, the left stereo image,
%   IR the right stereo image,
%   dim, the dimension of the patch as a 4D vector: dim(1), dim(2) is the
%   patch size in x-direction, dim(3), dim(4) the patch size in
%   y-direction.

    [~, ~, d] = size(IL);

    
    C = cell(1, d * (dim(2) + dim(1) + 1) * (dim(4) + dim(3) + 1));
    V = cell(1, d * (dim(2) + dim(1) + 1) * (dim(4) + dim(3) + 1));
    
    idx = 1;
    for u=1:d
        for i=-dim(1):dim(2)
            if(i < 0)
                I1x = [IL(:, 1:-i, d) IL(:, 1:end+i, d)];
                I2x = [IR(:, 1:-i, d) IR(:, 1:end+i, d)];
            else
                I1x = [IL(:, i+1:end, d) fliplr(IL(:, end-i+1:end, d))];
                I2x = [IR(:, i+1:end, d) fliplr(IR(:, end-i+1:end, d))];
            end
            
            for j=-dim(3):dim(4)
                if(j < 0)
                    C{idx} = [I1x(1:-j, :); I1x(1:end+j, :)];
                    V{idx} = [I2x(1:-j, :); I2x(1:end+j, :)];
                else
                    C{idx} = [I1x(j+1:end, :); flipud(I1x(end-j+1:end, :))];
                    V{idx} = [I2x(j+1:end, :); flipud(I2x(end-j+1:end, :))];
                end
                
                idx = idx+1;
            end
        end
    end
    
end

