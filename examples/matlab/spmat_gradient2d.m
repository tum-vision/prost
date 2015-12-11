function [grad] = spmat_gradient2d(nx, ny, L)
%spmat_gradient2d Assembles linear operator for gradient
%   Input args are nx, ny, the dimension of the image and L, the number of labels.
%   Returns linear operator for gradient. Has the form (Dx1 | Dx2 |
%   ... | Dy1 | Dy2 | ...)

    dy = spdiags([[-ones(ny - 1, 1); 0], ones(ny, 1)], [0, 1], ny, ny);
    dy = kron(speye(nx), dy);
    
    dx = spdiags([[-ones(ny*(nx-1),1); zeros(ny, 1)], ones(nx*ny,1)], ...
                 [0, ny], nx*ny,nx*ny);
    
    grad = cat(1, kron(speye(L), dx), kron(speye(L), dy));
end

