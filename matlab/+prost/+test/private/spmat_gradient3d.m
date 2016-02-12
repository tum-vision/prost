function [grad] = spmat_gradient3d(nx, ny, L)
%spmat_gradient2d Assembles linear operator for gradient
%   Input args are nx, ny, the dimension of the image and L, the number of labels.
%   Returns linear operator for gradient. Has the form (Dx1 | Dx2 |
%   ... | Dy1 | Dy2 | ... | Dz1 | Dz2 | ...)
%   Assumes Dirichlet boundary at z = L, otherwise Neumann

    dy = spdiags([[-ones(ny - 1, 1); 0], ones(ny, 1)], [0, 1], ny, ny);
    dy = kron(speye(nx), dy);
    
    dx = spdiags([[-ones(ny*(nx-1),1); zeros(ny, 1)], ones(nx*ny,1)], ...
                 [0, ny], nx*ny,nx*ny);

    dz = spdiags([-ones(ny*nx*L,1) ones(ny*nx*L,1)], ...
                 [0, ny*nx], nx*ny*L, nx*ny*L);
    
    grad = cat(1, ...
               kron(speye(L), dx), ...
               kron(speye(L), dy), ...
               dz);
end
