%
% gradforw2d.m
%
% Builds a finite-difference discretization of the 2d gradient operator using
% forward-differences with Neumann boundary conditions as a sparse matrix. 
%
% nx: width of the image
% ny: height of the image
% nc: number of channels
%

function K = gradForw2D(nx, ny, nc)
    dy = spdiags([[-ones(ny - 1, 1); 0], ones(ny, 1)], [0, 1], ny, ny);
    dy = kron(speye(nx), dy);
    
    dx = spdiags([[-ones(ny*(nx-1),1); zeros(ny, 1)], ones(nx*ny,1)], [0, ny], nx*ny,nx*ny);
    K = kron(speye(nc), [dx; dy]);
end
