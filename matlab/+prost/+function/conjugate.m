function [func] = conjugate(fun)
% CONJUGATE  func = conjugate(fun)
%
%   Returns the convex conjugate of the input function. Implemented
%   via Moreau's identity.

    func = @(idx, count) prost.prox.moreau( fun(idx, count) );
    
end
