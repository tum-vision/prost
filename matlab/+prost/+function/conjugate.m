function [func] = conjugate(fun)
% CONJUGATE  func = conjugate(fun)
%
%   Returns the convex conjugate of the input function. Implemented
%   via Moreau's identity.

    data = { child };   
    func = @(idx, count) { 'moreau', 0, 0, false, data }; 
    
end
