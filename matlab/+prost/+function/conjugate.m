function [func] = conjugate(fun)
% CONJUGATE  func = conjugate(fun)
%
%   Returns the convex conjugate of the input function. Implemented
%   via Moreau's identity.

    func = @(idx, count) prox_moreau( fun(idx, count) );
    
end

function [prox] = prox_moreau(child)

    data = { child };
    prox = { 'moreau', child{2}, child{3}, child{4}, data };

end
