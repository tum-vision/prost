function [func] = permute(fun, perm)
% PERMUTE  func = permute(fun, perm)
%
%   Returns the function composition with a permutation matrix.

    func = @(idx, count) prox_permute( fun(idx, count), perm );
    
end

function [prox] = prox_permute(child, perm)

    data = { child, perm };
    prox = { 'permute', child{2}, child{3}, child{4}, data };

end
