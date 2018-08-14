function [func] = permute(fun, perm)
% PERMUTE  func = permute(fun, perm)
%
%   Returns the function composition with a permutation given by perm.
%   Note that the permutation indices should be local, i.e. 
%   starting from zero and going to size - 1.

    func = @(idx, count) prox_permute( fun(idx, count), perm );
    
end

function [prox] = prox_permute(child, perm)

    data = { child, perm };
    prox = { 'permute', child{2}, child{3}, child{4}, data };

end
