function [func] = zero()
% ZERO  func = zero()
%
%   Zero function.
    
    func = @(idx, count) prost.prox.zero(idx, count);
end
