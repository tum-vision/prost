function [func] = transform(fun, a, b, c, d, e)
% TRANSFORM  func = transform(fun, a, b, c, d, e)
%
%  Returns the tranformed function c * f(ax - b) + dx + (e/2) x^2.
   
    func = @(idx, count) prost.prox.transform( fun(idx, count), a, ...
                                               b, c, d, e );
end
