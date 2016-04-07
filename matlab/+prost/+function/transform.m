function [func] = transform(fun, a, b, c, d, e)
% TRANSFORM  func = transform(fun, a, b, c, d, e)
%
%  Returns the tranformed function c * f(ax - b) + dx + (e/2) x^2,
%  where f is any other function. 
%
%  For example, the following two functions are the same:
%
%  - prost.function.sum_1d('square', 1, f, lmb)
%  - prost.function.transform(prost.function.sum_1d('square'), 1, f, lmb)
%  
%  and both represent f(x) = (lmb / 2) |x-f|^2.

    switch nargin
      case 1
        a = 1;
        b = 0;
        c = 1;
        d = 0;
        e = 0;

      case 2
        b = 0;
        c = 1;
        d = 0;
        e = 0;

      case 3
        c = 1;
        d = 0;
        e = 0;

      case 4
        d = 0;
        e = 0;

      case 5
        e = 0;
    end
        
        
    func = @(idx, count) prox_transform( fun(idx, count), a, ...
                                         b, c, d, e );
end

function [prox] = prox_transform(child, a, b, c, d, e)
  
    data = { a, b, c, d, e, child };
    prox = { 'transform', child{2}, child{3}, child{4}, data };

end
