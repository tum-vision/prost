function [func] = sum_1d(fun, a, b, c, d, e, alpha, beta)
% general class of functions
% sum_{i=1}^{count} c*f(ax - b) + dx + 0.5ex^2 
% fn is a string describing the 1d function f
% alpha and beta are parameters which further parametrize
% the function f.
    
    switch nargin
      case 6
        alpha = 0;
        beta = 0;
        
      case 7
        beta = 0;
    end

    func = @(idx, count) prost.prox.sum_1d(idx, count, fun, a, b, c, d, e, ...
                                           alpha, beta);

end

