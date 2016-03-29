function [func] = sum_1d(fun, a, b, c, d, e, alpha, beta)
% SUM_1D  func = sum_1d(fun,a,b,c,d,e,alpha,beta). 
%
%   Implements a separable sum of 1D functions, parametrized by
%   the equation: c * f_{alpha,beta}(ax - b) + dx + 0.5 ex^2. 
%   Depending on fun, alpha and beta parametrize the function f.
%
%   The parameters (a, b, c, d, e, alpha, beta) can be scalars or
%   vectors of the size of the variable. 
%
%   fun is a string describing the 1d function and can be chosen as
%   one of the following:
%
%   'abs'       f(z) = |z|
%   'huber'     f(z) = z^2 / (2 alpha),     if |z| <= alpha
%                      |z| - (alpha / 2),   if |z|  > alpha. 
%   'ind_box01' f(z) = I(0 <= z <= 1)   
%   'ind_leq0'  f(z) = I(z <= 0)
%   'ind_geq0'  f(z) = I(z >= 0)
%   'ind_eq0'   f(z) = I(z = 0)
%   'l0'        f(z) = #nonzero(z)
%   'max_pos0'  f(z) = max(0, z)
%   'square'    f(z) = (1/2) z^2
%   'zero'      f(z) = 0
    
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
