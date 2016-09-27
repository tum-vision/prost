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
%   'abs'         f(z) = |z|
%   'huber'       f(z) = z^2 / (2 alpha),     if |z| <= alpha
%                        |z| - (alpha / 2),   if |z|  > alpha. 
%   'ind_box01'   f(z) = I(0 <= z <= 1)   
%   'ind_leq0'    f(z) = I(z <= 0)
%   'ind_geq0'    f(z) = I(z >= 0)
%   'ind_eq0'     f(z) = I(z = 0)
%   'l0'          f(z) = #nonzero(z)
%   'lq'          f(z) = |z|^alpha, alpha >= 0
%   'lq_plus_eps' f(z) = (|z|+beta)^alpha, alpha >= 0, beta >= 0
%   'max_pos0'    f(z) = max(0, z)
%   'square'      f(z) = (1/2) z^2
%   'trunclin'    f(z) = min(alpha |z|, beta)
%   'truncquad'   f(z) = min(alpha z^2, beta)
%   'zero'        f(z) = 0
%
%   Examples:
%   - prost.function.sum_1d('square', 1, f): (1/2) |u-f|^2
%   - prost.function.sum_1d('abs', lmb, f): lmb |u-f|
%   - prost.function.sum_1d('ind_geq0', 1, 0, 1, f): I(u>=0) + <u, f>
    
    switch nargin
      case 1
        a = 1;
        b = 0;
        c = 1;
        d = 0;
        e = 0;
        alpha = 0;
        beta = 0;
        
      case 2
        b = 0;
        c = 1;
        d = 0;
        e = 0;
        alpha = 0;
        beta = 0;
        
      case 3
        c = 1;
        d = 0;
        e = 0;
        alpha = 0;
        beta = 0;

      case 4
        d = 0;
        e = 0;
        alpha = 0;
        beta = 0;
      
      case 5
        e = 0;
        alpha = 0;
        beta = 0;
        
      case 6
        alpha = 0;
        beta = 0;
        
      case 7
        beta = 0;
    end

    func = @(idx, count) { strcat('elem_operation:1d:', fun), idx, count, true, { ...
        count, 1, false, { a, b, c, d, e, alpha, beta } } };
    
end
