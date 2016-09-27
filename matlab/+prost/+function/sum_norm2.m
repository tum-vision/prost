function [func] = sum_norm2(dim, interleaved, fun, a, b, c, d, e, alpha, beta)
% SUM_NORM2  func = sum_norm2(dim, interleaved, fun, a, b, c, d, e, alpha, beta)
%
%  Implements a separable sum of parametrized 2-norms, given by the
%  equation: h(|x|), where |.| denotes the dim-dimensional
%  Euclidean norm and h : R^dim -> R^+. h is parametrized as:
%
%  h(|x|) = c * f_{alpha,beta}(a|x| - b) + d|x| + 0.5 e|x|^2. 
%
%  fun is a string describing the 1d function and can be chosen as
%  one of the following:
%
%  'abs'         f(z) = |z|
%  'huber'       f(z) = z^2 / (2 alpha),     if |z| <= alpha
%                       |z| - (alpha / 2),   if |z|  > alpha. 
%  'ind_box01'   f(z) = I(0 <= z <= 1)   
%  'ind_leq0'    f(z) = I(z <= 0)
%  'ind_geq0'    f(z) = I(z >= 0)
%  'ind_eq0'     f(z) = I(z = 0)
%  'l0'          f(z) = #nonzero(z)
%  'lq'          f(z) = |z|^alpha, alpha >= 0
%  'lq_plus_eps' f(z) = (|z|+beta)^alpha, alpha >= 0, beta >= 0
%  'max_pos0'    f(z) = max(0, z)
%  'square'      f(z) = (1/2) z^2
%  'trunclin'    f(z) = min(alpha |z|, beta)
%  'truncquad'   f(z) = min(alpha z^2, beta)
%  'zero'        f(z) = 0
%
%  If interleaved is set to true, the individual dim-dimensional
%  vectors inside the norm are assumed to be stored in an
%  interleaved manner, i.e., 
%  x_1, x_2, ..., x_dim, x_1, x_2, ..., x_dim, ... opposed to
%  x_1, x_1, ..., x_1, x_2, x_2, ..., x_2, ...
%
%  Examples:
%   - prost.function.sum_norm2(2, true, 'abs', lmb): \sum_i lmb ||g_i||_2 
%   - prost.function.sum_norm2(2, true, 'ind_leq0', 1, 1, 1): \sum_i I(||g_i||_2 <= 1)

    switch nargin
      case 3
        a = 1;
        b = 0;
        c = 1;
        d = 0;
        e = 0;
        alpha = 0;
        beta = 0;
        
      case 4
        b = 0;
        c = 1;
        d = 0;
        e = 0;
        alpha = 0;
        beta = 0;
        
      case 5
        c = 1;
        d = 0;
        e = 0;
        alpha = 0;
        beta = 0;

      case 6
        d = 0;
        e = 0;
        alpha = 0;
        beta = 0;
      
      case 7
        e = 0;
        alpha = 0;
        beta = 0;
      
      case 8
        alpha = 0;
        beta = 0;
        
      case 9
        beta = 0;
    end
    
    coeffs = { a, b, c, d, e, alpha, beta };
    
    func = @(idx, count) { strcat('elem_operation:norm2:', fun), ...
                        idx, count, false, { count / dim, dim, interleaved, coeffs } };

end
