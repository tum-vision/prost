function [func] = sum_norm2(dim, interleaved, fun, a, b, c, d, e, alpha, beta)
% SUM_NORM2  func = sum_norm2(dim, interleaved, fun, a, b, c, d, e, alpha, beta)
%
%  Implements a separable sum of parametrized 2-norms, given by the
%  equation: h(|x|), where |.| denotes the dim-dimensional
%  Euclidean norm. h is parametrized exactly the same way as in
%  sum_1d. For more information, type 'help prost.function.sum_1d'.
%
%  If interleaved is set to true, the individual dim-dimensional
%  vectors inside the norm are assumed to be stored in an
%  interleaved manner, i.e., 
%  x_1, x_2, ..., x_dim, x_1, x_2, ..., x_dim, ... opposed to
%  x_1, x_1, ..., x_1, x_2, x_2, ..., x_2, ...
%
    
    switch nargin
      case 8
        alpha = 0;
        beta = 0;
        
      case 9
        beta = 0
    end
    
    func = @(idx, count) prost.prox.sum_norm2(idx, count / dim, dim, ...
                                              interleaved, fun, a, b, c, ...
                                              d, e, alpha, beta);

end
