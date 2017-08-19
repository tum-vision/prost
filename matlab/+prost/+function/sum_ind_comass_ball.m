function [func] = sum_ind_comass_ball(n, interleaved)
% SUM_IND_COMASS_BALL  func = sum_ind_comass_ball(n, interleaved)
%
% Indicator function of the unit ball of the comass norm of a 
% 2-vector in R^n, currently only n \in {4, 5} supported.

if n == 4
    dim = 6;
    func = @(idx, count) { 'elem_operation:ind_comass4_ball', idx, count, false, { count / dim, dim, interleaved } };
    
elseif n == 5
    dim = 10;
    func = @(idx, count) { 'elem_operation:ind_comass5_ball', idx, count, false, { count / dim, dim, interleaved } };
else
    error('Indicator of comass norm ball not implemented for n \notin {4, 5}');
end

end
