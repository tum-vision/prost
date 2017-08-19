function [func] = sum_mass_norm(n, interleaved)
% SUM_MASS_NORM  func = sum_mass_norm(n, interleaved)
%
% Mass norm of a 2-vector in R^n, currently only n \in {4, 5} supported.

if n == 4
    dim = 6;
    func = @(idx, count) { 'elem_operation:mass4', idx, count, false, { count / dim, dim, interleaved } };
    
elseif n == 5
    dim = 10;
    func = @(idx, count) { 'elem_operation:mass5', idx, count, false, { count / dim, dim, interleaved } };
else
    error('Mass norm not implemented for n \notin {4, 5}');
end

end
