function [func] = zero()

    func = @(idx, count) { 'zero', idx, count, true, { } };

end
