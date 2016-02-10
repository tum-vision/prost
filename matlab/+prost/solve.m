function [result] = solve(prob, backend, opts)
    
    result = prost_('solve_problem', prob, backend, opts);
    
end
