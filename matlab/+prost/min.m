function [problem] = min(primal_vars, primal_constrained_vars)
% MIN  Creates a linearly constrained primal problem from the specified variables
   
    problem.type = 'min';
    problem.primal_vars = primal_vars;
    problem.primal_constrained_vars = primal_constrained_vars;
    problem.num_primals = prod(size(primal_vars));
    problem.num_constrained_primals = prod(size(primal_constrained_vars));
       
    primal_idx = 0;
    for i=1:problem.num_primals
        primal_vars{i}.idx = primal_idx;
        primal_idx = primal_idx + primal_vars{i}.dim;
    end
    
    primal_constrained_idx = 0;
    for i=1:problem.num_constrained_primals
        primal_constrained_vars{i}.idx = primal_constrained_idx;
        primal_constrained_idx = primal_constrained_idx + primal_constrained_vars{i}.dim;
    end

    problem.data.linop = {};
    problem.data.prox_g = {};
    problem.data.prox_f = {};
    problem.data.prox_gstar = {};
    problem.data.prox_fstar = {};

    for i=1:problem.num_primals
        problem.data.prox_g{i} = primal_vars{i}.fun(...
            primal_vars{i}.idx, primal_vars{i}.dim);
        
        if ~isempty(primal_vars{i}.linop)
            num_pairs = prod(size(primal_vars{i}.pairing));
            for j=1:num_pairs
                problem.data.linop{end + 1} = primal_vars{i}.linop{j}(...
                    primal_vars{i}.pairing{j}.idx, ...
                    primal_vars{i}.idx, ...
                    primal_vars{i}.pairing{j}.dim, ...
                    primal_vars{i}.dim);
            end
        end
    end

    for i=1:problem.num_constrained_primals
        problem.data.prox_f{i} = primal_constrained_vars{i}.fun(...
            primal_constrained_vars{i}.idx, primal_constrained_vars{i}.dim);
    end

    problem.data.scaling = 'alpha';
    problem.data.scaling_alpha = 1;
    problem.data.scaling_left = 1;
    problem.data.scaling_right = 1;
    
end
