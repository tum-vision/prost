function [problem] = min_max(primal_vars, dual_vars)
% MIN_MAX  Creates a saddle point problem from the specified primal and dual
%          variables.
   
    problem.type = 'min_max';
    problem.primal_vars = primal_vars;
    problem.dual_vars = dual_vars;
    problem.num_primals = prod(size(primal_vars));
    problem.num_duals = prod(size(dual_vars));
       
    primal_idx = 0;
    for i=1:problem.num_primals
        primal_vars{i}.idx = primal_idx;
        primal_idx = primal_idx + primal_vars{i}.dim;
    end
    
    dual_idx = 0;
    for i=1:problem.num_duals
        dual_vars{i}.idx = dual_idx;
        dual_idx = dual_idx + dual_vars{i}.dim;
    end

    problem.data.linop = {};
    problem.data.prox_g = {};
    problem.data.prox_fstar = {};
    problem.data.prox_gstar = {};
    problem.data.prox_f = {};

    for i=1:problem.num_primals
        problem.data.prox_g{i} = primal_vars{i}.fun(...
            primal_vars{i}.idx, primal_vars{i}.dim);
        
        if ~isempty(primal_vars{i}.linop)
            problem.data.linop{end + 1} = primal_vars{i}.linop(...
                primal_vars{i}.pairing.idx, ...
                primal_vars{i}.idx);
        end
    end

    for i=1:problem.num_duals
        problem.data.prox_fstar{i} = dual_vars{i}.fun(...
            dual_vars{i}.idx, dual_vars{i}.dim);
    end

    problem.data.scaling = 'alpha';
    problem.data.scaling_alpha = 1;
    problem.data.scaling_left = 1;
    problem.data.scaling_right = 1;
    
end
