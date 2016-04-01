function [problem] = min_max(primal_vars, dual_vars)
% MIN_MAX  Creates a saddle point problem from the specified primal and dual
%          variables.
   
    problem.type = 'min_max';
    problem.primal_vars = primal_vars;
    problem.dual_vars = dual_vars;
    problem.num_primals = prod(size(primal_vars));
    problem.num_duals = prod(size(dual_vars));
       
    % compute primal indices
    primal_idx = 0;
    for i=1:problem.num_primals
        primal_vars{i}.idx = primal_idx;
        
        sub_idx = primal_idx;
        num_subvars = prod(size(primal_vars{i}.subvars));
        for j=1:num_subvars
            primal_vars{i}.subvars{j}.idx = sub_idx;
            sub_idx = sub_idx + primal_vars{i}.subvars{j}.dim;
        end
        
        primal_idx = primal_idx + primal_vars{i}.dim;
    end
    
    % compute dual indices
    dual_idx = 0;
    for i=1:problem.num_duals
        dual_vars{i}.idx = dual_idx;

        sub_idx = dual_idx;
        num_subvars = prod(size(dual_vars{i}.subvars));
        for j=1:num_subvars
            dual_vars{i}.subvars{j}.idx = sub_idx;
            sub_idx = sub_idx + dual_vars{i}.subvars{j}.dim;
        end
        
        dual_idx = dual_idx + dual_vars{i}.dim;
    end

    problem.data.linop = {};
    problem.data.prox_g = {};
    problem.data.prox_fstar = {};
    problem.data.prox_gstar = {};
    problem.data.prox_f = {};

    for i=1:problem.num_primals
       
        % add primal prox
        if ~isempty(primal_vars{i}.fun)
            problem.data.prox_g{end + 1} = primal_vars{i}.fun(...
                primal_vars{i}.idx, primal_vars{i}.dim);
        end

        num_subvars = prod(size(primal_vars{i}.subvars));
        for j=1:num_subvars
            if ~isempty(primal_vars{i}.subvars{j}.fun)
                problem.data.prox_g{end + 1} = primal_vars{i}.subvars{j}.fun(...
                    primal_vars{i}.subvars{j}.idx, primal_vars{i}.subvars{j}.dim);
            end           
        end
        
        % add linops
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
        
        for j=1:num_subvars
            if ~isempty(primal_vars{i}.subvars{j}.linop)
                num_pairs = prod(size(primal_vars{i}.subvars{j}.pairing));
                for k=1:num_pairs
                    problem.data.linop{end + 1} = primal_vars{i}.subvars{j}.linop{k}(...
                        primal_vars{i}.subvars{j}.pairing{k}.idx, ...
                        primal_vars{i}.idx, ...
                        primal_vars{i}.subvars{j}.pairing{k}.dim, ...
                        primal_vars{i}.dim);
                end
            end
        end
    end

    for i=1:problem.num_duals
        num_subvars = prod(size(dual_vars{i}.subvars));

        if ~isempty(dual_vars{i}.fun)
            problem.data.prox_fstar{end + 1} = dual_vars{i}.fun(...
                dual_vars{i}.idx, dual_vars{i}.dim);
        end

        for j=1:num_subvars
            if ~isempty(dual_vars{i}.subvars{j}.fun)
                problem.data.prox_fstar{end + 1} = dual_vars{i}.subvars{j}.fun(...
                    dual_vars{i}.subvars{j}.idx, dual_vars{i}.subvars{j}.dim);
            end
        end
    end

    problem.data.scaling = 'alpha';
    problem.data.scaling_alpha = 1;
    problem.data.scaling_left = 1;
    problem.data.scaling_right = 1;
    
end
