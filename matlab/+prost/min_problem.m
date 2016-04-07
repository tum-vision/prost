classdef min_problem < prost.problem
    properties
        num_primal_vars
        num_constrained_vars
        
        primal_vars 
        constrained_vars
    end
    
    methods
        function obj = min_problem(primals, constraineds)
            obj = obj@prost.problem();
            
            obj.num_primal_vars = prod(size(primals));
            obj.num_constrained_vars = prod(size(constraineds));

            obj.primal_vars = primals;
            obj.constrained_vars = constraineds;
           
            % initialize indices of primal variables and set prox_g to zero
            primal_idx = 0;
            for i=1:obj.num_primal_vars
                obj.primal_vars{i}.idx = primal_idx;
                
                num_sub_vars = prod(size(primals{i}.sub_vars));
                sub_var_idx = 0;
                for j=1:num_sub_vars
                    obj.primal_vars{i}.sub_vars{j}.idx = primal_idx + sub_var_idx;
                    
                    sub_var_idx = sub_var_idx + primals{i}.sub_vars{j}.dim;
                end
                
                if (sub_var_idx ~= primals{i}.dim) && (num_sub_vars ...
                                                       > 0)
                    error(['Size of subvariables does not match size ' ...
                           'of parent variable.']);
                end
                
                primal_idx = primal_idx + primals{i}.dim;
            end
        
            % initialize indices of constrained variables and set
            % prox_f to zero
            constrained_idx = 0;
            for i=1:obj.num_constrained_vars
                obj.constrained_vars{i}.idx = constrained_idx;
                
                num_sub_vars = prod(size(constraineds{i}.sub_vars));
                sub_var_idx = 0;
                for j=1:num_sub_vars
                    obj.constrained_vars{i}.sub_vars{j}.idx = constrained_idx + sub_var_idx;
                    
                    sub_var_idx = sub_var_idx + constraineds{i}.sub_vars{j}.dim;
                end
                
                if (sub_var_idx ~= constraineds{i}.dim) && (num_sub_vars ...
                                                     > 0)
                    error(['Size of subvariables does not match size ' ...
                           'of parent variable.']);
                end

                constrained_idx = constrained_idx + constraineds{i}.dim;
            end
            
            obj.nrows = constrained_idx;
            obj.ncols = primal_idx;            
        end
        
        function obj = add_function(obj, var, func)
            for i=1:obj.num_primal_vars
                num_sub_vars = prod(size(obj.primal_vars{i}.sub_vars));
                for j=1:num_sub_vars
                    if obj.primal_vars{i}.sub_vars{j} == var
                        obj.data.prox_g{end + 1} = ...
                            func(obj.primal_vars{i}.sub_vars{j}.idx, obj.primal_vars{i}.sub_vars{j}.dim);
                        return;
                    end
                end
                
                if obj.primal_vars{i} == var
                    obj.data.prox_g{end + 1} = ...
                        func(obj.primal_vars{i}.idx, obj.primal_vars{i}.dim);
                    return;
                end
            end
            
            for i=1:obj.num_constrained_vars
                num_sub_vars = prod(size(obj.constrained_vars{i}.sub_vars));
                for j=1:num_sub_vars
                    if obj.constrained_vars{i}.sub_vars{j} == var
                        obj.data.prox_f{end + 1} = ...
                            func(obj.constrained_vars{i}.sub_vars{j}.idx, obj.constrained_vars{i}.sub_vars{j}.dim);
                        return;
                    end
                end
                
                if obj.constrained_vars{i} == var
                    obj.data.prox_f{end + 1} = ...
                        func(obj.constrained_vars{i}.idx, obj.constrained_vars{i}.dim);
                    return;
                end
            end
            
            error('Variable not registered in problem!');
        end
        
        function obj = add_constraint(obj, pv, dv, block)
            row = -1;
            col = -1;
            constrained_dim = -1;
            primal_dim = -1;
                       
            % find primal variable and set column
            for i=1:obj.num_primal_vars
                num_sub_vars = prod(size(obj.primal_vars{i}.sub_vars));
                for j=1:num_sub_vars
                    if obj.primal_vars{i}.sub_vars{j} == pv
                        col = obj.primal_vars{i}.sub_vars{j}.idx;
                        primal_dim = obj.primal_vars{i}.sub_vars{j}.dim;
                    end
                end
                
                if obj.primal_vars{i} == pv
                    col = obj.primal_vars{i}.idx;
                    primal_dim = obj.primal_vars{i}.dim;
                end
            end

            % find constrained variable and set row
            for i=1:obj.num_constrained_vars
                num_sub_vars = prod(size(obj.constrained_vars{i}.sub_vars));
                for j=1:num_sub_vars
                    if obj.constrained_vars{i}.sub_vars{j} == dv
                        row = obj.constrained_vars{i}.sub_vars{j}.idx;
                        constrained_dim = obj.constrained_vars{i}.dim;
                    end
                end
                
                if obj.constrained_vars{i} == dv
                    row = obj.constrained_vars{i}.idx;
                    constrained_dim = obj.constrained_vars{i}.dim;
                end
            end
            
            if (row == -1) || (col == -1)
                error('Variable pair not registered in problem.');
            end
            
            nrows = constrained_dim;
            ncols = primal_dim;
            
            block_size_pair = block(row, col, nrows, ncols);
            
            obj.data.linop{end + 1} = block_size_pair{1};
            
            sz = block_size_pair{2};
            if (sz{1} ~= constrained_dim) || (sz{2} ~= primal_dim)
                error(['Size of block does not fit size of primal/constrained ' ...
                       'variable.']);
            end
        end
        
        function obj = fill_variables(obj, result)
            for i=1:obj.num_primal_vars
                idx = obj.primal_vars{i}.idx;
                obj.primal_vars{i}.val = result.x(idx+1:idx+ ...
                                                  obj.primal_vars{i}.dim);
                
                num_sub_vars = prod(size(obj.primal_vars{i}.sub_vars));
                for j=1:num_sub_vars
                    abs_idx = obj.primal_vars{i}.sub_vars{j}.idx - obj.primal_vars{i}.idx;
                    obj.primal_vars{i}.sub_vars{j}.val = ...
                        obj.primal_vars{i}.val(abs_idx+1:abs_idx+obj.primal_vars{i}.sub_vars{j}.dim);
                end
            end

            for i=1:obj.num_constrained_vars
                idx = obj.constrained_vars{i}.idx;
                obj.constrained_vars{i}.val = result.z(idx+1:idx+ ...
                                                obj.primal_vars{i}.dim);
                
                num_sub_vars = prod(size(obj.constrained_vars{i}.sub_vars));
                for j=1:num_sub_vars
                    abs_idx = obj.constrained_vars{i}.sub_vars{j}.idx - obj.constrained_vars{i}.idx;
                    obj.constrained_vars{i}.sub_vars{j}.val = ...
                        obj.constrained_vars{i}.val(abs_idx+1:abs_idx+obj.constrained_vars{i}.sub_vars{j}.dim);
                end
            end
        end
    end
end
