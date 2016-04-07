function [result] = solve(prob, backend, opts)
% SOLVE  Solves the problem prob using the specified backend and
%        options. 
    result = prost_('solve_problem', prob.data, backend, opts);
    prob.fill_variables( result );
    
    % % read back result
    % if strcmp(prob.type, 'min_max')
    %     for i=1:prob.num_primals
    %         idx = prob.primal_vars{i}.idx;
    %         prob.primal_vars{i}.val = result.x(idx+1:idx+prob.primal_vars{i}.dim);
    %     end
        
    %     for i=1:prob.num_duals
    %         idx = prob.dual_vars{i}.idx;
    %         prob.dual_vars{i}.val = result.y(idx+1:idx+prob.dual_vars{i}.dim);
    %     end
    % elseif strcmp(prob.type, 'min')
    %     for i=1:prob.num_primals
    %         idx = prob.primal_vars{i}.idx;
    %         prob.primal_vars{i}.val = result.x(idx+1:idx+prob.primal_vars{i}.dim);
    %     end
        
    %     for i=1:prob.num_constrained_primals
    %         idx = prob.primal_constrained_vars{i}.idx;
    %         prob.primal_constrained_vars{i}.val = result.z(idx+1:idx+prob.primal_constrained_vars{i}.dim);
    %     end        
    % end
   
end
