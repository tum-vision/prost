function get_all_variables(result, p_vars, pc_vars, d_vars, dc_vars)
% GET_ALL_VARIABLES get_all_variables(result, p_vars, pc_vars, d_vars, dc_vars)
%
% Retrieves all variables from primal-dual
% solution. The ordering is given by the problems
%                   
% min_{x,z} f(z) + g(x) s.t. z = Ax
%
% and
%
% min_{y,w} f^*(y) + g^*(w) s.t. w = -A^T y
%
% x: p_vars
% z: pc_vars
% y: d_vars
% w: dc_vars   
%
% TODO: doesn't set subvariables yet.
    
    num_primals = prod(size(p_vars));
    num_con_primals = prod(size(pc_vars));
    num_duals = prod(size(d_vars));
    num_con_duals = prod(size(dc_vars));
    
    idx = 0;
    for i=1:num_primals
        p_vars{i}.val = result.x(idx+1:idx+p_vars{i}.dim);
        idx = idx + p_vars{i}.dim;
    end

    idx = 0;
    for i=1:num_con_primals
        pc_vars{i}.val = result.z(idx+1:idx+pc_vars{i}.dim);
        idx = idx + pc_vars{i}.dim;
    end

    idx = 0;
    for i=1:num_duals
        d_vars{i}.val = result.y(idx+1:idx+d_vars{i}.dim);
        idx = idx + d_vars{i}.dim;
    end

    idx = 0;
    for i=1:num_con_duals
        dc_vars{i}.val = result.w(idx+1:idx+dc_vars{i}.dim);
        idx = idx + dc_vars{i}.dim;
    end
    
end
