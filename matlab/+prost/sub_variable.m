classdef sub_variable < handle
    properties
        dim
        val
        parent
        
        idx
    end
    
    methods
        function h = sub_variable(parent, dim)
            h.dim = dim;
            h.parent = parent;
            h.val = zeros(dim, 1);

            parent.sub_vars{end+1} = h;
        end
    end
end
