classdef variable < handle
    properties
        dim
        val
        sub_vars        
        idx
    end
    
    methods
        function obj = variable(dim)
            obj.dim = dim;
            obj.val = zeros(dim, 1);
            obj.sub_vars = {};
        end
    end
end
