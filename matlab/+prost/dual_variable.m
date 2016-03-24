classdef dual_variable < handle
    properties
        dim
        idx
        val
        fun
    end
    
    methods
        function h = dual_variable(dim)
            h.dim = dim;
        end
    end
end
