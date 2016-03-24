classdef primal_variable < handle
    properties
        dim
        idx
        val
        fun
        linop
        pairing
    end
    
    methods
        function h = primal_variable(dim)
            h.dim = dim;
        end
    end
end
