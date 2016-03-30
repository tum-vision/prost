classdef variable < handle
    properties
        dim
        idx
        val
        fun
        linop
        pairing
    end
    
    methods
        function h = variable(dim)
            h.dim = dim;
            h.pairing = {};
            h.linop = {};
            h.fun = prost.function.zero();
        end
    end
end
