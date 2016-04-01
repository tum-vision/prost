classdef sub_variable < handle
    properties
        dim
        idx
        val
        fun
        linop
        pairing
        parent
    end
    
    methods
        function h = sub_variable(parent, dim)
            h.dim = dim;
            h.parent = parent;
            h.pairing = {};
            h.linop = {};
            h.fun = prost.function.zero();

            parent.fun = {};
            parent.subvars{end+1} = h;
        end
    end
end
