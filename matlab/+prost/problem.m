classdef problem < handle
    properties
        data
        nrows
        ncols
    end
    
    methods
        function obj = problem()
            obj.set_scaling_alpha(1);
            obj.data.prox_f = {};
            obj.data.prox_fstar = {};
            obj.data.prox_g = {};
            obj.data.prox_gstar = {};
            obj.data.linop = {};
            obj.nrows = 0;
            obj.ncols = 0;
        end   
        
        function obj = set_scaling_identity(obj)
            obj.data.scaling = 'identity';
        end
        
        function obj = set_scaling_alpha(obj, alpha)
            obj.data.scaling = 'alpha';
            obj.data.scaling_alpha = alpha;
        end

        function obj = set_scaling_custom(obj, left, right)
            obj.data.scaling = 'custom';
            obj.data.scaling_left = left;
            obj.data.scaling_right = right;
        end   
    end
    
end
