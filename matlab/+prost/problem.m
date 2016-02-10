function [prob] = problem(varargin)

    p = inputParser;
    addOptional(p, 'linop', {});
    addOptional(p, 'prox_f', {});
    addOptional(p, 'prox_g', {});
    addOptional(p, 'prox_fstar', {});
    addOptional(p, 'prox_gstar', {});
    addOptional(p, 'scaling', 'alpha');
    addOptional(p, 'scaling_alpha', 1);
    addOptional(p, 'scaling_left', 1);
    addOptional(p, 'scaling_right', 1);

    p.parse(varargin{:});
    
    prob = p.Results;
    
end
