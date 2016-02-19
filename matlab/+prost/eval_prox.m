function [result] = eval_prox(prox, arg, tau, Tau, verbose)
    
    if nargin < 5
        verbose = false
    end
    
    result = prost_('eval_prox', prox, arg, tau, Tau, verbose);
    
end
