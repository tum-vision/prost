function [result, timing] = eval_prox(prox, arg, tau, Tau, verbose)
    
    if nargin < 5
        verbose = false;
    end
    
    [result, timing] = prost_('eval_prox', prox(0, size(arg,1)), arg, tau, Tau, verbose);
    
end
