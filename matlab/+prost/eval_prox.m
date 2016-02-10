function [result] = eval_prox(prox, arg, tau, Tau)
    
    result = prost_('eval_prox', prox, arg, tau, Tau);
    
end
