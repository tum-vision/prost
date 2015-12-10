function [prox] = prox_plus_linterm(child, c)
    
data = { child, c };
prox = { 'plus_linterm', 0, 0, 0, 0, 0, data };

end
