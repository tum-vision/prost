function [prox] = prox_moreau(child)
    
data = { child };
prox = { 'moreau', 0, 0, 0, 0, 0, data };

end
