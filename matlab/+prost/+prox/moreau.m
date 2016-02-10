function [prox] = moreau(child)
    
data = { child };
prox = { 'moreau', 0, 0, false, data };

end
