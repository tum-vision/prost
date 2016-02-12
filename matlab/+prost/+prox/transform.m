function [prox] = transform(child, a, b, c, d, e)
    
data = { a, b, c, d, e, child };
prox = { 'transform', child{2}, child{3}, child{4}, data };

end
