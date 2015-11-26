function [prox] = prox_hyperplane( idx, count, dim, b)

data = { b };
prox = { 'hyperplane', idx, count, dim, false, false, data };
end
