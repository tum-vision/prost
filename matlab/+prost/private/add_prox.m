function [prox] = add_prox(prox_to_add, prox)
    num_prox = prod(size(prox));
    prox_idx = -1;
    idx = prox_to_add{2};

    for i=1:num_prox
        data_prox = prox{i};
        
        if data_prox{2} == idx
            prox_idx = i;
            break;
        end
    end

    if prox_idx == -1
        prox{end + 1} = prox_to_add;
    else
        prox{prox_idx} = prox_to_add;
    end
end