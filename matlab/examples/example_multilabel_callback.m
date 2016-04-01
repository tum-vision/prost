function [is_converged] = example_multilabel_callback(it, x, y, ny, ...
                                                      nx, L, im)

    result.x = x;
    result.y = y;
    u = prost.variable(nx*ny*L);
    prost.get_all_variables(result, {u}, {}, {}, {});
    
    u = reshape(u.val, [ny nx L]);
    imshow([im, u]);
    
    is_converged = false;
    
    fprintf('\n');
    
end

