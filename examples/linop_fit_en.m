function [K, y] = linop_fit_en(m, n, L, d, deg, U)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    if(mod(d-1, L-1) ~= 0 || L-1 > d-1)
        error('#labels + #points combination is not permitted: (d-1) mod (L-1) has to be 0 and L-1 <= d.');
    end
    
    
    range = (d-1) / (L-1);

    rows1 = zeros(1, m*n*(L-1)*(range-1)*(deg+1));
    cols1 = zeros(1, m*n*(L-1)*(range-1)*(deg+1));
    vals1 = zeros(1, m*n*(L-1)*(range-1)*(deg+1));
    
    rows2 = zeros(1, m*n*(L-2)*2*(deg+1));
    cols2 = zeros(1, m*n*(L-2)*2*(deg+1));
    vals2 = zeros(1, m*n*(L-2)*2*(deg+1));
    
    y = zeros(m*n*(L-1)*(range+1), 1);
    
    for j=1:n
        for i=1:m
            for k=1:L-1
                x = ((k-1)*range:k*range)';
                y((j-1)*m*(L-1)*(range+1) + (i-1)*(L-1)*(range+1) + (k-1)*(range+1) + 1:(j-1)*m*(L-1)*(range+1) + (i-1)*(L-1)*(range+1) + k*(range+1)) = U((j-1)*m*d + (i-1)*d + x + 1);
                
                idx = (j-1)*m*(L-1)*(range+1)*(deg+1) + (i-1)*(L-1)*(range+1)*(deg+1) + (k-1)*(range+1)*(deg+1)+1:(j-1)*m*(L-1)*(range+1)*(deg+1) + (i-1)*(L-1)*(range+1)*(deg+1) + k*(range+1)*(deg+1);
                
                vals = zeros((range+1)*(deg+1), 1);
                for l=1:deg+1
                    vals((l-1)*(range+1)+1:l*(range+1)) = x.^(deg+1-l);
                end
                
                vals1(idx) = vals;
                rows1(idx) = repmat(((j-1)*m*(L-1)*(range+1) + (i-1)*(L-1)*(range+1) + (k-1)*(range+1)+1:(j-1)*m*(L-1)*(range+1) + (i-1)*(L-1)*(range+1) + k*(range+1))', [(deg+1), 1]);
                
                cols = zeros((range+1)*(deg+1), 1);
                for l=1:deg+1
                    cols((l-1)*(range+1)+1:l*(range+1)) = repmat((j-1)*m*(L-1)*(deg+1) + (i-1)*(L-1)*(deg+1) + (k-1)*(deg+1)+l, [range+1, 1]);
                end
                cols1(idx) = cols;

                if(k >= 2)                    
                    idx = (j-1)*m*(L-2)*2*(deg+1) + (i-1)*(L-2)*2*(deg+1) + (k-2)*2*(deg+1)+1:(j-1)*m*(L-2)*2*(deg+1) + (i-1)*(L-2)*2*(deg+1) + (k-1)*2*(deg+1);
                    
                    vals = zeros(2*(deg+1), 1);
                    for l=1:deg+1
                        vals(l) = x(1)^(deg+1-l);
                    end
                    for l=1:deg+1
                        vals(deg+1+l) = -x(1)^(deg+1-l);
                    end
                    vals2(idx) = vals;
                    rows2(idx) = repmat((j-1)*m*(L-2) + (i-1)*(L-2) + (k-1), [2*(deg+1), 1]);
                    cols2(idx) = (k-2)*(deg+1)+(1:2*(deg+1));
                end
            end
        end
    end
    
    X = sparse(rows1, cols1, vals1, m*n*(L-1)*(range+1), (deg+1)*m*n*(L-1));
    Aeq = sparse(rows2, cols2, vals2, m*n*(L-2), (deg+1)*m*n*(L-1));

    K = [X; Aeq];
end

