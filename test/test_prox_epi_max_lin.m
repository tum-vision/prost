
t = [0;0; 1;0; 0;0.9; 0;7; 1; 1];
b = [1;2;3;9; 3];
v = [-60;25;20; 10; 30; 7; -10; -30; 7];
idx = [0;5;10];
count = [5;5;5];
proj = pdsolver_eval_prox(prox_epi_max_lin(0, 3, 3, false, [t;t;t], [b;b;b], idx, count), v, 0, ones(6, ...
                                                  1));
                                              
%y = pdsolver_eval_prox(prox_halfspace(0, 1, 3, false, [t(1:3);-1]), v, 1, ones(4, ...
%                                                  1));
                                              
A=[t(3:4), t(5:6), t(7:8)];
A=[A;[-1, -1, -1]];
inv=pinv(A);
%proj2 = (eye(3)-A*inv)*v - inv'*b(2:4)    

x=linspace(-80,80,600);
y=linspace(-80,80,600);

[X,Y]=meshgrid(x,y); 

Z1 = t(1)*X+t(2)*Y + b(1);
Z2 = t(3)*X+t(4)*Y + b(2);
Z3 = t(5)*X+t(6)*Y + b(3);
Z4 = t(7)*X+t(8)*Y + b(4);
Z5 = t(9)*X+t(10)*Y + b(5);

Z = max(cat(3, Z1, Z2, Z3, Z4, Z5), [], 3);
hold on;
axis equal;
mesh(X,Y,Z);
plot3(v(1), v(2), v(3),'o');
plot3(v(4), v(5), v(6),'o');
plot3(v(7), v(8), v(9),'o');
plot3(proj(1), proj(2), proj(3), 'o');
plot3(proj(4), proj(5), proj(6), 'o');
plot3(proj(7), proj(8), proj(9), 'o');