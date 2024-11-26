% clc
% clear 
% close all
% % tangent

% syms x;
% f = 3*x^2;
% dx = diff(f,x);
% x1 = 3;
% y1 = subs(f,x,x1);
% slope = subs(dx,x,x1)
% fplot(f,[-10,10])
% hold on
% tangent = (x-x1)*slope + y1;
% fplot(tangent, [-10,10])
% plot(x1,y1,'k*') 

% contourcopy
% 
% 
% syms x y z;
% f = [3*x^2, 5*x*y,3*z]
% contf = 3*x^2 + 5*x*y + 3*Z;
% 
% p = inline(vectorize(f(1)),'x','y','z');
% q = inline(vectorize(f(2)),'x','y','z');
% r = inline(vectorize(f(3)),'x','y','z');
% 
% s = linspace(0,2,5);
% [X,Y,Z] = meshgrid(s,s,s);
% 
% t = p(X,Y,Z);
% u = q(X,Y,Z);
% v = r(X,Y,Z);
% 
% 
% quiver3(X,Y,Z,t,u,v,3)
% hold on
% ezcontour(contf, [-10 10])

% solid of rev
% 
% syms x 
% f = 3*sin(x);
% axis = 2;
% lt = [0,4];
% 
% vol = pi*abs(double(int((f-axis)^2, x, lt(1), lt(2))))
% c = matlabFunction(f);
% x_vals = linspace(-10 , 10);
% 
% [X,Y,Z] = cylinder(c(x_vals)-axis,100);
% surf(Z,Y+axis,X, 'EdgeColor','none',FaceAlpha=0.6)

% minmax single var
% syms x
% hold on
% f = 3*x^2;
% dx = diff(f,x);
% critpts = double(solve(dx));
% cmin = min(critpts) - 2;
% cmax = max(critpts) + 2;
% dxx = diff(dx,x);
% for i = 1:length(critpts)
%     y = subs(f,x,critpts(i));
%     dxx_sol = subs(dxx,x,critpts(i));
% 
%     disp(dxx_sol)
%     if(dxx_sol == 0)
%         fprintf("inflexion at,%.2f y=%.2f",critpts(i),y);
%     elseif(dxx_sol > 0)
% 
%         fprintf("min at,%.2f y=%.2f",critpts(i),y);
%     else
%         fprintf("max at,%.2f y=%.2f",critpts(i),y);
%     end
%     plot(y,critpts(i), 'k*')
% 
% end
% fplot(f,[cmin, cmax])
% 


% min max 2 var

% syms x y 
% hold on
% f = 2*x^3+x*y^2+5*x^2+y^2;
% fx = diff(f,x);
% fy = diff(f,y);
% fxx = diff(fx,x);
% fxy = diff(fx,y);
% fyy = diff(fy,y);
% 
% [cx, cy] = solve([fx==0, fy==0],[x,y]);
% cx = double(cx);
% cy = double(cy);
% 
% D = fxx*fyy - fxy^2;
% 
% for i = 1:length(cx)
%     dval = subs(D,{x,y}, {cx(i),cy(i)});
%     fval = subs(fxx,{x,y}, {cx(i),cy(i)} );
% 
% 
%     if(dval > 0 && fval > 0)
%         disp("minima")
%     elseif(dval > 0 && fval < 0)
%         disp("maxio")
%     elseif(dval < 0)
%         disp("saddle")
%     else
%         disp("fur")
%     end
% 
%     plot3(cx(i),cy(i), subs(f,{x,y}, {cx(i),cy(i)}), 'k*')
% 
% end
% 
% s = linspace(-10,10,100);
% [X,Y] = meshgrid(s,s);
% Z = double(subs(f, {x,y}, {X,Y}));
% figure(1)
% mesh(X,Y,Z)
% hold off

% syms x
% f = x^2-3;
% g = x;
% 
% cx = solve(f-g);
% 
% fplot(f)
% hold on
% fplot(g)
% for i = 1:length(cx)-1
%     xrang = linspace(cx(i), cx(i+1), 100);
%     yrn = subs(f,x,xrang);
%     grn = subs(g,x,xrang);
%     fill([xrang, fliplr(xrang)], [yrn, fliplr(grn)],'g')
% end


%lagrange 
% syms x y lam
% 
% 
% f = x^2 + 2*y^2; % Enter f(x,y) to be extremized 
% constraint = x^2 + y^2 - 1; % Enter the constraint function g(x,y)
% lag = f - lam * constraint; 
% 
% lagx = diff(lag,x);
% lagy = diff(lag,y);
% 
% [ax,ay,alam] = solve([lagx == 0, lagy==0,constraint ==0], [x,y,lam]);
% cx= double(ax)
% cy = double(ay)
% alam = double(alam)
% 
% 
% 
% realx = []
% realy = []
% %real only pts
% for i = 1:length(ax)
%     if isreal(cx(i)) && isreal(cy(i))
%     realx = [realx, cx(i)];
%     realy = [realy, cy(i)];
%     end
% end
% 
% T = double(subs(f,{x,y}, {realx,realy}));
% figure()
% for i = 1: length(T)
%     D = [realx(i)+3,realx(i)-3,realy(i)+3,realy(i)-3];
%     ezcontour(f,D)
%     hold on
%     ezplot(constraint,D)
%     plot(realx(i),realy(i),'k.', MarkerSize=30)
% end

%integration
syms x y z;
forig = log(x^2 + y^2 + z^2);
f = gradient(forig)
p = inline(vectorize(f(1)), 'x','y','z');
q = inline(vectorize(f(2)), 'x','y','z');
r = inline(vectorize(f(3)), 'x','y','z');

s = linspace(0,4,5);
[X,Y,Z] = meshgrid(s,s,s);
t = p(X,Y,Z);
u = q(X,Y,Z);
v = r(X,Y,Z);

quiver3(X,Y,Z,t,u,v,3);
hold on

disp("gradient is: ")
