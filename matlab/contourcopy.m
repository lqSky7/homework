syms x y
l = 4*x^2 - 5*y*x;
f = gradient(l)
p = inline(vectorize(f(1)), 'x','y');
q = inline(vectorize(f(2)), 'x','y');

a = linspace(0,4,20);
[X,Y] = meshgrid(a,a);

u = p(X,Y);
v = q(X,Y);

quiver(u,v,X,Y,1);
hold on
ezcontour(l,[-20,30])