syms x y z
f = [x, y*x, x*y*z];

p = inline(vectorize(f(1)), 'x','y','z');
q = inline(vectorize(f(2)), 'x','y','z');
r = inline(vectorize(f(3)), 'x','y','z');

s = linspace(9,10,20);
l= s;
k= s;
[X,Y,Z] = meshgrid(s,l,k);, 'x','y','z'
t = p(X,Y,Z);
u = q(X,Y,Z);
v = r(X,Y,Z);
quiver3(X,Y,Z,t,u,v,1);