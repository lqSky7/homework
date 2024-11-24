syms x
y = 3*x^2;
x1 = 1;
y1 = subs(y,x,x1);
dx = diff(y,x);
slope = subs(dx, x, x1);
hold on
plot(x1,y1);
fplot(y)
fplot(slope * (x - x1) + y1, 'g');
hold off