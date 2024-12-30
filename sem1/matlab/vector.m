syms x y 
f = [x^2, y*x];
p = inline(vectorize(f(1)), 'x', 'y'); % Define 'p' with 'x' and 'y' as inputs
q = inline(vectorize(f(2)), 'x', 'y'); % Define 'q' with 'x' and 'y' as inputs

a = linspace(-10, 10,30);
b = a;
[X, Y] = meshgrid(a, b);
u = p(X, Y); % Evaluate 'p' at grid points
v = q(X, Y); % Evaluate 'q' at grid points

quiver(X, Y, u, v, 1)
axis on
xlabel('x') % Label x-axis
ylabel('y') % Label y-axis
title('Vector Field Plot') % Add title for clarity
