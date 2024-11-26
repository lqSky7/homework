% Define parameter t
t = linspace(0, 2*pi, 100);

% For a circle with radius r centered at (0,0):
r = 1;  % radius (change this value to adjust circle size)
x = r * cos(t);
y = r * sin(t);

% Plot the circle
plot(x, y)
axis equal  % this makes the circle look circular instead of elliptical
grid on
title('Circle with Radius 1')
xlabel('x')
ylabel('y')