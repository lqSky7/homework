syms x
fx= x^1/2;
range = [0,4];
axisrot = 1;

vol = pi*abs(double(int((fx-axisrot)^2, range(1), range(2))));
fprintf(" volume is %d", vol);

c = matlabFunction(fx);
x_range = linspace(range(1), range(2), 100);

[X,Y,Z] = cylinder(c(x_range)-axisrot, 100);
Z = range(1) + Z.*(range(2)-range(1));
surf(Z,Y+axisrot,X, 'EdgeColor','none', FaceAlpha='0.6')
view(3)