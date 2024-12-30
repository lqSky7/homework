clc
clear all
close all
syms x y lambda

f = x^2 + 2*y^2; % Enter f(x,y) to be extremized 
constraint = x^2 + y^2 - 1; % Enter the constraint function g(x,y)
lag = f - lambda * constraint;

lagx = diff(lag, x);
lagy = diff(lag, y);

[ax, ay, alam] = solve([lagx==0, lagy==0, constraint==0], [x,y,lambda]);
ax = double(ax);
ay = double(ay);

realx = [];
realy = [];

for i = 1:length(ax)
    if(isreal(ax(i)) && isreal(ay(i)))
        realx = [realx, ax(i)];
        realy = [realy, ay(i)];
    end
end

T = double(subs(f, {x,y}, {realx, realy}));
figure(1);


for i = 1:length(T)
    range = [-10 10 -10 10];
    fprintf("critpoits are %.2f %.2f \n", realx(i), realy(i));
    fprintf("func value at those is %.2f \n", T(i));

    ezcontour(f, range);
    hold on
    ezplot(constraint, range);
    plot(realx(i),realy(i),'k.' ,'MarkerSize', 15, 'Color', 'r');
end
