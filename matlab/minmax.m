syms x
fx = x^3+7+x^2;

dx = diff(fx,x);
dxx = diff(dx,x);


critpoint = double(solve(dxx));

cmin = min(critpoint) - 2;
cmax = max(critpoint) + 2;

for i = 1 : length(critpoint)
    disp(critpoint(i))
    dxx_sol = subs(dxx,x,critpoint(i));
    y_out = subs(fx, x, critpoint(i));
    if(dxx_sol == 0)
        disp("inflexion point");
    elseif(dxx_sol > 0)
        disp("min");
    elseif(dxx_sol < 0)
        disp("max");
        sprintf(critpoint(i))
    end
    hold on
    plot(critpoint(i), y_out, 'r*')
end

fplot(fx,[cmin, cmax], 'g')