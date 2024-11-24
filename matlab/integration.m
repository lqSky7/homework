syms x;
fx = x^2;
gx = x;

intsec = double(solve(fx-gx));
cmin = min(intsec) -2;
cmax = max(intsec) +2;

hold on
fplot(fx, [cmin, cmax])
fplot(gx, [cmin, cmax])

area = 0;

for i = 1: length(intsec) - 1
    x_range = linspace(intsec(i), intsec(i+1));
    fxx = subs(fx, x, x_range);
    gxx = subs(gx, x, x_range);

    fill([x_range, fliplr(x_range)], [fxx, fliplr(gxx)],'g')
    area = area + abs(double(int(fx-gx, x, intsec(i), intsec(i+1))));
end
sprintf("%d",area)