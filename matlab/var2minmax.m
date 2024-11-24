syms x;
syms y;

f = 2*x^3+x*y^2+5*x^2+y^2;
dx = diff(f,x);
dy = diff(f,y);

[cx, cy] = solve(dx,dy);
cx = double(cx);
cy = double(cy);

dxy = diff(dy,x);
dyy = diff(dy,y);
dxx = diff(dx,x);

D = dxx*dyy - dxy^2;

for i = 1 : length(cx)
    fxx = subs(dxx,{x,y}, {cx(i),cy(i)});
    dval = subs(D,{x,y}, {cx(i),cy(i)});
    
    if(dval > 0 && fxx > 0)
    sprintf("Local Minima At: %.2f %.2f", cx(i), (cy(i)))
    elseif(dval > 0 && fxx < 0)
    sprintf("Local max At: %.2f %.2f", (cx(i)), (cy(i)))
    elseif(dval < 0)
    sprintf("saddle At: %.2f %.2f", (cx(i)), (cy(i)))
    else
    sprintf("no conclusion: %.2f %.2f", (cx(i)), (cy(i)))
    end
end

[xvals, yvals] = meshgrid(linspace(-10,10,50), linspace(-10,10,50));
zvals = double(subs(f,{x,y},{xvals,yvals}));

figure
mesh(xvals, yvals, zvals)
hold all
for i = 1:length(cx)
z = subs(f, {x,y}, {cx(i), cy(i)});
plot3(cx(i),cy(i),z,'ro', MarkerSize=10, Color='r')
end
hold off