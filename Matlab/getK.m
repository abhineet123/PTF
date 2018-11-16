clear all;

dx=0.7;
dy=0.4;
dxi=1-dx;
dyi=1-dy;
a=(dx*dy)/(sqrt(dxi*dxi+dyi*dyi));
b=(dxi*dy)/(sqrt(dx*dx+dyi*dyi));
c=(dxi*dyi)/(sqrt(dx*dx+dy*dy));
d=(dx*dyi)/(sqrt(dxi*dxi+dy*dy));

k=1/(a+b+c+d)

a=k*a
b=k*b
c=k*c
d=k*d

