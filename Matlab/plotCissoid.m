k=7;
n_samples=1000;
x=linspace(0,2*k,n_samples);
y=sqrt(-(x.*x.*x)./(x-2*k));
figure, plot(x, y), grid on;