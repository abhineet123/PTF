clear all;
close all;
colRGBDefs;

k=importdata('matlab.txt');


figure;
subplot(1, 3, 1), plot(1:length(k), k(:, 1)), legend('hist');
grid on;

subplot(1, 3, 2), hold on;
matlab_grad=gradient(k(:, 1));
plot(1:length(k), k(:, 2), 'Color', col_rgb{strcmp(col_names,'green')});
plot(1:length(k), matlab_grad, 'Color', col_rgb{strcmp(col_names,'black')});
legend('grad', 'matlab grad');
grid on;

subplot(1, 3, 3), hold on;
matlab_hess=gradient(k(:, 2));
matlab_hess2=gradient(matlab_grad);

plot(1:length(k), k(:, 3), 'Color', col_rgb{strcmp(col_names,'green')});
plot(1:length(k), matlab_hess, 'Color', col_rgb{strcmp(col_names,'red')});
plot(1:length(k), matlab_hess2, 'Color', col_rgb{strcmp(col_names,'black')});

legend('hess', 'matlab hess', 'matlab hess2');
grid on;


