function draw_gt(cx, cy)
% plot ground truth template onto the plot
cx = [cx cx(1)];
cy = [cy cy(1)];
hold on;
plot(cx, cy, 'g', 'LineWidth', 1.5);
hold off;