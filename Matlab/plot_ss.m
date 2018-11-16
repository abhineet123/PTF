sr_8dof=importdata('sr_8dof.txt')
sr_6dof=importdata('sr_6dof.txt')
sr_4dof=importdata('sr_4dof.txt')
sr_2dof=importdata('sr_2dof.txt')

x=1:6
hold on
plot(x, sr_2dof, 'Color', 'r', 'LineStyle', '-', 'LineWidth', 3, 'Marker', 's', 'MarkerSize', 15)
plot(x, sr_4dof,  'Color', 'g', 'LineStyle', '-', 'LineWidth', 3, 'Marker', '^', 'MarkerSize', 15)
plot(x, sr_6dof,  'Color', 'b', 'LineStyle', '-', 'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 15)
plot(x, sr_8dof,  'Color', 'k', 'LineStyle', '-', 'LineWidth', 3, 'Marker', 'd', 'MarkerSize', 15)
