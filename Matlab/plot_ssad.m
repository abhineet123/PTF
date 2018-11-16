ad_8dof=importdata('ad_8dof.txt')
ad_6dof=importdata('ad_6dof.txt')
ad_4dof=importdata('ad_4dof.txt')
ad_2dof=importdata('ad_2dof.txt')

x=1:6
hold on
plot(x, ad_2dof, 'Color', 'r', 'LineStyle', '-', 'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 15)
plot(x, ad_4dof,  'Color', 'g', 'LineStyle', '-', 'LineWidth', 3, 'Marker', 's', 'MarkerSize', 15)
plot(x, ad_6dof,  'Color', 'b', 'LineStyle', '-', 'LineWidth', 3, 'Marker', 'v', 'MarkerSize', 15)
plot(x, ad_8dof,  'Color', 'k', 'LineStyle', '-', 'LineWidth', 3, 'Marker', 'd', 'MarkerSize', 15)
