ad_ssd=importdata('ad_ssd.txt')
ad_scv=importdata('ad_scv.txt')
ad_ncc=importdata('ad_ncc.txt')
x=1:6
hold on
plot(x, ad_ssd(:, 1), 'LineWidth', mean(ad_ssd(:, 2))*7, 'Color', 'r', 'LineStyle', '-', 'MarkerStyle', 's')
plot(x, ad_scv(:, 1), 'LineWidth', mean(ad_scv(:, 2))*7, 'Color', 'g', 'LineStyle', '-', 'MarkerStyle', '^')
plot(x, ad_ncc(:, 1), 'LineWidth', mean(ad_ncc(:, 2))*7, 'Color', 'b', 'LineStyle', '-', 'MarkerStyle', 'o')

