ad_ic=importdata('ad_ic.txt')
ad_esm=importdata('ad_esm.txt')
ad_nnic=importdata('ad_nnic.txt')
ad_pf=importdata('ad_pf.txt')

x=1:6
hold on
plot(x, ad_ic, 'Color', 'r', 'LineStyle', '-', 'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 15)
plot(x, ad_esm,  'Color', 'g', 'LineStyle', '-', 'LineWidth', 3, 'Marker', 's', 'MarkerSize', 15)
plot(x, ad_nnic,  'Color', 'b', 'LineStyle', '-', 'LineWidth', 3, 'Marker', 'v', 'MarkerSize', 15)
plot(x, ad_pf,  'Color', 'k', 'LineStyle', '-', 'LineWidth', 3, 'Marker', 'd', 'MarkerSize', 15)
