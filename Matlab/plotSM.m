plot_cols='rgbk';
plot_markers='s^od';
line_width_sr=4;
line_width_ad=5;
line_style_sr=':';
line_style_ad='-';
marker_size=10;
bar_width=0.7;
font_size=20;
y_label_sr='\textbf{Success Rate ($\frac{|S|}{|F|}$)}';
y_label_ad='\textbf{Average Drift (Pixels)}';
x_label='\textbf{Sequences}';

sm_ssd=importdata('sm_ssd.txt');
sm_ssd_ad=importdata('sm_ssd_ad.txt');

sm_scv=importdata('sm_scv.txt');
sm_scv_ad=importdata('sm_scv_ad.txt');

sm_ncc=importdata('sm_ncc.txt');
sm_ncc_ad=importdata('sm_ncc_ad.txt');

sm_ssd8=sm_ssd.data(:, 1:3);
sm_ssd6=sm_ssd.data(:, 4:7);
sm_ssd4=sm_ssd.data(:, 8:11);
sm_ssd2=sm_ssd.data(:, 12:15);

sm_ssd_ad8=sm_ssd_ad.data(:, 1:3);
sm_ssd_ad6=sm_ssd_ad.data(:, 4:7);
sm_ssd_ad4=sm_ssd_ad.data(:, 8:11);
sm_ssd_ad2=sm_ssd_ad.data(:, 12:15);

sm_scv8=sm_scv.data(:, 1:3);
sm_scv6=sm_scv.data(:, 4:7);
sm_scv4=sm_scv.data(:, 8:11);
sm_scv2=sm_scv.data(:, 12:15);

sm_scv_ad8=sm_scv_ad.data(:, 1:3);
sm_scv_ad6=sm_scv_ad.data(:, 4:7);
sm_scv_ad4=sm_scv_ad.data(:, 8:11);
sm_scv_ad2=sm_scv_ad.data(:, 12:15);

sm_ncc8=sm_ncc.data(:, 1:3);
sm_ncc6=sm_ncc.data(:, 4:7);
sm_ncc4=sm_ncc.data(:, 8:11);
sm_ncc2=sm_ncc.data(:, 12:15);

sm_ncc_ad8=sm_ncc_ad.data(:, 1:3);
sm_ncc_ad6=sm_ncc_ad.data(:, 4:7);
sm_ncc_ad4=sm_ncc_ad.data(:, 8:11);
sm_ncc_ad2=sm_ncc_ad.data(:, 12:15);

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 16);
set(0,'DefaultAxesFontWeight', 'bold');

sr_data=sm_ssd6;
ad_data=sm_ssd_ad6;
plot_title='SSD 6DOF';

x_data=1:6;
% plot(x_data, sm_ssd8(:, 1), 'r:s',...
%     x_data, sm_ssd8(:, 2), 'g:^',...
%     x_data, sm_ssd8(:, 3), 'b:o', 'LineWidth', 3, 'MarkerSize', 10);

[ax,fig_ad,fig_sr]=plotyy(x_data, ad_data, x_data, sr_data, 'bar', 'plot');

sr_plot_count=length(fig_sr);
for i=1:sr_plot_count
    set(fig_sr(i), 'Color', plot_cols(i));
    set(fig_sr(i), 'Marker', plot_markers(i));
end
ad_plot_count=length(fig_ad);
for i=1:ad_plot_count
    set(fig_ad(i), 'EdgeColor', plot_cols(i));
    set(fig_ad(i), 'FaceColor', 'w');
end

set(fig_sr, 'LineWidth', line_width_sr);
set(fig_sr, 'LineStyle', line_style_sr);
set(fig_sr, 'MarkerSize', marker_size);

set(fig_ad, 'LineStyle', line_style_ad);
set(fig_ad, 'LineWidth', line_width_ad);
set(fig_ad, 'BarWidth', bar_width);

ax_sr=ax(2);
ax_ad=ax(1);

yticks_sr=0:0.1:1.0;
max_ad=ceil(max(max(ad_data)));
yticks_ad=linspace(0, max_ad, length(yticks_sr));
yticks_label_sr=strtrim(cellstr(num2str(yticks_sr'))');
yticks_label_ad=strtrim(cellstr(num2str(yticks_ad'))');


set(ax_sr,'Ytick', yticks_sr);
set(ax_sr,'YtickLabel',yticks_label_sr);
set(ax_sr,'YAxisLocation', 'left');
set(ax_sr,'YGrid', 'on');
set(ax_sr,'XGrid', 'on');
set(ax_sr,'FontSize', font_size);
set(ax_sr, 'XTick', x_data);
set(ax_sr, 'XTickLabel', sm_ssd.textdata(2:7, 1));
set(ax_sr,'Ylim',[min(yticks_sr), max(yticks_sr)]);
ylabel(ax_sr, y_label_sr, 'interpreter', 'latex', 'Fontweight', 'Bold');
xlabel(ax_sr, x_label, 'interpreter', 'latex', 'Fontweight', 'Bold');

set(ax_ad,'Ytick', yticks_ad);
set(ax_ad,'YtickLabel',yticks_label_ad);
set(ax_ad,'YAxisLocation', 'right');
set(ax_ad,'YGrid', 'on');
set(ax_ad,'XGrid', 'on');
set(ax_ad,'FontSize', font_size);
set(ax_ad,'Xtick',[]);
set(ax_ad,'XtickLabel',[]);
set(ax_ad,'Ylim',[min(yticks_ad), max(yticks_ad)]);
ylabel(ax_ad, y_label_ad, 'interpreter', 'latex', 'Fontweight', 'Bold');


pfig_sr=get(fig_sr);
pfig_ad=get(fig_ad);

pax_sr=get(ax_sr);
pax_ad=get(ax_ad);

if sr_plot_count==4
    legend_sr={'IC', 'ESM', 'NNIC', 'PF'};
elseif sr_plot_count==3
    legend_sr={'IC', 'ESM', 'NNIC'};
end
if ad_plot_count==4
    legend_ad={'IC', 'ESM', 'NNIC', 'PF'};
elseif sr_plot_count==3
    legend_ad={'IC', 'ESM', 'NNIC'};
end

[leg_handle_ad, icons_ad, plots_ad, str_ad]=legend(ax_ad, legend_ad, 'interpreter','none');
[leg_handle_sr, icons_sr, plots_sr, str_sr]=legend(ax_sr, legend_sr, 'interpreter','none');
%set(icons_ad,'linewidth',2);
set(leg_handle_ad, 'Box', 'off')
%set(leg_handle_sr, 'Box', 'off')
set(leg_handle_ad, 'Color', 'w')
set(leg_handle_sr, 'Color', 'w')
set(icons_sr(9:12),'LineWidth',3);
set(icons_sr(9:12),'MarkerSize',8);
set(icons_ad(1:4),'LineWidth',1);

plh_sr=get(leg_handle_sr);
plh_ad=get(leg_handle_ad);

picons_sr=get(icons_sr(9));
pplots_sr=get(plots_sr);




title(plot_title);
