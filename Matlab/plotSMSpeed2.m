speed_ssd=importdata('speed_ssd.txt');

speed_scv=importdata('speed_scv.txt');

speed_ncc=importdata('speed_ncc.txt');

speed_ssd8=speed_ssd.data(:, 1:3);
speed_ssd6=speed_ssd.data(:, 4:7);
speed_ssd4=speed_ssd.data(:, 8:11);
speed_ssd2=speed_ssd.data(:, 12:15);

speed_scv8=speed_scv.data(:, 1:3);
speed_scv6=speed_scv.data(:, 4:7);
speed_scv4=speed_scv.data(:, 8:11);
speed_scv2=speed_scv.data(:, 12:15);

speed_ncc8=speed_ncc.data(:, 1:3);
speed_ncc6=speed_ncc.data(:, 4:7);
speed_ncc4=speed_ncc.data(:, 8:11);
speed_ncc2=speed_ncc.data(:, 12:15);


set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 16);
set(0,'DefaultAxesFontWeight', 'bold');

plot_cols='rgbk';
plot_markers='s^od';
line_width_sr=4;
line_width_ad=5;
line_style_sr=':';
marker_size=10;
bar_width=0.7;
font_size=20;
y_label_sr='\textbf{Success Rate ($\frac{|S|}{|F|}$)}';
x_label='\textbf{Sequences}';

sr_data=speed_ssd6;
ad_data=speed_ssd_ad6;
plot_title='SSD 6DOF';
enl_factor=1.2;

x_data=1:6;
% plot(x_data, speed_ssd8(:, 1), 'r:s',...
%     x_data, speed_ssd8(:, 2), 'g:^',...
%     x_data, speed_ssd8(:, 3), 'b:o', 'LineWidth', 3, 'MarkerSize', 10);

fig_sr=plot(x_data, sr_data);

sr_plot_count=length(fig_sr);
for i=1:sr_plot_count
    set(fig_sr(i), 'Color', plot_cols(i));
    set(fig_sr(i), 'Marker', plot_markers(i));
end

set(fig_sr, 'LineWidth', line_width_sr);
set(fig_sr, 'LineStyle', line_style_sr);
set(fig_sr, 'MarkerSize', marker_size);


ax_sr=gca;

max_sr=ceil(max(max(sr_data)));
yticks_sr=linspace(0, max_sr, 10);
yticks_label_sr=strtrim(cellstr(num2str(yticks_sr'))');


set(ax_sr,'Ytick', yticks_sr);
set(ax_sr,'YtickLabel',yticks_label_sr);
set(ax_sr,'YAxisLocation', 'left');
set(ax_sr,'YGrid', 'on');
set(ax_sr,'XGrid', 'on');
set(ax_sr,'FontSize', font_size);
set(ax_sr, 'XTick', x_data);
set(ax_sr, 'XTickLabel', speed_ssd.textdata(2:7, 1));
set(ax_sr,'Ylim',[min(yticks_sr), max(yticks_sr)]);
ylabel(ax_sr, y_label_sr, 'interpreter', 'latex', 'Fontweight', 'Bold');
xlabel(ax_sr, x_label, 'interpreter', 'latex', 'Fontweight', 'Bold');

pfig_sr=get(fig_sr);

pax_sr=get(ax_sr);

if sr_plot_count==4
    legend_sr={'IC', 'ESM', 'NNIC', 'PF'};
elseif sr_plot_count==3
    legend_sr={'IC', 'ESM', 'NNIC'};
end
[leg_handle_sr, icons_sr, plots_sr, str_sr]=legend(ax_sr, legend_sr, 'interpreter','none');

if sr_plot_count==4
    set(icons_sr(9:12),'LineWidth',3);
    set(icons_sr(9:12),'MarkerSize',8);
    set(icons_ad(1:4),'LineWidth',1);
elseif sr_plot_count==3
    set(icons_sr(7:9),'LineWidth',3);
    set(icons_sr(7:9),'MarkerSize',8);
end

current_pos=get(leg_handle_sr, 'Position');
new_width=current_pos(3)*enl_factor;
new_height=current_pos(4)*enl_factor;
new_x0=current_pos(1)-(new_width-current_pos(3));
new_y0=current_pos(2)-(new_height-current_pos(4));
set(leg_handle_sr, 'Position', [new_x0,new_y0,  new_width, new_height]);

plh_sr=get(leg_handle_sr);
picons_sr=get(icons_sr(9));
pplots_sr=get(plots_sr);

title(plot_title);
