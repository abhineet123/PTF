function plotDecompResults(set_xticks, colors, linestyles)
if nargin<1
    set_xticks=1;
end
if nargin<2
    colors='rgbk';
%     colors=['r', 'g', 'r', 'g'];
end
if nargin<3
    linestyles='--::';
%     linestyles=['-', '-', ':', ':'];
end
fwd_data=importdata('result_fwd.txt');
inv_data=importdata('result_inv.txt');
ls_data=importdata('result_ls_fwd.txt');

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 12);
%set(0,'DefaultAxesFontWeight', 'bold');
% hold on;
nlines=3;
% for i=1:nlines
%     lines(i)=plot(data.data(:, 1), data.data(:, i+1), '-s',...
%         'LineWidth',2, ');
%     %legend(data.colheaders(i+1), 'interpreter','none');
% end
% figure;
data_count=size(fwd_data.data, 1)

affine_data=zeros(data_count, 3);
affine_data(:, 1)=fwd_data.data(:, 2);
affine_data(:, 2)=inv_data.data(:, 2);
affine_data(:, 3)=ls_data.data(:, 2);

se2_data=zeros(data_count, 3);
se2_data(:, 1)=fwd_data.data(:, 3);
se2_data(:, 2)=inv_data.data(:, 3);
se2_data(:, 3)=ls_data.data(:, 3);

rt_data=zeros(data_count, 3);
rt_data(:, 1)=fwd_data.data(:, 4);
rt_data(:, 2)=inv_data.data(:, 4);
rt_data(:, 3)=ls_data.data(:, 4);

trans_data=zeros(data_count, 3);
trans_data(:, 1)=fwd_data.data(:, 5);
trans_data(:, 2)=inv_data.data(:, 5);
trans_data(:, 3)=ls_data.data(:, 5);

x_data=1:data_count;
xtick_labels=fwd_data.textdata(:, 2);

figure
affine_plot=plot(x_data, affine_data, '-s', 'LineWidth',2);
for i=1:nlines
    set(affine_plot(i), 'Color', colors(i));
    set(affine_plot(i), 'LineStyle', linestyles(i));
end

plot_legend={'Forward Decomposition', 'Inverse Decomposition', 'Least Squares'};
legend(plot_legend, 'interpreter','none');
set(gca, 'XTick', x_data);
set(gca, 'XTickLabel', xtick_labels);
xlabel('Sequences');
ylabel('Mean Error');
title('Mean Corner Distance Error: Affine');
grid on;

figure
se2_plot=plot(x_data, se2_data, '-s', 'LineWidth',2);
for i=1:nlines
    set(se2_plot(i), 'Color', colors(i));
    set(se2_plot(i), 'LineStyle', linestyles(i));
end
plot_legend={'Forward Decomposition', 'Inverse Decomposition', 'Least Squares'};
legend(plot_legend, 'interpreter','none');
set(gca, 'XTick', x_data);
set(gca, 'XTickLabel', xtick_labels);
xlabel('Sequences');
ylabel('Mean Error');
title('Mean Corner Distance Error: SE2');
grid on;

figure
rt_plot=plot(x_data, rt_data, '-s', 'LineWidth',2);
for i=1:nlines
    set(rt_plot(i), 'Color', colors(i));
    set(rt_plot(i), 'LineStyle', linestyles(i));
end
plot_legend={'Forward Decomposition', 'Inverse Decomposition', 'Least Squares'};
legend(plot_legend, 'interpreter','none');
set(gca, 'XTick', x_data);
set(gca, 'XTickLabel', xtick_labels);
xlabel('Sequences');
ylabel('Mean Error');
title('Mean Corner Distance Error: RT');
grid on;

figure
trans_plot=plot(x_data, trans_data, '-s', 'LineWidth',2);
for i=1:nlines
    set(trans_plot(i), 'Color', colors(i));
    set(trans_plot(i), 'LineStyle', linestyles(i));
end
plot_legend={'Forward Decomposition', 'Inverse Decomposition', 'Least Squares'};
legend(plot_legend, 'interpreter','none');
set(gca, 'XTick', x_data);
set(gca, 'XTickLabel', xtick_labels);
xlabel('Sequences');
ylabel('Mean Error');
title('Mean Corner Distance Error: Trans');
grid on;


