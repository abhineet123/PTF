clear all;
colRGBDefs;
label = 'B_8';
colors = {'black', 'red', 'green', 'blue'};
line_thickness = 2;

data=importdata('data.txt');

centroid_x=mean(data(1, :));
centroid_y=mean(data(2, :));

half_width=mean(abs(data(1, :)-centroid_x));
half_height=mean(abs(data(2, :)-centroid_y));

min_x = centroid_x - half_width;
min_y = centroid_y - half_height;

plot_min_x = min_x - 40;
plot_min_y = min_y - 40;

plot_width = 2*half_width + 80;
plot_height = 2*half_height + 80;

plot_max_x = plot_min_x + plot_width;
plot_max_y = plot_min_y + plot_height;

data(:, end+1)=data(:, 1);
n_corners=size(data, 1)/2;
figure, hold on;
for i=1:n_corners    
x1=data(i*2 - 1, :);
y1=-data(i*2, :);
if i==1
    mean_x=mean(x1);
    mean_y=mean(y1);
end
line(x1, y1, 'Color', col_rgb{strcmp(col_names,colors{i})}, 'LineWidth', line_thickness);
end
text(mean_x,mean_y, label, 'FontSize', 18, 'FontWeight', 'bold');
set(gca,'xlim',[plot_min_x, plot_max_x]);
set(gca,'ylim',[-plot_max_y, -plot_min_y]);

% set(gca, 'visible', 'off') ; 
set(gca,'xtick',[]);
set(gca,'xticklabel',[]);
set(gca,'ytick',[]);
set(gca,'yticklabel',[]);