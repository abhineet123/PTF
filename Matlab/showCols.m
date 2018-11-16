clear all;
colRGBDefs;
row_length=12;
n_cols=length(col_names);
w = 10;
h = 10;
x = 1;
y = 1;

figure;
hold on;
col_id=1;
 plot_legend={};
 max_x=w*row_length;
max_y=h*n_cols/row_length;

for col=col_names
    rectangle('Position',[x,y,w,h],...
        'FaceColor',col_rgb{strcmp(col_names,col)},...
        'EdgeColor',col_rgb{strcmp(col_names,col)},...
        'LineWidth',3);
%     annotation('textbox',[x,y,w,h],'String',col,'FitBoxToText','on');
%     plot_legend=[plot_legend col];
    col_id=col_id+1;
    x=x+w;
    if col_id>row_length
        col_id=1;
        x=1;
        y=y+h;
    end
end
% legend(plot_legend);