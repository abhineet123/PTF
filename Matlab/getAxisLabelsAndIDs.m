function [x_label, y_label, x_id, y_id] = getAxisLabelsAndIDs(grid_type)
if strcmp(grid_type,'trans')
    y_id=2;
    x_id=3;
    y_label='t_x';
    x_label='t_y';
elseif strcmp(grid_type,'rtx')
    y_id=4;
    x_id=2;
    y_label='theta';
    x_label='t_x';
elseif strcmp(grid_type,'rty')
    y_id=4;
    x_id=3;
    y_label='theta';
    x_label='t_y';
elseif strcmp(grid_type,'rs')
    y_id=5;
    x_id=4;
    y_label='scale';
    x_label='theta';
elseif strcmp(grid_type,'shear')
    y_id=6;
    x_id=7;
    y_label='a';
    x_label='b';
elseif strcmp(grid_type,'proj')
    y_id=8;
    x_id=9;
    y_label='v1';
    x_label='v2';
elseif strcmp(grid_type,'trans2')
    y_id=2;
    x_id=3;
    y_label='t_x';
    x_label='t_y';
else
    error('Invalid grid type: %s', grid_type);
end


end

