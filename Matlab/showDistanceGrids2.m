function showDistanceGrids2(seq_id, grid_id, appearance_id, start_id, end_id, tracker_id, inc_type, filter_id, kernel_size)
evalin('base','clear');
appearance_models = {
    'ssd',...%0
    'mi',...%1
    'ncc',...%2
    'scv',...%3
    'mi2',...%4
    'ncc2'...%5
    };
tracker_types = {
    'gt',...%0
    'esm',...%1
    'ic',...%2
    'nnic',...%3
    'pf'%4
    };
filter_types = {
    'none',...%0
    'gauss',...%1
    'box',...%2
    'norm_box',...%3
    'bilateral',...%4
    'median',...%5
    'gabor',...%6
    'sobel',...%7
    'scharr',...%8
    'LoG',...%9
    'DoG',...%10
    'laplacian',...%11
    'canny'%12
    };
pw_opt_types = {
    'pre',...%0
    'post',...%1
    'independent'%2
    };
grid_types = {
    'trans',...%0
    'rs',...%1
    'shear',...%2
    'proj',...%3
    'rtx',...%4
    'rty',...%5
    'stx',...%6
    'sty'%7
    };
sequences = {
    'nl_bookI_s3',...%0
    'nl_bookII_s3',...%1
    'nl_bookIII_s3',...%2
    'nl_cereal_s3',...%3
    'nl_juice_s3',...%4
    'nl_mugI_s3',...%5
    'nl_mugII_s3',...%6
    'nl_mugIII_s3',...%7
    'nl_bookI_s4',...%8
    'nl_bookII_s4',...%9
    'nl_bookIII_s4',...%10
    'nl_cereal_s4',...%11
    'nl_juice_s4',...%12
    'nl_mugI_s4',...%13
    'nl_mugII_s4',...%14
    'nl_mugIII_s4',...%15
    'nl_bus',...%16
    'nl_highlighting',...%17
    'nl_letter',...%18
    'nl_newspaper'...%19
    };
arg_id=9;
if nargin<arg_id
    kernel_size = 9 ;
    arg_id=arg_id-1;
end
if nargin<arg_id
    filter_id = 0 ;
    arg_id=arg_id-1;
end
if nargin<arg_id
    inc_type = 'ic' ;
    arg_id=arg_id+1;
end
if nargin<arg_id
    tracker_id = 0 ;
    arg_id=arg_id-1;
end
if nargin<arg_id
    end_id = 500 ;
    arg_id=arg_id-1;
end
if nargin<arg_id
    start_id = 1 ;
    arg_id=arg_id-1;
end
if nargin<arg_id
    appearance_id = 10 ;
    arg_id=arg_id-1;
end
if nargin<arg_id
    grid_id = 10 ;
    arg_id=arg_id-1;
end
if nargin<arg_id
    seq_id = 0 ;
    arg_id=arg_id-1;
end
% grid_id = 0 ;
% appearance_id = 3;
% filter_id = 0 ;
% start_id = 1 ;
% end_id = 500 ;
% kernel_size = 10 ;
% tracker_id = 0;
% inc_type = 'fc' ;

use_pw = 0 ;
pw_opt_id = 1 ;

view_az = 0;
view_al = 0;
d_az = 5;
d_al = 5;
max_al=45;
opt_al=-45;

max_iters=100;

seq_name = sequences{seq_id+1};
grid_type = grid_types{grid_id+1};
filter_type = filter_types{filter_id+1};
tracker_type=tracker_types{tracker_id+1};
pw_opt_type= pw_opt_types{pw_opt_id+1};
appearance_model= appearance_models{appearance_id+1};

fprintf('seq_name: %s\n', seq_name);
fprintf('grid_type: %s\n', grid_type);
fprintf('filter_type: %s\n', filter_type);
fprintf('tracker_type: %s\n', tracker_type);
fprintf('pw_opt_type: %s\n', pw_opt_type);
fprintf('appearance_model: %s\n', appearance_model);


tracker_hom_params=[];
tracker_corners=[];

img_width = 800;
img_height = 600;

tracking_mode=1;

hom_y_id=2;
hom_x_id=3;

if ~strcmp(tracker_type,'gt')
    tracking_mode=1;
    use_pw = 0 ;
    tracker_hom_params=importdata(sprintf('Tracking Result/%s_params_inv.txt', seq_name));
    tracker_corners=importdata(sprintf('Tracking Result/%s_%s.txt', seq_name, tracker_type));
    tracker_corners=tracker_corners.data;
    n_params=size(tracker_hom_params, 1);
    
    if strcmp(grid_type, 'trans')
        hom_y_id=2;
        hom_x_id=3;
    elseif strcmp(grid_type, 'rs')
        hom_y_id=5;
        hom_x_id=4;
    elseif strcmp(grid_type, 'shear')
        hom_y_id=6;
        hom_x_id=7;
    elseif strcmp(grid_type, 'proj')
        hom_y_id=8;
        hom_x_id=9;
    end
end

if strcmp(filter_type,'none')
    if use_pw
        dist_dir=sprintf('Distance Data/#PW/%s/%s_%s_%s', seq_name, inc_type, pw_opt_type, grid_type);
    else
        dist_dir=sprintf('Distance Data/%s_%s_%s/%s_%s', seq_name, appearance_model, tracker_type, inc_type, grid_type);
    end
    src_img_dir=sprintf('Image Data/%s', seq_name);
    dist_plot_dir=sprintf('Distance Plots/%s_%s_%s/%s_%s', seq_name, appearance_model, tracker_type, inc_type, grid_type);
else
    if use_pw
        dist_dir=sprintf('Distance Data/#PW/%s_%s%d/%s_%s_%s', seq_name, filter_type, kernel_size, inc_type, pw_opt_type, grid_type);
    else
        dist_dir=sprintf('Distance Data/%s_%s_%s_%s%d/%s_%s', seq_name, appearance_model, tracker_type, filter_type, kernel_size, inc_type, grid_type);
    end
    src_img_dir=sprintf('Image Data/%s_%s%d', seq_name, filter_type, kernel_size);
    dist_plot_dir=sprintf('Distance Plots/%s_%s_%s_%s%d/%s_%s', seq_name, appearance_model, tracker_type, filter_type, kernel_size, inc_type, grid_type);
end

fprintf('src_img_dir: %s\n', src_img_dir);
fprintf('dist_dir: %s\n', dist_dir);
fprintf('dist_plot_dir: %s\n', dist_plot_dir);

if ~exist(dist_plot_dir,'dir')
    mkdir(dist_plot_dir)
end

if ~exist(dist_dir,'dir')
    error('Folder containing the distance files is not present:\n%s', dist_dir);
end

if ~exist(src_img_dir,'dir')
    error('Folder containing the image files is not present:\n%s', src_img_dir);
end

if strcmp(grid_type,'trans')
    ytype='tx';
    xtype='ty';
elseif strcmp(grid_type,'rtx')
    ytype='theta';
    xtype='tx';
elseif strcmp(grid_type,'rty')
    ytype='theta';
    xtype='ty';
elseif strcmp(grid_type,'rs')
    ytype='scale';
    xtype='theta';
elseif strcmp(grid_type,'shear')
    ytype='a';
    xtype='b';
elseif strcmp(grid_type,'proj')
    ytype='v1';
    xtype='v2';
else
    error('Invalid grid type: %s', grid_type);
end

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 16);

%ssd_fig=figure;
ssd_fig=figure;
pssd_fig=get(ssd_fig);

fig_title=sprintf('%s_%s_%s_%s', inc_type, grid_type, seq_name, appearance_model);
set(ssd_fig, 'Position', [0, 50, 850, 650]);
set(ssd_fig, 'Name', fig_title);
%set(ssd_fig, 'KeyPressFcn', @figCallback);

rotate3d(ssd_fig);

img_fig=figure;
set(img_fig, 'Position', [1000, 50, 800, 600]);
set(img_fig, 'Name', seq_name);

surf_x=importdata(sprintf('%s/%s_vec.txt', dist_dir, xtype));
surf_y=importdata(sprintf('%s/%s_vec.txt', dist_dir, ytype));

hom_params_id=1;
if tracking_mode
    while(tracker_hom_params(hom_params_id, 1)<start_id+1)
        hom_params_id=hom_params_id+1;
    end
end

for frame_id=start_id:end_id
    
    %fprintf('frame_id: %d\n', frame_id);
    %curr_img_gs=importdata(sprintf('Image Data/%s/curr_img_gs_%d.txt', seq_name, frame_id));
    
    dist_fname_bin=sprintf('%s/dist_grid_%d.bin', dist_dir, frame_id);
    dist_fid=fopen(dist_fname_bin);
    surf_z=fread(dist_fid, [length(surf_x) length(surf_y)], 'float64', 'a');
    surf_z=surf_z';
    fclose(dist_fid);
    
    src_img_fname_bin=sprintf('%s/frame_%d_gs.bin', src_img_dir, frame_id);
    img_fid=fopen(src_img_fname_bin);
    img_bin=fread(img_fid, [img_width img_height], 'uint8', 'a');
    img_bin=uint8(img_bin');
    fclose(img_fid);
    
    
    figure(img_fig), imshow(img_bin);
    if tracking_mode
        curr_corners_x=tracker_corners(frame_id+1, [1, 3, 5, 7, 1]);
        curr_corners_y=tracker_corners(frame_id+1, [2, 4, 6, 8, 2]);
        figure(img_fig), hold on, plot(curr_corners_x,curr_corners_y,'Color','r','LineWidth',2), hold off;
    end
    
    figure(ssd_fig), surf(surf_x, surf_y, surf_z,...
        'FaceColor', 'interp', 'FaceAlpha', 0.5);
    
    fprintf('Frame: %4d :: ', frame_id);
    if tracking_mode
        fprintf('hom_params_id: %d param_frame_id: %d\t', hom_params_id, tracker_hom_params(hom_params_id, 1));
        n_iter=0;
        hold on;
        while(hom_params_id<=n_params && tracker_hom_params(hom_params_id, 1)==frame_id+1)
            n_iter=n_iter+1;
            y_val=tracker_hom_params(hom_params_id, hom_y_id);
            x_val=tracker_hom_params(hom_params_id, hom_x_id);
            z_val=interp2(surf_x, surf_y, surf_z, x_val, y_val);
            pt_col=[n_iter/max_iters, n_iter/max_iters, n_iter/max_iters];
            plot3(x_val,y_val,z_val,'o','markerfacecolor',pt_col,'markersize',5);
            hom_params_id=hom_params_id+1;
        end
        fprintf('n_iter: %d\t', n_iter);
        hold off;
    end
    view(view_az, view_al);
    view_az=view_az-d_az;
    view_al=view_al+d_al;
    if view_al>=max_al
        view_al=max_al;
        d_al=-d_al;
    elseif view_al<=opt_al
        view_al=opt_al;
        d_al=-d_al;
    end
    
    xlabel(ytype);
    ylabel(xtype);
    plot_title=sprintf('%s_%s_%s_%s_%s%d_%s_%d', inc_type, grid_type, seq_name,...
        appearance_model, filter_type, kernel_size, tracker_type, frame_id);
    title(plot_title, 'interpreter','none');
    getframe(ssd_fig);
    
    grid_img_fname=sprintf('%s/frame_%d.fig', dist_plot_dir, frame_id);
    saveas(ssd_fig, grid_img_fname)
    
    if appearance_id==1 || appearance_id==2 || appearance_id==3
        [opt_row_val, opt_row_ids]=max(surf_z);
        [opt_col_val, opt_col_id]=max(opt_row_val);
    else
        surf_z(surf_z<0)=inf;
        [opt_row_val, opt_row_ids]=min(surf_z);
        [opt_col_val, opt_col_id]=min(opt_row_val);
    end
    opt_row_id=opt_row_ids(opt_col_id);
    
    fprintf('OptDist:\t%12.6f\t at:\t (%3d, %3d)\t with\t %s:\t %8.5f\t and\t %s:\t %8.5f\n',...
        opt_col_val, opt_row_id, opt_col_id, xtype, surf_x(opt_col_id),...
        ytype, surf_y(opt_row_id));
end

W = who;
putvar(W{:})

end

