clear all;
getParamLists();
dist_params=importdata('../distanceGridParams.txt');
param_names=dist_params.textdata;
param_vals=dist_params.data;
% seq_id = param_vals(strcmp(param_names,'seq_id'));
inc_id = param_vals(strcmp(param_names,'inc_id'));
grid_id = param_vals(strcmp(param_names,'grid_id')) ;
appearance_id = param_vals(strcmp(param_names,'appearance_id'));
tracker_id = param_vals(strcmp(param_names,'tracker_id'));
% start_id = param_vals(strcmp(param_names,'start_id')) ;
filter_id = param_vals(strcmp(param_names,'filter_id')) ;
kernel_size = param_vals(strcmp(param_names,'kernel_size')) ;
pw_opt_id = param_vals(strcmp(param_names,'opt_id')) ;
pw_opt_dof = param_vals(strcmp(param_names,'dof')) ;

use_mtf_data = 1;
data_id = 0;
db_root_dir=sprintf('../../Datasets');
mtf_diag_root_dir='../C++/MTF/log/diagnostics';

% ACTOR_IDS=(0 1 1 3 3 0)
% SEQ_IDS=(2 41 48 2 11 39)
% FRAME_IDS=(362 13 100 583 150 172)

actor_id = 3;
seq_id = 11;
update_type = 0;
opt_type = 0;
am_name = 'ncc';
am_name_disp='';
ssm_name = '2';
frame_gap = 0;
start_id = 0;
end_id = 0;
file_start_id = 0;
file_end_id = -1;
state_ids = 0;
use_inv_data = 0;
show_img = 0;
pause_exec = 1;
show_controls = 1;
plot_font_size = 24;


data_types={
    'Norm',...%0
    'StdJac',...%1
    'Std',...%2
    'Std2',...%3
    'Jacobian',...%4
    'Hessian' %5
    };

actor = actors{actor_id+1};

min_appearance_ids=[0, 1, 5, 7, 11];
tracking_mode=0;
max_iters=100;
speed_factor=1;

rotate_surf=0;
view_az = 0;
view_al = 0;
d_az = 5;
d_al = 5;
max_al=45;
min_al=-45;

if file_start_id<frame_gap
    file_start_id=frame_gap;
end

if end_id<start_id
    end_id=start_id;
end

seq_name = sequences{actor_id + 1}{seq_id + 1};
grid_type = grid_types{grid_id+1};
filter_type = filter_types{filter_id+1};
tracker_type=tracker_types{tracker_id+1};
pw_opt_type= pw_opt_types{pw_opt_id+1};
if use_mtf_data
    appearance_model = am_name;
else
    appearance_model = appearance_models{appearance_id+1};
end
inc_type = inc_types{inc_id+1};

if file_end_id<file_start_id
    db_n_frames=importdata(sprintf('%s/%s/n_frames.txt',db_root_dir, actor));
    n_frames=db_n_frames.data(strcmp(strtrim(db_n_frames.textdata),seq_name));
    file_end_id=n_frames-1;
    fprintf('Using file_end_id=%d\n', file_end_id);
end

use_pw=0;
if strcmp(tracker_type,'pw')
    use_pw=1;
    tracker_type=sprintf('pw_%s%d', pw_opt_type, pw_opt_dof);
end

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', plot_font_size);
set(0,'DefaultAxesFontWeight', 'bold');
    
fprintf('seq_name: %s\n', seq_name);
fprintf('grid_type: %s\n', grid_type);
fprintf('filter_type: %s\n', filter_type);
fprintf('inc_type: %s\n', inc_type);
fprintf('tracker_type: %s\n', tracker_type);
fprintf('appearance_model: %s\n', appearance_model);
fprintf('show_img: %d\n', show_img);
fprintf('use_pw: %d\n', use_pw);

if use_pw
    fprintf('pw_opt_type: %s\n', pw_opt_type);
end

opt_min=0;
if ~use_mtf_data && any(appearance_id==min_appearance_ids)
    opt_min=1;
end

src_img_dir=sprintf('../../Image Data');
if use_mtf_data
    dist_dir=sprintf('%s/%s/3D', mtf_diag_root_dir, seq_name);
    curr_data_type = data_types{data_id+1};
    if use_inv_data
        curr_data_type=sprintf('inv_%s', curr_data_type);
    end
    dist_fname_bin = sprintf('%s/%s_%s_%d_%s_%d_%d_%d.bin',...
        dist_dir, am_name, ssm_name, update_type, curr_data_type,...
        frame_gap, file_start_id, file_end_id);
else
    dist_root_dir='Distance Data';
    dist_dir=sprintf('%s/%s_%s', dist_root_dir, seq_name, appearance_model);
    src_img_fname_bin=sprintf('%s/%s', src_img_dir, seq_name);
    if ~strcmp(filter_type,'none')
        dist_dir=sprintf('%s_%s%d', dist_dir, filter_type, kernel_size);
        src_img_fname_bin=sprintf('%s_%s%d', src_img_fname_bin, filter_type, kernel_size);
    end
    src_img_fname_bin=sprintf('%s.bin', src_img_fname_bin);
    dist_fname_bin=sprintf('%s/%s_%s_%s.bin', dist_dir, tracker_type, inc_type, grid_type);
end

src_img_fname_bin=sprintf('%s/%s', src_img_dir, seq_name);
if ~strcmp(filter_type,'none')
    src_img_fname_bin=sprintf('%s_%s%d', src_img_fname_bin, filter_type, kernel_size);
end
src_img_fname_bin=sprintf('%s.bin', src_img_fname_bin);

fprintf('src_img_dir: %s\n', src_img_dir);
fprintf('dist_dir: %s\n', dist_dir);
fprintf('dist_fname_bin: %s\n', dist_fname_bin);

if ~exist(dist_fname_bin,'file')
    error('File containing the distance data is not present:\n%s', dist_fname_bin);
end

[x_label, y_label, x_id, y_id] = getAxisLabelsAndIDs(grid_type);

surf_fig=figure;
fig_title=sprintf('%s_%s_%s_%s', inc_type, grid_type, seq_name, appearance_model);
set(surf_fig, 'Position', [0, 50, 850, 650]);
set(surf_fig, 'Name', fig_title);



if show_controls
    control_fig=figure;
    p_control_fig=get(control_fig);
    set(control_fig,'name','Control','numbertitle','off');
%     x_pos=0;
%     y_pos=500;
%     pb_height=30;
%     pb_width=50;
%     slider_height=80;
%     slider_width=20;
%     txt_height=15;
    x_pos=10;
    y_pos=200;
    pb_height=30;
    pb_width=50;
    slider_height=80;
    slider_width=20;
    txt_height=15;
    [slider_speed, txt_speed, pb_pause, pb_back, pb_next, pb_rewind, tb_rot, pb_exit] = createWidgets(control_fig,...
        x_pos, y_pos, pb_width, pb_height, slider_width, slider_height, txt_height);

    set(control_fig,'toolbar','figure');
    set (control_fig, 'Position', [50,100,200,300]);

%     set(surf_fig,'toolbar','figure');
end
psurf_fig=get(surf_fig);
rotate3d(surf_fig);

dist_fid=fopen(dist_fname_bin);
file_end_id=fread(dist_fid, 1 , 'uint32', 'a');
file_start_id=fread(dist_fid, 1 , 'uint32', 'a');
surf_x_len=fread(dist_fid, 1 , 'uint32', 'a');
surf_y_len=fread(dist_fid, 1 , 'uint32', 'a');
surf_x=fread(dist_fid, surf_x_len , 'float64', 'a');
surf_y=fread(dist_fid, surf_y_len , 'float64', 'a');

if start_id<file_start_id
    fprintf('start_id is smaller than file_start_id: %d\n', file_start_id);
    start_id=file_start_id;
end
if end_id>file_end_id
    fprintf('end_id is larger than the number of frames in the distance data file: %d\n', file_end_id);
    end_id=file_end_id;
end
if tracking_mode
    if use_pw
        tracker_hom_fname=sprintf('Tracking Data/%s_%s/%s_%s_params_inv.txt', seq_name, appearance_model, tracker_type, inc_type);
        tracker_corners_fname=sprintf('Tracking Data/%s_%s/%s_%s_corners.txt', seq_name, appearance_model, tracker_type, inc_type);
    else
        tracker_hom_fname=sprintf('Tracking Data/%s_%s/%s_params_inv.txt', seq_name, appearance_model, tracker_type);
        tracker_corners_fname=sprintf('Tracking Data/%s_%s/%s_corners.txt', seq_name, appearance_model, tracker_type);
    end
    
    
    fprintf('tracker_hom_fname: %s\n', tracker_hom_fname);
    fprintf('tracker_corners_fname: %s\n', tracker_corners_fname);
    
    tracker_hom_params=importdata(tracker_hom_fname);
    [x_vals, y_vals, n_iters, pt_cols, max_frame_id] = getPointsToPlot(tracker_hom_params,...
        max_iters, x_id, y_id);
end

fprintf('file_end_id: %d\n', file_end_id);
fprintf('file_start_id: %d\n', file_start_id);
fprintf('surf_x_len: %d\n', surf_x_len);
fprintf('surf_y_len: %d\n', surf_y_len);

grid_start_pos = ftell(dist_fid);
dist_grid_size=surf_x_len*surf_y_len*8;
fseek(dist_fid, dist_grid_size*(start_id-file_start_id), 'cof');

if show_img
    if ~exist(src_img_fname_bin,'file')
        error('File containing the image data is not present:\n%s', src_img_fname_bin);
    end
    if tracking_mode
        tracker_corners=importdata(tracker_corners_fname);
        tracker_corners=tracker_corners.data;
        no_of_frames=size(tracker_corners, 1)-1;
    end
    getAnnotatedImages;
    img_fig=figure;
    set(img_fig, 'Position', [1000, 50, 800, 600]);
    set(img_fig, 'Name', seq_name);
end
end_exec=0;
frame_id=start_id;
if use_mtf_data
    if isempty(am_name_disp)
        am_name_disp = upper(appearance_model);       
    end
     plot_title_templ = sprintf('%s %s', am_name_disp, seq_name);
else
    plot_title_templ=sprintf('%s_%s_%s_%s_%s%d_%s', inc_type, grid_type, seq_name,...
        appearance_model, filter_type, kernel_size, tracker_type);
end

last_frame_id=0;
frame_id_diff=1;
while ~end_exec
    if frame_id<start_id
        pause_exec=1;
        if show_controls
            set(pb_pause, 'String', 'Resume');
        end
        frame_id=start_id;
    end
    if frame_id>file_end_id
        pause_exec=1;
        if show_controls
            set(pb_pause, 'String', 'Resume');
        end
        frame_id=file_end_id;
    end
    if frame_id==file_end_id
        fseek(dist_fid,0, 'bof');
        file_end_id=fread(dist_fid, 1 , 'uint32', 'a');
    end
    %fprintf('end_exec: %d\n', end_exec);
    tic;
    
    if frame_id~=last_frame_id
        curr_frame_id=frame_id;
        curr_grid_pos=grid_start_pos + dist_grid_size*(curr_frame_id-file_start_id);
        fseek(dist_fid,curr_grid_pos, 'bof');
        surf_z=fread(dist_fid, [length(surf_x) length(surf_y)], 'float64', 'a');
        surf_z=surf_z';
        figure(surf_fig), surf_handle=surf(surf_x, surf_y, surf_z,...
            'FaceColor', 'interp', 'FaceAlpha', 0.5);
        
        time_intvl=toc;
        
        if ~pause_exec && rotate_surf
            view(view_az, view_al);
            view_az=view_az-d_az;
            view_al=view_al+d_al;
            if view_al>=max_al
                view_al=max_al;
                d_al=-d_al;
            elseif view_al<=min_al
                view_al=min_al;
                d_al=-d_al;
            end
        end
        
        xlabel(x_label);
        ylabel(y_label);
        zlabel(sprintf('f_{%s}', lower(am_name_disp)));
        if use_mtf_data
            plot_title=sprintf('%s frame %d', plot_title_templ, curr_frame_id);
        else
            plot_title=sprintf('%s_%d_%d', plot_title_templ, curr_frame_id, speed_factor);
        end        
        title(plot_title, 'interpreter','none');
        
        fprintf('Frame:\t%4d ::\t', curr_frame_id);
        
        if tracking_mode && curr_frame_id<=max_frame_id
            hold on;
            for i=1:n_iters(curr_frame_id)
                z_val=interp2(surf_x, surf_y, surf_z, x_vals(curr_frame_id, i), y_vals(curr_frame_id, i));
                plot3(x_vals(curr_frame_id, i),y_vals(curr_frame_id, i),z_val,...
                    'o','markerfacecolor',pt_cols(curr_frame_id, i, :),'markersize',5);
            end
            fprintf('n_iter:\t%d\t', n_iters(curr_frame_id));
            hold off;
        end
        
        if show_img
            img_bin=img_mat{curr_frame_id};
            set(0,'CurrentFigure',img_fig);
            imshow(img_bin);
            if tracking_mode
                curr_corners_x=tracker_corners(curr_frame_id+1, [1, 3, 5, 7, 1]);
                curr_corners_y=tracker_corners(curr_frame_id+1, [2, 4, 6, 8, 2]);
                hold on, plot(curr_corners_x,curr_corners_y,'Color','r','LineWidth',2), hold off;
            end
        end
        time_intvl2=toc;
        
        if opt_min
            surf_z(surf_z<0)=inf;
            [opt_row_val, opt_row_ids]=min(surf_z);
            [opt_col_val, opt_col_id]=min(opt_row_val);
        else
            [opt_row_val, opt_row_ids]=max(surf_z);
            [opt_col_val, opt_col_id]=max(opt_row_val);
        end
        opt_row_id=opt_row_ids(opt_col_id);
        
        time_intvl3=toc;
        fprintf('OptDist:\t%12.6f\t at:\t (%3d, %3d)\t with\t %s:\t %8.5f\t and\t %s:\t %8.5f\t',...
            opt_col_val, opt_row_id, opt_col_id, x_label, surf_x(opt_col_id),...
            y_label, surf_y(opt_row_id));
        fprintf('FPS:\t%10.8f\tFPS2:\t%10.8f\tFPS3:\t%10.8f\n',...
            1.0/time_intvl, 1.0/time_intvl2, 1.0/time_intvl3);
        last_frame_id=curr_frame_id;
    end
    
    if ~pause_exec
        frame_id=frame_id+speed_factor*frame_id_diff;
    else
        pause(0.1);
    end
end

fclose(dist_fid);
% if show_img
%     fclose(img_fid);
% end
disp('Exiting...');
% if close_figs_on_exit
%     close all;
% end


