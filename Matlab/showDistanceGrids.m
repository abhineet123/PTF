clear all;
getParamLists();
dist_params=importdata('../distanceGridParams.txt');
param_names=dist_params.textdata;
param_vals=dist_params.data;

actor_id = param_vals(strcmp(param_names,'actor_id'));
seq_id = param_vals(strcmp(param_names,'seq_id'));
challenge_id = param_vals(strcmp(param_names,'challenge_id'));
inc_id = param_vals(strcmp(param_names,'inc_id'));
grid_id = param_vals(strcmp(param_names,'grid_id')) ;
appearance_id = param_vals(strcmp(param_names,'appearance_id'));
tracker_id = param_vals(strcmp(param_names,'tracker_id'));
start_id = param_vals(strcmp(param_names,'start_id')) ;
filter_id = param_vals(strcmp(param_names,'filter_id')) ;
kernel_size = param_vals(strcmp(param_names,'kernel_size')) ;
end_id = param_vals(strcmp(param_names,'end_id')) ;
pw_opt_id = param_vals(strcmp(param_names,'opt_id')) ;
pw_sel_opt = param_vals(strcmp(param_names,'selective_opt')) ;
pw_opt_dof = param_vals(strcmp(param_names,'dof')) ;
show_img = param_vals(strcmp(param_names,'show_img'));
upd_template = param_vals(strcmp(param_names,'upd_template'));

img_root_dir=sprintf('../../Image Data');
dist_root_dir='../../Distance Data';
tracking_root_dir='../../Tracking Data';

min_appearance_ids=[0, 1, 5, 7, 11, 12];
speed_factor=1;
tracking_mode=0;
max_iters=100;

rotate_surf=0;
view_az = 0;
view_al = 0;
d_az = 5;
d_al = 5;
max_al=45;
min_al=-45;

surf_width=850;
surf_height=650;

if end_id<start_id
    end_id=start_id;
end

actor = actors{actor_id + 1};
seq_name = sequences{actor_id + 1}{seq_id + 1};
challenge = challenges{challenge_id + 1};
grid_type = grid_types{grid_id+1};
filter_type = filter_types{filter_id+1};
tracker_type=tracker_types{tracker_id+1};
pw_opt_type= pw_opt_types{pw_opt_id+1};
appearance_model= appearance_models{appearance_id+1};
inc_type = inc_types{inc_id+1};

if strcmp(actor,'METAIO')
    seq_name=sprintf('%s_%s', seq_name, challenge);
end

if upd_template
    tracker_typee=sprintf('%su', tracker_type);
end

use_pw=0;
if strcmp(tracker_type,'pw') || strcmp(tracker_type,'pwu')
    use_pw=1;
    tracker_type=sprintf('%s%d_%s%d', tracker_type, pw_sel_opt, pw_opt_type, pw_opt_dof);
elseif strcmp(tracker_type,'ppw') || strcmp(tracker_type,'ppwu')
    use_pw=1;
    tracker_type=sprintf('%s%d_%s', tracker_type, pw_sel_opt, pw_opt_type);
end

fprintf('actor: %s\n', actor);
fprintf('seq_name: %s\n', seq_name);
fprintf('challenge: %s\n', challenge);
fprintf('grid_type: %s\n', grid_type);
fprintf('filter_type: %s\n', filter_type);
fprintf('inc_type: %s\n', inc_type);
fprintf('tracker_type: %s\n', tracker_type);
fprintf('appearance_model: %s\n', appearance_model);
fprintf('show_img: %d\n', show_img);
fprintf('use_pw: %d\n', use_pw);

if use_pw
    fprintf('pw_opt_type: %s\n', pw_opt_type);
    fprintf('pw_sel_opt: %s\n', pw_sel_opt);
end

opt_min=0;
if any(appearance_id==min_appearance_ids)
    opt_min=1;
end

dist_dir=sprintf('%s/%s_%s', dist_root_dir, appearance_model, seq_name);
src_img_fname=sprintf('%s/%s', img_root_dir, seq_name);
if ~strcmp(filter_type,'none')
    dist_dir=sprintf('%s_%s%d', dist_dir, filter_type, kernel_size);
    src_img_fname=sprintf('%s_%s%d', src_img_fname, filter_type, kernel_size);
end
src_img_fname=sprintf('%s.bin', src_img_fname);
if strcmp(pw_opt_type, 'ind')
    dist_fname=sprintf('%s/gt_%s_%s.bin', dist_dir, inc_type, grid_type);
else
    dist_fname=sprintf('%s/%s_%s_%s.bin', dist_dir, tracker_type, inc_type, grid_type);
end
fprintf('img_root_dir: %s\n', img_root_dir);
fprintf('dist_dir: %s\n', dist_dir);
fprintf('dist_fname: %s\n', dist_fname);

if ~exist(dist_fname,'file')
    error('Distance data file is not present:\n%s', dist_fname);
end

dist_fid=fopen(dist_fname);
file_end_id=fread(dist_fid, 1 , 'uint32', 'a');
file_start_id=fread(dist_fid, 1 , 'uint32', 'a');
if isempty(file_end_id) || isempty(file_start_id)
    error('Distance data file is empty: \n%s', dist_fname);
end
surf_x_len=fread(dist_fid, 1 , 'uint32', 'a');
surf_y_len=fread(dist_fid, 1 , 'uint32', 'a');
surf_x=fread(dist_fid, surf_x_len , 'float64', 'a');
surf_y=fread(dist_fid, surf_y_len , 'float64', 'a');
len_x=length(surf_x);
len_y=length(surf_y);

if start_id<file_start_id
    fprintf('start_id is smaller than file_start_id: %d\n', file_start_id);
    start_id=file_start_id;
end
if end_id>file_end_id
    fprintf('end_id is larger than the number of frames in the distance data file: %d\n', file_end_id);
    end_id=file_end_id;
end

grid_start_pos = ftell(dist_fid);
dist_grid_size=surf_x_len*surf_y_len*8;
fseek(dist_fid, dist_grid_size*(start_id-file_start_id), 'cof');
[x_label, y_label, x_id, y_id] = getAxisLabelsAndIDs(grid_type);

if tracking_mode
    if use_pw
        tracker_hom_fname=sprintf('%s/%s_%s/%s_%s_params_inv.txt', tracking_root_dir, appearance_model, seq_name, tracker_type, inc_type);
        tracker_corners_fname=sprintf('%s/%s_%s/%s_%s_corners.txt', tracking_root_dir, appearance_model, seq_name, tracker_type, inc_type);
    else
        tracker_hom_fname=sprintf('%s/%s_%s/%s_params_inv.txt', tracking_root_dir, appearance_model, seq_name, tracker_type);
        tracker_corners_fname=sprintf('%s/%s_%s/%s_corners.txt', tracking_root_dir, appearance_model, seq_name, tracker_type);
    end
    if ~exist(tracker_hom_fname,'file')
        error('File containing the tracking params is not present:\n%s', tracker_hom_fname);
    end
    if ~exist(tracker_corners_fname,'file')
        error('File containing the tracking corners is not present:\n%s', tracker_corners_fname);
    end
    
    fprintf('tracker_hom_fname: %s\n', tracker_hom_fname);
    fprintf('tracker_corners_fname: %s\n', tracker_corners_fname);
    
    tracker_hom_params=importdata(tracker_hom_fname);
    tracker_corners=importdata(tracker_corners_fname);
    tracker_corners=tracker_corners.data;
    
    [x_vals, y_vals, n_iters, pt_cols, max_frame_id] = getPointsToPlot(tracker_hom_params,...
        max_iters, x_id, y_id);
end
if ~exist('max_frame_id', 'var')
    max_frame_id=1500;
end
dist_grid_pos=zeros(max_frame_id, 1);
for frame_id=1:max_frame_id
    dist_grid_pos(frame_id)=grid_start_pos + dist_grid_size*(frame_id-file_start_id);
end

fprintf('file_end_id: %d\n', file_end_id);
fprintf('file_start_id: %d\n', file_start_id);
fprintf('surf_x_len: %d\n', surf_x_len);
fprintf('surf_y_len: %d\n', surf_y_len);
fprintf('max_frame_id: %d\n', max_frame_id);

if show_img
    if ~exist(src_img_fname,'file')
        error('Image data file is not present:\n%s', src_img_fname);
    end
    img_fid=fopen(src_img_fname);
    img_width=fread(img_fid, 1, 'uint32', 'a');
    img_height=fread(img_fid, 1, 'uint32', 'a');
    if isempty(img_width) || isempty(img_height)
        error('Image data file is empty: \n%s', src_img_fname);
    end
    img_size=img_width*img_height;
    img_start_pos=ftell(img_fid);
    
    img_pos=zeros(max_frame_id, 1);
    for frame_id=1:max_frame_id
        img_pos(frame_id)=img_start_pos + img_size*(frame_id-1);
    end
    
    img_fig=figure;
    set(img_fig, 'Position', [1000, 50, 800, 600]);
    set(img_fig, 'Name', seq_name);
end

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 16);
surf_fig=figure;
fig_title=sprintf('%s_%s_%s_%s', inc_type, grid_type, seq_name, appearance_model);
set(surf_fig, 'Position', [0, 50, surf_width, surf_height]);
set(surf_fig, 'Name', fig_title);
x_pos=0;
y_pos=500;
pb_height=30;
pb_width=50;
slider_height=80;
slider_width=20;
txt_height=15;
[slider_speed, txt_speed, pb_pause, pb_back, pb_next, pb_rewind, tb_rot, pb_exit] = createWidgets(surf_fig,...
    x_pos, y_pos, pb_width, pb_height, slider_width, slider_height, txt_height);
set(surf_fig,'toolbar','figure')
psurf_fig=get(surf_fig);
rotate3d(surf_fig);

pause_exec=0;
end_exec=0;
frame_id=start_id;
% surf_z=zeros(surf_y_len, surf_x_len);
% surf_handle=surf(surf_x, surf_y, surf_z,...
%     'FaceColor', 'interp', 'FaceAlpha', 0.5);
% tracking_mode=0;
plot_title_templ=sprintf('%s_%s_%s_%s_%s%d_%s', inc_type, grid_type, seq_name,...
    appearance_model, filter_type, kernel_size, tracker_type);
last_frame_id=0;
frame_id_diff=1;

%---------------------------------------------------------------------%
%---------------------------Start Animation---------------------------%
%---------------------------------------------------------------------%
while ~end_exec
    if frame_id<start_id
        pause_exec=1;
        set(pb_pause, 'String', 'Resume');
        frame_id=start_id;
    end
    if frame_id>file_end_id
        pause_exec=1;
        set(pb_pause, 'String', 'Resume');
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
        %fprintf('frame_id: %d\n', frame_id);
        %curr_img_gs=importdata(sprintf('Image Data/%s/curr_img_gs_%d.txt', seq_name, frame_id));
        %curr_grid_pos=grid_start_pos + dist_grid_size*(curr_frame_id-file_start_id);
        fseek(dist_fid,dist_grid_pos(curr_frame_id), 'bof');
        surf_z=fread(dist_fid, [len_x len_y], 'float64', 'a');
        surf_z=surf_z';
        
        %set(surf_handle, 'ZData', surf_z);
        
        %set(0,'CurrentFigure',surf_fig);
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
        plot_title=sprintf('%s_%d_%d', plot_title_templ, curr_frame_id, speed_factor);
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
            fseek(img_fid, img_pos(curr_frame_id), 'bof');
            img_bin=fread(img_fid, [img_width img_height], 'uint8', 'a');
            img_bin=uint8(img_bin');
            set(0,'CurrentFigure',img_fig);
            imshow(img_bin);
            if tracking_mode
                curr_corners_x=tracker_corners(curr_frame_id+1, [1, 3, 5, 7, 1]);
                curr_corners_y=tracker_corners(curr_frame_id+1, [2, 4, 6, 8, 2]);
                hold on, plot(curr_corners_x,curr_corners_y,'Color','r','LineWidth',2), hold off;
            end
        end
        pause(0.001);
        %drawnow;
        
        %getframe(surf_fig);
        %grid_img_fname=sprintf('%s/frame_%d.fig', dist_plot_dir, curr_frame_id);
        %saveas(surf_fig, grid_img_fname)
        
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
if show_img
    fclose(img_fid);
end
disp('Exiting...');
