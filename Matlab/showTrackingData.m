clear all;
[actors, sequences, challenges, tracker_types, filter_types,  inc_types, pw_opt_types, grid_types, appearance_models] = getParamLists();
dist_params=importdata('distanceGridParams.txt');
param_names=dist_params.textdata;
param_vals=dist_params.data;

actor_id = param_vals(strcmp(param_names,'actor_id'));
seq_id = param_vals(strcmp(param_names,'seq_id'));
challenge_id = param_vals(strcmp(param_names,'challenge_id'));
inc_id = param_vals(strcmp(param_names,'inc_id'));
grid_id = param_vals(strcmp(param_names,'grid_id')) ;
appearance_id = param_vals(strcmp(param_names,'appearance_id'));
tracker_id = param_vals(strcmp(param_names,'tracker_id'));
filter_id = param_vals(strcmp(param_names,'filter_id')) ;
kernel_size = param_vals(strcmp(param_names,'kernel_size')) ;
end_id = param_vals(strcmp(param_names,'end_id')) ;
pw_opt_id = param_vals(strcmp(param_names,'opt_id')) ;
pw_sel_opt = param_vals(strcmp(param_names,'selective_opt')) ;
pw_opt_dof = param_vals(strcmp(param_names,'dof')) ;
%show_img = param_vals(strcmp(param_names,'show_img'));
upd_template = param_vals(strcmp(param_names,'upd_template'));

plot_cols = 'rgbkcmy';
plot_markers = 's^od*xv>';
track_legend = {'tx', 'ty', 'theta', 'scale', 'a', 'b', 'v1', 'v2'};
show_legend = 0;
n_cols = length(plot_cols);
n_markers = length(plot_markers);

start_id = 1;
show_img = 1;

img_root_dir=sprintf('../Image Data');
tracking_root_dir='../Tracking Data';

speed_factor=1;

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

src_img_fname=sprintf('%s/%s', img_root_dir, seq_name);
if ~strcmp(filter_type,'none')
    src_img_fname=sprintf('%s_%s%d', src_img_fname, filter_type, kernel_size);
end
src_img_fname=sprintf('%s.bin', src_img_fname);

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

param_x=tracker_hom_params(:, 1);
param_y=tracker_hom_params(:, 2:end);
n_params=size(param_y, 2);
plot_params=ones(n_params, 1);
max_frame_id=max(param_x)-min(param_x)+1;

fprintf('start_id: %d\n', start_id);
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

param_width=850;
param_height=650;
param_fig=figure;
xlim([start_id max_frame_id])
fig_title=sprintf('%s_%s_%s_%s', inc_type, grid_type, seq_name, appearance_model);
set(param_fig, 'Position', [0, 50, param_width, param_height]);
set(param_fig, 'Name', fig_title);
%legend(gca, track_legend);

x_pos=0;
y_pos=500;
pb_height=30;
pb_width=50;
slider_height=80;
slider_width=20;
txt_height=15;
[slider_speed, txt_speed, pb_pause, pb_back, pb_next, pb_rewind, pb_exit] = createTrackerWidgets(param_fig,...
    x_pos, y_pos, pb_width, pb_height, slider_width, slider_height, txt_height);
set(param_fig,'toolbar','figure')

pause_exec=0;
end_exec=0;
frame_id=start_id;

last_frame_id=0;
frame_id_diff=1;

file_start_id = start_id;
file_end_id = max_frame_id;

while ~end_exec
    if frame_id<start_id
        pause_exec=1;
        set(pb_pause, 'String', 'Resume');
        frame_id=start_id;
    end
    if frame_id>max_frame_id
        pause_exec=1;
        %set(pb_pause, 'String', 'Resume');
        frame_id=max_frame_id;
    end
    %fprintf('end_exec: %d\n', end_exec);
    tic;
    
    if frame_id~=last_frame_id
        curr_frame_id=frame_id;
        fprintf('Frame:\t%4d ::\t', curr_frame_id);
        
        figure(param_fig);
        for i=1:n_params
            if plot_params(i)
                col_id = mod(i, n_cols);
                if i>=n_cols
                    col_id=col_id+1;
                end                
                marker_id = mod(i, n_markers);
                if i>=n_markers
                    marker_id=marker_id+1;
                end
                plot(param_x(start_id:curr_frame_id),...
                    param_y(start_id:curr_frame_id, i),...
                    'Color', plot_cols(col_id));                
                hold on;
            end
        end
        grid on;        
        xlim([start_id max_frame_id]);
        if show_legend  
            legend(gca, track_legend);
            set_legend=1;
        end
        hold off;
        
        if show_img
            fseek(img_fid, img_pos(curr_frame_id), 'bof');
            img_bin=fread(img_fid, [img_width img_height], 'uint8', 'a');
            img_bin=uint8(img_bin');
            set(0,'CurrentFigure',img_fig);
            imshow(img_bin);
            curr_corners_x=tracker_corners(curr_frame_id+1, [1, 3, 5, 7, 1]);
            curr_corners_y=tracker_corners(curr_frame_id+1, [2, 4, 6, 8, 2]);
            hold on, plot(curr_corners_x,curr_corners_y,'Color','r','LineWidth',2), hold off;
        end
        %pause(0.001);
        %getframe(surf_fig);
        
        time_intvl2=toc;
        
        fprintf('FPS:\t%10.8f\n', 1.0/time_intvl2);
        last_frame_id = curr_frame_id;
    end
    
    if ~pause_exec
        frame_id=frame_id+speed_factor*frame_id_diff;
    else
        figure(param_fig);
        legend(gca, track_legend);
        pause(0.01);
    end
end

if show_img
    fclose(img_fid);
end
disp('Exiting...');
