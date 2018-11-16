clear all;
getParamLists();

dist_params=importdata('../distanceGridParams.txt');
param_names=dist_params.textdata;
param_vals=dist_params.data;
actor_id = param_vals(strcmp(param_names,'actor_id'));
seq_id = param_vals(strcmp(param_names,'seq_id'));
challenge_id = param_vals(strcmp(param_names,'challenge_id'));
inc_id = param_vals(strcmp(param_names,'inc_id'));
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

img_root_dir=sprintf('../../Image Data');
dist_root_dir='../../Distance Data';
tracking_root_dir='../../Tracking Data';

grid_ids = [0, 1, 2, 3];
min_appearance_ids=[0, 1, 5, 7, 11];

speed_factor=1;
tracking_mode=1;
max_iters=100;

rotate_surf=0;
view_az = 0;
view_al = 0;
d_az = 5;
d_al = 5;
max_al=45;
min_al=-45;

surf_width=500;
surf_height=400;

if end_id<start_id
    end_id=start_id;
end

actor = actors{actor_id + 1};
seq_name = sequences{actor_id + 1}{seq_id + 1};
challenge = challenges{challenge_id + 1};
filter_type = filter_types{filter_id+1};
tracker_type=tracker_types{tracker_id+1};
pw_opt_type= pw_opt_types{pw_opt_id+1};
appearance_model= appearance_models{appearance_id+1};
inc_type = inc_types{inc_id+1};
grid_types = grid_type_list(grid_ids+1);
surf_count=length(grid_ids);

if strcmp(actor,'METAIO')
    seq_name=sprintf('%s_%s', seq_name, challenge);
end

%if upd_template
    %tracker_typee=sprintf('%su', tracker_type);
%end

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

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 16);

fprintf('img_root_dir: %s\n', img_root_dir);
fprintf('dist_dir: %s\n', dist_dir);

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
end

if show_img
    if ~exist(src_img_fname,'file')
        error('File containing the image data is not present:\n%s', src_img_fname);
    end
    img_fid=fopen(src_img_fname);
    img_width=fread(img_fid, 1, 'uint32', 'a');
    img_height=fread(img_fid, 1, 'uint32', 'a');
    if isempty(img_width) || isempty(img_height)
        error('Image data file is empty: \n%s', src_img_fname);
    end
    
    img_size=img_width*img_height;
    img_start_pos=ftell(img_fid);    
    img_fig=figure;
    set(img_fig, 'Position', [1200, 50, 800, 600]);
    set(img_fig, 'Name', seq_name);
end