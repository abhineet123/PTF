% close all;
clear all;
colRGBDefs;
getParamLists;

db_root_dir=sprintf('../../Datasets');
img_root_dir=sprintf('../../Image Data');
root_dir='../C++/MTF/log';

ACTOR_IDS=[3, 0, 1, 1, 3, 3, 0];
SEQ_IDS=[11, 2, 41, 48, 2, 11, 39];
FRAME_IDS=[1, 362, 13, 100, 583, 150, 172];

config_id = 0;

actor_id = 3;
seq_id = 8;
start_id = 1;
update_type = 0;
opt_type = 0;
out_prefix = '';
out_prefix = 'ccre24b100a1b';
am_name = 'CCRE/100';
am_name_disp = '';
ilm_name = '0';
ssm_name = '2';
frame_gap = 0;
file_start_id = 0;
file_end_id = 0;
state_ids = 0;
use_inv_data = 0;
show_img = 0;
pause_exec = 1;
speed_factor = 1;

plot_only_norm = 1;
plot_norm_in_one_fig = 1;
plot_feat_norm = 0;
plot_likelihood = 1;
plot_num_likelihood = 0;
likelihood_alpha = 100;
likelihood_beta = 1;
likelihood_type = 1;
invert_feat_norm = 1;

plot_only_jac = 0;
plot_num_jac = 0;

plot_only_hess = 1;
plot_num_hess = 1;
plot_misc_hess = 0;
plot_sec_ord_hess = 1;
plot_font_size = 24;

normalized_fig = 0;

if config_id>0
    actor_id = ACTOR_IDS(config_id);
    seq_id = SEQ_IDS(config_id);
    start_id = FRAME_IDS(config_id);
end

% start_id = start_id + 1;
file_start_id = file_start_id + 1;
file_end_id = file_end_id + 1;

if plot_only_norm
    plot_only_jac = 0;
    plot_only_hess = 0;
end

if plot_only_jac
    plot_only_hess = 0;
end
if isempty(am_name_disp)
    am_name_disp = am_name;       
end
set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', plot_font_size);
set(0,'DefaultAxesFontWeight', 'bold');

plot_cols_123={
    'red',...
    'green',...
    'blue',...
    'cyan',...
    'magenta',...
    'purple',...
    'forest_green',...
    'slate_gray'
};
plot_legend_123={
    't_x',...
    't_y',...
    };

actor = actors{actor_id+1};
seq_name = sequences{actor_id + 1}{seq_id + 1};

if file_end_id<file_start_id
    db_n_frames=importdata(sprintf('%s/%s/n_frames.txt',db_root_dir, actor));
    n_frames=db_n_frames.data(strcmp(strtrim(db_n_frames.textdata),seq_name));
    file_end_id=n_frames;
    fprintf('Using file_end_id=%d\n', file_end_id);
end

if file_start_id<=frame_gap
    file_start_id=frame_gap+1;
end
if file_end_id<=file_start_id
    file_end_id=file_start_id;
end

src_img_fname=sprintf('%s/%s.bin', img_root_dir, seq_name);

end_exec = 0;
frame_id_diff = 1;

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
    
    img_pos=zeros(file_end_id, 1);
    for frame_id=1:file_end_id
        img_pos(frame_id)=img_start_pos + img_size*(frame_id-1);
    end
    
    img_fig=figure;
    set(img_fig, 'Position', [1000, 50, 800, 600]);
    set(img_fig, 'Name', seq_name);
end

if plot_only_norm
    if plot_feat_norm
        data_types={
        'FeatNorm'...%0
        };
    elseif plot_likelihood
        data_types={
        'Likelihood'...%0
        };
    else
        data_types={
        'Norm'...%0
        };
    end
    

    if plot_num_jac
        data_types=[
            data_types,...
            {'Jacobian'}
            ];
    end
elseif plot_only_hess
    data_types={
        'Std',...%0
        'Std2'...%1
        };
    if plot_misc_hess
        data_types=[
            data_types,...
            {'InitSelf',...%5
            'CurrSelf',...%6
            'ESM',...%7
            'SumOfStd',...%8
            'SumOfSelf'...%9
            }
            ];
        if plot_sec_ord_hess
            data_types=[
                data_types,...
                {'InitSelf2',...%5
                'CurrSelf2',...%6
                'ESM2',...%7
                'SumOfStd2',...%8
                'SumOfSelf2'...%9
                }
                ];
        end
    end
    
    if plot_num_hess
        data_types=[
            data_types,...
            {'Hessian'}
            ];
    end
elseif plot_only_jac
    data_types={
        'Norm',...%0
        'StdJac'...%1
        };
    if plot_num_jac
        data_types=[
            data_types,...
            {'Jacobian'}
            ];
    end
else
    data_types={
        'Norm',...%1
        'StdJac',...%2
        'Std',...%3
        'Std2',...%4
        'Jacobian',...%5
        'Hessian'%6
        };
    if plot_misc_hess
        data_types=[
            data_types,...
            {'InitSelf',...%7
            'CurrSelf',...%8
            'SumOfStd',...%9
            'SumOfSelf'...%10
            }
            ];
    end
end

% data_ids=1:11;
n_data=length(data_types);
plot_data=cell(n_data, 1);
data_fnames=cell(n_data, 1);
bin_diag_fids=cell(n_data, 1);

for data_id=1:n_data
    curr_data_type = data_types{data_id};
    if use_inv_data
        curr_data_type=sprintf('inv_%s', curr_data_type);
    end
    if isempty(out_prefix)
        out_prefix=sprintf('%s_%s', am_name, ssm_name);
         if ~strcmp(ilm_name,'0')
             out_prefix=sprintf('%s_%s', out_prefix, ilm_name);
         end
    end
    bin_data_fname = sprintf('%s/diagnostics/%s/%s_%d_%s_%d_%d_%d.bin',...
        root_dir, seq_name, out_prefix, update_type, curr_data_type,...
        frame_gap, file_start_id-1, file_end_id-1);
    
    if ~exist(bin_data_fname,'file')
        error('Distance data file is not present:\n%s', bin_data_fname);
    end
    diag_file_id=fopen(bin_data_fname);
    diag_res=fread(diag_file_id, 1 , 'uint32', 'a');
    file_state_size=fread(diag_file_id, 1 , 'uint32', 'a');
    if isempty(diag_res) || isempty(file_state_size)
        error('Diagnostic data file for %s is empty: \n%s', curr_data_type, bin_data_fname);
    else
        fprintf('Reading data for %s from %s\n', curr_data_type, bin_data_fname);
    end
    bin_diag_fids{data_id}=diag_file_id;
end

if length(state_ids)>1 || state_ids>0
    state_size=length(state_ids);
else
    state_ids=1:file_state_size;
    state_size=file_state_size;
end
data_rows=diag_res;
data_cols=2*file_state_size;

n_plots=10*state_size;
n_axes=4*state_size;
diag_figs=cell(state_size, 1);
plot_handles=cell(n_plots, 1);
axes_handles=cell(n_axes, 1);
diag_lkl_figs=cell(state_size, 1);
lkl_plot_handles=cell(n_plots, 1);

dist_fid=bin_diag_fids{1};
grid_start_pos = ftell(dist_fid);
dist_grid_size=data_rows*data_cols*8;
dist_grid_pos=zeros(file_end_id, 1);
for i=1:file_end_id
    dist_grid_pos(i)=grid_start_pos + dist_grid_size*(i-file_start_id);
end

for state_id = state_ids
    diag_figs{state_id}=figure;
    if normalized_fig
        set (diag_figs{state_id}, 'Units', 'normalized', 'Position', [0,0.03,1.00,0.88]);
    end
end

main_fig=figure;
set(main_fig,'name','Control','numbertitle','off')
x_pos=10;
y_pos=200;
pb_height=30;
pb_width=50;
slider_height=80;
slider_width=20;
txt_height=15;
[slider_speed, txt_speed, pb_pause, pb_back, pb_next, pb_rewind, pb_exit] = createWidgets2D(main_fig,...
    x_pos, y_pos, pb_width, pb_height, slider_width, slider_height, txt_height);
set(main_fig,'toolbar','figure')
pmainfig=get(main_fig);
line_width=2;
set (main_fig, 'Position', [50,100,200,300]);
if pause_exec
    set(pb_pause, 'String', 'Resume');
else
    set(pb_pause, 'String', 'Pause');
end
hac_data=zeros(diag_res, 1);
frame_id=start_id;
prev_frame_id = 0;
file_n_frames=file_end_id - file_start_id + 1;
while ~end_exec
    if frame_id<start_id
        pause_exec=1;
        set(pb_pause, 'String', 'Resume');
        frame_id=start_id;
    end
    if frame_id>file_n_frames
        pause_exec=1;
        set(pb_pause, 'String', 'Resume');
        frame_id=file_n_frames;
    end
    if(frame_id ~= prev_frame_id)
        curr_frame_id=frame_id;
        for i=1:n_data
            fseek(bin_diag_fids{i},dist_grid_pos(curr_frame_id), 'bof');
            data=fread(bin_diag_fids{i}, [data_rows data_cols] , 'float64', 'a');
            plot_data{i}=data;
        end
        if plot_only_hess
            plotDiagHess;
        elseif plot_only_jac
            plotDiagJac;
        else
            plotDiagAll;
        end
        if show_img
            fseek(img_fid, img_pos(curr_frame_id + file_start_id), 'bof');
            img_bin=fread(img_fid, [img_width img_height], 'uint8', 'a');
            img_bin=uint8(img_bin');
            set(0,'CurrentFigure',img_fig);
            imshow(img_bin);
        end
        prev_frame_id=curr_frame_id;
    end
    if ~pause_exec
        frame_id=frame_id+speed_factor*frame_id_diff;
    end
    pause(0.1);
end
for i=1:n_data
    fclose(bin_diag_fids{i});
end
