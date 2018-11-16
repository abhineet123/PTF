clear all;
[tracker_types, filter_types,  inc_types, pw_opt_types, grid_types, appearance_models, sequences] = getParamLists();
dist_params=importdata('distanceGridParams.txt');
param_names=dist_params.textdata;
param_vals=dist_params.data;
seq_id = param_vals(strcmp(param_names,'seq_id'));
last_seq_id=seq_id-1;
end_exec=0;

no_of_seqs=length(sequences);

fig_width=400;
fig_height=400;

vert_alpha_fig=figure;
set(vert_alpha_fig, 'Position', [0, 50, fig_width, fig_height]);
set(vert_alpha_fig, 'Name', 'vert_alpha');

pb_next = uicontrol(vert_alpha_fig,'Style','pushbutton','String','Next','Value',0,...
    'Position',[5 300 50 30], 'Callback', @(h, e)  evalin('base', 'seq_id=seq_id+1;'));
pb_back = uicontrol(vert_alpha_fig,'Style','pushbutton','String','Back','Value',0,...
    'Position',[5 270 50 30], 'Callback', @(h, e) evalin('base', 'seq_id=seq_id-1;'));
pb_exit = uicontrol(vert_alpha_fig,'Style','pushbutton','String','Exit','Value',0,...
    'Position',[5 240 50 30], 'Callback', @(h, e) evalin('base', 'end_exec=1; close all;'));

horz_alpha_fig=figure;
set(horz_alpha_fig, 'Position', [fig_width, 50, fig_width, fig_height]);
set(horz_alpha_fig, 'Name', 'horz_alpha');

prev_vert_alpha_fig=figure;
set(prev_vert_alpha_fig, 'Position', [2*fig_width, 50, fig_width, fig_height]);
set(prev_vert_alpha_fig, 'Name', 'prev_vert_alpha');

prev_horz_alpha_fig=figure;
set(prev_horz_alpha_fig, 'Position', [3*fig_width, 50, fig_width, fig_height]);
set(prev_horz_alpha_fig, 'Name', 'prev_horz_alpha');

vert_alpha_diff_fig=figure;
set(vert_alpha_diff_fig, 'Position', [0, fig_height+80, fig_width, fig_height]);
set(vert_alpha_diff_fig, 'Name', 'vert_alpha_diff');

horz_alpha_diff_fig=figure;
set(horz_alpha_diff_fig, 'Position', [fig_width, fig_height+80, fig_width, fig_height]);
set(horz_alpha_diff_fig, 'Name', 'horz_alpha_diff');

vert_slope_fig=figure;
set(vert_slope_fig, 'Position', [2*fig_width, fig_height+80, fig_width, fig_height]);

horz_slope_fig=figure;
set(horz_slope_fig, 'Position', [3*fig_width, fig_height+80, fig_width, fig_height]);

tracking_data_root_dir = 'Tracking Data/0bject Grid';

while ~end_exec
    if seq_id~=last_seq_id
        if seq_id<0
            seq_id=0;
            continue;
        elseif seq_id>=no_of_seqs
            seq_id=no_of_seqs-1;
            continue;
        end
        curr_seq_id=seq_id;
        last_seq_id=curr_seq_id;
        
        seq_name = sequences{curr_seq_id+1};
        fprintf('seq_name: %s\n', seq_name);
        
        tracking_data_dir = sprintf('%s/%s', tracking_data_root_dir, seq_name);
        vert_params_file = sprintf('%s/vert_params.txt',tracking_data_dir);
        horz_params_file = sprintf('%s/horz_params.txt',tracking_data_dir);
        vert_alpha_file = sprintf('%s/vert_alpha.txt',tracking_data_dir);
        horz_alpha_file = sprintf('%s/horz_alpha.txt',tracking_data_dir);
        prev_vert_alpha_file = sprintf('%s/prev_vert_alpha.txt',tracking_data_dir);
        prev_horz_alpha_file = sprintf('%s/prev_horz_alpha.txt',tracking_data_dir);
        vert_alpha_diff_file = sprintf('%s/vert_alpha_diff.txt',tracking_data_dir);
        horz_alpha_diff_file = sprintf('%s/horz_alpha_diff.txt',tracking_data_dir);
        
        vert_params=importdata(vert_params_file);
        horz_params=importdata(horz_params_file);
        
        vert_size=size(vert_params, 2);
        horz_size=size(horz_params, 2);
        
        vert_slope=vert_params(:, 1:vert_size/2);
        horz_slope=horz_params(:, 1:horz_size/2);
        
        vert_alpha=importdata(vert_alpha_file);
        horz_alpha=importdata(horz_alpha_file);
        prev_vert_alpha=importdata(prev_vert_alpha_file);
        prev_horz_alpha=importdata(prev_horz_alpha_file);
        vert_alpha_diff=importdata(vert_alpha_diff_file);
        horz_alpha_diff=importdata(horz_alpha_diff_file);
        
        figure(vert_alpha_fig);
        plot(vert_alpha), title(sprintf('vert_alpha_%s',seq_name), 'interpreter', 'none');
        
        set(0,'CurrentFigure',horz_alpha_fig);
        plot(horz_alpha), title(sprintf('horz_alpha_%s',seq_name), 'interpreter', 'none');
        
        set(0,'CurrentFigure',prev_vert_alpha_fig);
        plot(prev_vert_alpha), title(sprintf('prev_vert_alpha_%s',seq_name) , 'interpreter', 'none');
        
        set(0,'CurrentFigure',prev_horz_alpha_fig);
        plot(prev_horz_alpha), title(sprintf('prev_horz_alpha_%s',seq_name), 'interpreter', 'none');
        
        set(0,'CurrentFigure',vert_alpha_diff_fig);
        plot(vert_alpha_diff), title(sprintf('vert_alpha_diff_%s',seq_name), 'interpreter', 'none');
        
        set(0,'CurrentFigure',horz_alpha_diff_fig);
        plot(horz_alpha_diff), title(sprintf('horz_alpha_diff_%s',seq_name), 'interpreter', 'none');
        
        set(0,'CurrentFigure',vert_slope_fig);
        plot(vert_slope), title(sprintf('vert_slope_diff_%s',seq_name), 'interpreter', 'none');
        
        set(0,'CurrentFigure',horz_slope_fig);
        plot(horz_slope), title(sprintf('horz_slope_diff_%s',seq_name), 'interpreter', 'none');      
    end
    pause(0.1);
end




