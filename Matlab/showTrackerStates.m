clear all;
getParamLists;
seq_id = 23;
tracker_id = 0;
steps_per_frame = 25;

seq_name=sequences{seq_id+1};
tracker_type = tracker_types{tracker_id+1};

fname=sprintf('%s/tracker_states_%s_%d.txt', seq_name, tracker_type, steps_per_frame);
diff_fname=sprintf('%s/diff_states_%s_%d.txt', seq_name, tracker_type, steps_per_frame);

fprintf('Reading data from : %s\n', fname);

data=importdata(fname);
data=data(2:end, :);
data_count=size(data, 1)
no_of_frames=uint32(data_count/steps_per_frame)

diff_data=importdata(diff_fname);
diff_data=diff_data(2:end, :);

diff_tx=reshape(diff_data(:, 1), [steps_per_frame, no_of_frames])';
diff_ty=reshape(diff_data(:, 2), [steps_per_frame, no_of_frames])';
diff_scale=reshape(diff_data(:, 3), [steps_per_frame, no_of_frames])';
diff_theta=reshape(diff_data(:, 4), [steps_per_frame, no_of_frames])';


tx=data(:, 1);
tx_mat=reshape(tx, [steps_per_frame, no_of_frames])';
tx_frame=tx_mat(:, steps_per_frame);
tx_diff=tx_mat(:, 2:end)-tx_mat(:, 1:end-1);

ty=data(:, 2);
ty_mat=reshape(ty, [steps_per_frame, no_of_frames])';
ty_frame=ty_mat(:, steps_per_frame);
ty_diff=ty_mat(:, 2:end)-ty_mat(:, 1:end-1);

scale=data(:, 4);
scale_mat=reshape(scale, [steps_per_frame, no_of_frames])';
scale_frame=scale_mat(:, steps_per_frame);
scale_diff=scale_mat(:, 2:end)-scale_mat(:, 1:end-1);

theta=data(:, 3);
theta_mat=reshape(theta, [steps_per_frame, no_of_frames])';
theta_frame=theta_mat(:, steps_per_frame);
theta_diff=abs(theta_mat(:, 2:end)-theta_mat(:, 1:end-1));


plot_x=1:no_of_frames;

figure, plot(plot_x, tx_frame), title('tx');
figure, plot(plot_x, ty_frame), title('ty');
figure, plot(plot_x, theta_frame), title('theta');
figure, plot(plot_x, scale_frame), title('scale');


tx_fig = figure;
ty_fig = figure;
diff_tx_fig = figure;
diff_ty_fig = figure;

set(tx_fig, 'Position', [0, 50, 500, 400]);
set(tx_fig, 'Name', 'tx_fig');
pb_exit = uicontrol(tx_fig,'Style','pushbutton','String','Exit','Value',0,...
    'Position',[5 240 50 30],...
    'Callback', @(h, e) evalin('base', 'end_exec=1; close all;'));
pb_pause = uicontrol(tx_fig,'Style','pushbutton','String','Pause','Value',0,...
    'Position',[5 270 50 30],...
    'Callback', @(h, e) evalin('base', 'pause_exec=1-pause_exec;'));

set(ty_fig, 'Position', [500, 50, 500, 400]);
set(ty_fig, 'Name', 'ty_fig');

set(diff_tx_fig, 'Position', [0, 500, 500, 400]);
set(diff_tx_fig, 'Name', 'diff_tx_fig');

set(diff_ty_fig, 'Position', [500, 500, 500, 400]);
set(diff_ty_fig, 'Name', 'diff_ty_fig');

% theta_fig = figure;
i=0;
end_exec=0;
pause_exec=0;
while ~end_exec
    if ~pause_exec
        i=i+1;
    end
    figure(tx_fig), plot(tx_diff(i, :)), title(sprintf('tx %d', i));
    set(0,'CurrentFigure',ty_fig);
    plot(ty_diff(i, :)), title(sprintf('ty %d', i));
    
    set(0,'CurrentFigure',diff_tx_fig);
    plot(diff_tx(i, :)), title(sprintf('diff_tx %d', i));
    
    set(0,'CurrentFigure',diff_ty_fig);
    plot(diff_ty(i, :)), title(sprintf('diff_ty %d', i)); 
%     figure(theta_fig), plot(theta_diff(i, :)), title(sprintf('theta %d', i));
     pause(0.01);
end


