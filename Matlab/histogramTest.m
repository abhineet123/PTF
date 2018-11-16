clear all;
% fhist=importdata('fhist.txt');
% hist=importdata('hist.txt');
% fhist_fig=figure;
% surf(fhist, 'FaceColor', 'interp', 'FaceAlpha', 0.5);
% rotate3d(fhist_fig);
% 
% hist_fig=figure;
% surf(hist, 'FaceColor', 'interp', 'FaceAlpha', 0.5)
% rotate3d(hist_fig);

hist_types = {'floor',...
              'round',...
              'frac',...
              'mi',...
              'bspline'
};
hist_id = 4;
hist_type = hist_types{hist_id+1};

fprintf('hist_type: %s\n', hist_type);

end_exec=0;
pause_exec=0;
rotate_surf=0;
view_az = 0;
view_al = 0;
d_az = 5;
d_al = 5;
max_al=45;
min_al=-45;
hist_fid=fopen(sprintf('hist_%s.bin', hist_type));

n_bins_size=fread(hist_fid, 1 , 'uint32', 'a');
n_bins_vec=fread(hist_fid, n_bins_size , 'uint32', 'a');
hist_fig=figure;
rotate3d(hist_fig);
pb_pause = uicontrol(hist_fig,'Style','pushbutton','String','Pause','Value',0,...
    'Position',[0 500 50 30], 'Callback', @(h, e) evalin('base', 'pause_exec=1-pause_exec'));
pb_back = uicontrol(hist_fig,'Style','pushbutton','String','Back','Value',0,...
    'Position',[0 470 50 30], 'Callback', @(h, e) evalin('base', 'i=i-1'));
pb_next = uicontrol(hist_fig,'Style','pushbutton','String','Next','Value',0,...
    'Position',[0 440 50 30], 'Callback', @(h, e) evalin('base', 'i=i+1'));
pb_rewind = uicontrol(hist_fig,'Style','pushbutton','String','Rewind','Value',0,...
    'Position',[0 410 50 30], 'Callback', @(h, e) evalin('base', 'pause_exec=1; i_diff=-i_diff;'));
pb_exit = uicontrol(hist_fig,'Style','pushbutton','String','Stop','Value',0,...
    'Position',[0 380 50 30], 'Callback', @(h, e) figCallbackExit( h, e ));

set(hist_fig, 'Position', [0, 50, 850, 650]);

hist_cell=cell(n_bins_size, 1);
grid_vec_cell=cell(n_bins_size, 1);
for i=1:n_bins_size
    n_bins=n_bins_vec(i);
    grid_vec=fread(hist_fid, n_bins, 'float64', 'a');
    hist=fread(hist_fid, [n_bins n_bins], 'float64', 'a');
    hist_cell{i}=hist;
    grid_vec_cell{i}=grid_vec;
end
fclose(hist_fid);

i=1;
last_i=0;
i_diff=1;

while ~end_exec
    
    if i~=last_i
        n_bins=n_bins_vec(i);
        last_i=i;
        grid_vec=grid_vec_cell{i};
        hist=hist_cell{i};
        figure(hist_fig), surf(grid_vec, grid_vec, hist, 'FaceColor', 'interp', 'FaceAlpha', 0.5);
        plot_title=sprintf('%d', n_bins);
        title(plot_title, 'interpreter','none');
    end
    if ~pause_exec
        if rotate_surf
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
        i=i+i_diff;        
    end
    pause(0.1);
    if i>n_bins_size
        i=n_bins_size;
    end
    if i<1
        i=1;
    end
end
    
    