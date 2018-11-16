initVarsMulti;

dist_fids=cell(surf_count, 1);
surf_x_mat=cell(surf_count, 1);
surf_y_mat=cell(surf_count, 1);
surf_z_mat=cell(surf_count, end_id);

x_labels=cell(surf_count, 1);
y_labels=cell(surf_count, 1);
surf_figs=cell(surf_count, 1);
x_vals_mat=cell(surf_count, 1);
y_vals_mat=cell(surf_count, 1);
z_vals_mat=cell(surf_count, 1);
pt_cols_mat=cell(surf_count, 1);
n_iters_mat=cell(surf_count, 1);
plot_title_templ=cell(surf_count, 1);


grid_start_pos=zeros(surf_count, 1);
dist_grid_sizes=zeros(surf_count, 1);
min_x=zeros(surf_count, 1);
min_y=zeros(surf_count, 1);
max_x=zeros(surf_count, 1);
max_y=zeros(surf_count, 1);


for i=1:surf_count
    grid_type=grid_types{i};
    [x_label, y_label, x_id, y_id] = getAxisLabelsAndIDs(grid_type);
    dist_fname_bin=sprintf('%s/%s_%s_%s.bin', dist_dir, tracker_type, inc_type, grid_type);
    fprintf('grid_type: %s dist_fname_bin: %s\n', grid_type, dist_fname_bin);
    if ~exist(dist_fname_bin,'file')
        error('Distance data file not found:\n%s', dist_fname_bin);
    end
    dist_fid=fopen(dist_fname_bin);
    dist_fids{i}=dist_fid;
    
    file_end_id=fread(dist_fid, 1 , 'uint32', 'a');
    file_start_id=fread(dist_fid, 1 , 'uint32', 'a');
    surf_x_len=fread(dist_fid, 1 , 'uint32', 'a');
    surf_y_len=fread(dist_fid, 1 , 'uint32', 'a');
    
    fprintf('file_end_id: %d\n', file_end_id);
    fprintf('file_start_id: %d\n', file_start_id);
    fprintf('surf_x_len: %d\n', surf_x_len);
    fprintf('surf_y_len: %d\n', surf_y_len);   
    
    surf_x=fread(dist_fid, surf_x_len , 'float64', 'a');
    surf_y=fread(dist_fid, surf_y_len , 'float64', 'a');
    surf_x_mat{i}=surf_x;
    surf_y_mat{i}=surf_y;
    min_x(i)=min(surf_x);
    min_y(i)=min(surf_y);
    max_x(i)=max(surf_x);
    max_y(i)=max(surf_y);
    
    if start_id<file_start_id
        fprintf('start_id is smaller than file_start_id: %d\n', file_start_id);
        start_id=file_start_id;
    end
    if end_id>file_end_id
        fprintf('end_id is larger than the number of frames in the distance data file: %d\n', file_end_id);
        end_id=file_end_id;
    end
    
    for frame_id=start_id:end_id
        surf_z=fread(dist_fid, [length(surf_x) length(surf_y)], 'float64', 'a');
        surf_z=surf_z';
        surf_z_mat{i, frame_id}=surf_z;
    end
        
    
    grid_start_pos(i) = ftell(dist_fid);
    dist_grid_size=surf_x_len*surf_y_len*8;
    fseek(dist_fid, dist_grid_size*(start_id-file_start_id), 'cof');
    dist_grid_sizes(i)=dist_grid_size;
    x_labels{i}=x_label;
    y_labels{i}=y_label;
    
    surf_fig=figure;
    surf_figs{i}=surf_fig;
    
    grid_row_id=mod(i-1, surf_count/2);
    grid_col_id=floor((i-1)/(surf_count/2));
    fig_title=sprintf('%s_%s_%s_%s', inc_type, grid_type, seq_name, appearance_model);
    set(surf_fig, 'Position', [surf_width*grid_row_id, (surf_height+100)*grid_col_id+50,...
        surf_width, surf_height]);
    set(surf_fig, 'Name', fig_title);
    if i==1
        x_pos=0;
        y_pos=300;
        pb_height=30;
        pb_width=50;
        slider_height=80;
        slider_width=20;
        txt_height=15;
        [slider_speed, txt_speed, pb_pause, pb_back, pb_next, pb_rewind, tb_rot, pb_exit] = createWidgets(surf_fig,...
            x_pos, y_pos, pb_width, pb_height, slider_width, slider_height, txt_height);
    end
    set(surf_fig,'toolbar','figure');
    psurf_fig=get(surf_fig);
    rotate3d(surf_fig);
    
    if tracking_mode        
        [x_vals, y_vals, n_iters, pt_cols, max_frame_id] = getPointsToPlot(tracker_hom_params,...
            max_iters, x_id, y_id);
        z_vals=zeros(end_id, max_iters);
        for frame_id=start_id:end_id  
            surf_z=surf_z_mat{i, frame_id};
            z_vals_frame=interp2(surf_x, surf_y, surf_z, x_vals(frame_id, 1:n_iters(frame_id)), y_vals(frame_id, 1:n_iters(frame_id)));
            z_vals_frame(isnan(z_vals_frame))=0;
            %z_vals_frame_size=size(z_vals_frame)
            %n_iter=n_iters(frame_id)
            z_vals(frame_id, 1:n_iters(frame_id))=z_vals_frame;
        end        
        x_vals_mat{i}=x_vals;
        y_vals_mat{i}=y_vals;
        z_vals_mat{i}=z_vals;
        n_iters_mat{i}=n_iters;
        pt_cols_mat{i}=pt_cols;
    end
    plot_title_templ{i}=sprintf('%s_%s_%s_%s_%s%d_%s', inc_type, grid_type, seq_name,...
        appearance_model, filter_type, kernel_size, tracker_type);
end

if show_img   
    if ~exist('max_frame_id', 'var')
        max_frame_id=1500;
    end
    img_pos=zeros(max_frame_id, 1);
    for frame_id=1:max_frame_id
        img_pos(frame_id)=img_start_pos + img_size*(frame_id-1);
    end
end

pause_exec=0;
end_exec=0;
frame_id=start_id;
last_frame_id=0;
frame_id_diff=1;

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
    
    if frame_id~=last_frame_id
        tic;
        curr_frame_id=frame_id;
        for i=1:surf_count
            dist_fid=dist_fids{i};
            surf_x=surf_x_mat{i};
            surf_y=surf_y_mat{i};
            surf_z=surf_z_mat{i, curr_frame_id};
            surf_fig=surf_figs{i};
            x_label=x_labels{i};
            y_label=y_labels{i};          
            if i==1
                figure(surf_fig);
            else
                set(0,'CurrentFigure',surf_fig);
            end
            surf(surf_x, surf_y, surf_z,...
                'FaceColor', 'interp', 'FaceAlpha', 0.5);          
            
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
            plot_title=sprintf('%s_%d_%d', plot_title_templ{i}, curr_frame_id, speed_factor);
            title(plot_title, 'interpreter','none');
            
            fprintf('Frame:\t%4d ::\t', curr_frame_id);
            
            if tracking_mode
                x_vals=x_vals_mat{i};
                y_vals=y_vals_mat{i};
                z_vals=z_vals_mat{i};
                pt_cols=pt_cols_mat{i};
                n_iters=n_iters_mat{i};
                hold on;
                for j=1:n_iters(curr_frame_id)
                    z_val=z_vals(curr_frame_id, j);
                    plot3(x_vals(curr_frame_id, j),y_vals(curr_frame_id, j),z_val,...
                        'o','markerfacecolor',pt_cols(curr_frame_id, j, :),'markersize',5);
                end
                fprintf('n_iter:\t%d\t', n_iters(curr_frame_id));
                hold off;
            end          
            
            if opt_min
                surf_z(surf_z<0)=inf;
                [opt_row_val, opt_row_ids]=min(surf_z);
                [opt_col_val, opt_col_id]=min(opt_row_val);
            else
                [opt_row_val, opt_row_ids]=max(surf_z);
                [opt_col_val, opt_col_id]=max(opt_row_val);
            end
            opt_row_id=opt_row_ids(opt_col_id);            
            fprintf('OptDist:\t%12.6f\t at:\t (%3d, %3d)\t %5s:\t %8.5f\t %5s:\t %8.5f\n',...
                opt_col_val, opt_row_id, opt_col_id, x_label, surf_x(opt_col_id),...
                y_label, surf_y(opt_row_id));
        end
        time_intvl=toc;
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
        time_intvl2=toc;
        fprintf('FPS1:\t%10.8f\tFPS2:\t%10.8f\n', 1.0/time_intvl, 1.0/time_intvl2);
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
