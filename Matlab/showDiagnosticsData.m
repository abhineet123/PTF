% close all;
clear all;
colRGBDefs;
getParamLists;

db_root_dir=sprintf('../../Datasets');
img_root_dir=sprintf('../../Image Data');
root_dir='../C++/MTF_LIB/log';

actor_id = 0;
seq_id = 3;
update_type = 1;
opt_type = 0;
am_name = 'ccre';
ssm_name = '2';
start_id = 1;
file_start_id = 1;
file_end_id = 0;
state_ids = 2;
use_inv_data = 0;
plot_only_hessian = 1;
plot_num_hessian = 1;
plot_misc_hessian = 1;
plot_jacobian_only = 0;
plot_num_jac = 1;
show_img = 0;

actor = actors{actor_id+1};
seq_name = sequences{actor_id + 1}{seq_id + 1};

if file_end_id<file_start_id
    db_n_frames=importdata(sprintf('%s/%s/n_frames.txt',db_root_dir, actor));
    n_frames=db_n_frames.data(strcmp(db_n_frames.textdata,seq_name));
    file_end_id=n_frames;
    fprintf('Using file_end_id=%d\n', file_end_id);
end

src_img_fname=sprintf('%s/%s.bin', img_root_dir, seq_name);

pause_exec=1;
end_exec=0;
speed_factor=1;
frame_id_diff=1;

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

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 20);
set(0,'DefaultAxesFontWeight', 'bold');

if plot_only_hessian
    data_types={
        'hess',...%0
        'hess2'...%1
        };
    if plot_misc_hessian
        data_types=[
            data_types,...
            {'ihess2',...%5
            'chess2',...%6
            'mjhess2',...%7
            'mjhhess2'...%8
            }
            ];
    end
    if plot_num_hessian
        data_types=[
            data_types,...
            {'hess_num'}
            ];
    end
elseif plot_jacobian_only
    data_types={
        'norm',...%0
        'jac'...%1
        };
    if plot_num_jac
        data_types=[
            data_types,...
            {'jac_num'}
            ];
    end
else
    data_types={
        'norm',...%1
        'jac',...%2
        'hess',...%3
        'hess2',...%4
        'jac_num',...%5
        'hess_num'%6
        };
    if plot_misc_hessian
        data_types=[
            data_types,...
            {'ihess2',...%7
            'chess2',...%8
            'mjhess2',...%9
            'mjhhess2'...%10
            }
            ];
    end
end
data_types2={
    'hess2_nn',...
    'hess2_linear',...
    'hess2_bicubic',...
    'hess2_bspl'
    };

% data_ids=1:11;
n_data=length(data_types);
plot_data=cell(n_data, 1);
data_fnames=cell(n_data, 1);

bin_diag_fids=cell(n_data, 1);

for data_id=1:n_data
    curr_data_type= data_types{data_id};
    if use_inv_data
        curr_data_type=sprintf('inv_%s', curr_data_type);
    end
    bin_data_fname = sprintf('%s/diagnostics/%s/%s_%s_%d_%s_%d_%d.bin',...
        root_dir, seq_name, am_name, ssm_name, update_type, curr_data_type,...
        file_start_id-1, file_end_id-1);
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
        %         fprintf('diag_res: %d\n', diag_res);
        %         fprintf('state_size: %d\n', state_size);
    end
    bin_diag_fids{data_id}=diag_file_id;
end

if length(state_ids)>1 || state_ids>0
    state_size=length(state_ids);
else
    state_ids=1:file_state_size;
end
data_rows=diag_res;
data_cols=2*file_state_size;

n_plots=10*state_size;
n_axes=4*state_size;
diag_figs=cell(state_size, 1);
plot_handles=cell(n_plots, 1);
axes_handles=cell(n_axes, 1);

dist_fid=bin_diag_fids{1};
grid_start_pos = ftell(dist_fid);
dist_grid_size=data_rows*data_cols*8;
dist_grid_pos=zeros(file_end_id, 1);
for i=1:file_end_id
    dist_grid_pos(i)=grid_start_pos + dist_grid_size*(i-start_id);
end

for state_id = state_ids
    diag_figs{state_id}=figure;
end

main_fig=figure;
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
last_frame_id=0;
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
    if(frame_id ~= last_frame_id)
        curr_frame_id=frame_id;
        for i=1:n_data
            fseek(bin_diag_fids{i},dist_grid_pos(curr_frame_id), 'bof');
            data=fread(bin_diag_fids{i}, [data_rows data_cols] , 'float64', 'a');
            plot_data{i}=data;
        end
        if plot_only_hessian
            hessian_data=plot_data{strcmp(data_types,'hess')};
            hessian2_data=plot_data{strcmp(data_types,'hess2')};
            if plot_misc_hessian
                ihessian2_data=plot_data{strcmp(data_types,'ihess2')};
                chessian2_data=plot_data{strcmp(data_types,'chess2')};
                mjhessian2_data=plot_data{strcmp(data_types,'mjhess2')};
                mjhhessian2_data=plot_data{strcmp(data_types,'mjhhess2')};
            end
            if plot_num_hessian
                hessian_num_data=plot_data{strcmp(data_types,'hess_num')};
            end
            plot_id=1;
            for state_id = state_ids
                diag_fig=diag_figs{state_id};
                set(0,'CurrentFigure',diag_fig);
                %                 figure(diag_fig);
                x_label = sprintf('param %d frame %d', state_id, frame_id);
                set(diag_fig, 'Name', x_label);
                plot_title=sprintf('Different Hessians for %s (%s frame %d)',...
                    upper(am_name), seq_name, frame_id );
                title(plot_title, 'interpreter','none');
                if frame_id==start_id
                    hold off;
                    plot_handles{plot_id}=plot(hessian2_data(:, 2*state_id-1), hessian2_data(:, 2*state_id),...
                        'Color', col_rgb{strcmp(col_names,'red')}, 'LineWidth', line_width);
                    plot_legend={'FN'};
                else
                    set(plot_handles{plot_id},'ydata',hessian2_data(:, 2*state_id));
                    %             set(axes_handles{axis_id},'title',plot_title);
                end
                plot_id=plot_id+1;
                
                if opt_type==0
                    hac_val=min(hessian2_data(:, 2*state_id));
                else
                    hac_val=max(hessian2_data(:, 2*state_id));
                end
                hac_data(1:diag_res)=hac_val;
                hold on;
                if frame_id==start_id
                    plot_handles{plot_id}=plot(hessian2_data(:, 2*state_id-1), hac_data,...
                        'Color', col_rgb{strcmp(col_names,'red')},...
                        'LineWidth', line_width, 'LineStyle', '--');
                    plot_legend=[plot_legend, {'HAC'}];
                else
                    set(plot_handles{plot_id},'ydata',hac_data);
                end
                plot_id=plot_id+1;
                
                if plot_misc_hessian
                    if frame_id==start_id
                        plot_handles{plot_id}=plot(chessian2_data(:, 2*state_id-1), chessian2_data(:, 2*state_id),...
                            'Color', col_rgb{strcmp(col_names,'blue')}, 'LineWidth', line_width);
                        plot_legend=[plot_legend, {'CN'}];
                    else
                        set(plot_handles{plot_id},'ydata',chessian2_data(:, 2*state_id));
                    end
                    plot_id=plot_id+1;
                    
                    %                 hold on;
                    %                 if frame_id==start_id
                    %                     plot_handles{plot_id}=plot(mjhhessian2_data(:, 2*state_id-1), mjhhessian2_data(:, 2*state_id),...
                    %                         'Color', col_rgb{strcmp(col_names,'black')}, 'LineWidth', line_width);%
                    %                     plot_legend=[plot_legend, {'MN2'}];
                    %                 else
                    %                     set(plot_handles{plot_id},'ydata',mjhhessian2_data(:, 2*state_id));
                    %                 end
                    %                 plot_id=plot_id+1;
                    
                    
                    if frame_id==start_id
                        plot_handles{plot_id}=plot(mjhessian2_data(:, 2*state_id-1), mjhessian2_data(:, 2*state_id),...
                            'Color', col_rgb{strcmp(col_names,'orange')}, 'LineWidth', line_width);
                        plot_legend=[plot_legend, {'MN/MN2'}];
                    else
                        set(plot_handles{plot_id},'ydata',mjhessian2_data(:, 2*state_id));
                    end
                    plot_id=plot_id+1;
                    
                    hold on;
                    if frame_id==start_id
                        plot_handles{plot_id}=plot(ihessian2_data(:, 2*state_id-1), ihessian2_data(:, 2*state_id),...
                            'Color', col_rgb{strcmp(col_names,'cyan')}, 'LineWidth', line_width);
                        plot_legend=[plot_legend, {'IN'}];
                    else
                        set(plot_handles{plot_id},'ydata',ihessian2_data(:, 2*state_id));
                    end
                    plot_id=plot_id+1;
                end
                if plot_num_hessian
                    if frame_id==start_id
                        plot_handles{plot_id}=plot(hessian_num_data(:, 2*state_id-1), hessian_num_data(:, 2*state_id),...
                            'Color', col_rgb{strcmp(col_names,'green')});
                        plot_legend=[plot_legend, {'Numerical'}];
                    else
                        set(plot_handles{plot_id},'ydata',hessian_num_data(:, 2*state_id));
                    end
                    plot_id=plot_id+1;
                end
                
                
                hold on;
                if frame_id==start_id
                    plot_handles{plot_id}=plot(hessian_data(:, 2*state_id-1), hessian_data(:, 2*state_id),...
                        'Color', col_rgb{strcmp(col_names,'magenta')}, 'LineWidth', line_width);
                    plot_legend=[plot_legend, {'GN'}];
                    xlabel(['$T_x$'], 'interpreter','latex');
                    ylabel([sprintf('$\\frac{\\partial^2 F_{%s}}{\\partial T_x^2}$', am_name)], 'interpreter','latex');
                    xlhand = get(gca,'xlabel');
                    ylhand = get(gca,'ylabel');
                    set(xlhand,'fontsize',50);
                    set(ylhand,'fontsize',50);
                    legend(plot_legend);
                    grid on;
                else
                    set(plot_handles{plot_id},'ydata',hessian_data(:, 2*state_id));
                end
                plot_id=plot_id+1;
                
            end
        elseif plot_jacobian_only
            norm_data=plot_data{strcmp(data_types,'norm')};
            jacobian_data=plot_data{strcmp(data_types,'jac')};
            if(plot_num_jac)
                jacobian_num_data=plot_data{strcmp(data_types,'jac_num')};
            end
            plot_id=1;
            for state_id = state_ids
                diag_fig=diag_figs{state_id};
                set(0,'CurrentFigure',diag_fig);                
                plot_title=sprintf('Norm');
                fig_title=sprintf('Norm and Jacobian for %s (%s frame %d)',...
                    upper(am_name), seq_name, frame_id );
                set(diag_fig, 'Name', fig_title);
                if frame_id==start_id
                    subplot(1,2,1);
                    plot_handles{plot_id}=plot(norm_data(:, 2*state_id-1), norm_data(:, 2*state_id),...
                        'LineWidth', line_width);
                    title(plot_title, 'interpreter','none'), grid on;
                    xlabel(['$T_x$'], 'interpreter','latex');
                    ylabel([sprintf('$F_{%s}$', am_name)], 'interpreter','latex');
                    xlhand = get(gca,'xlabel');
                    ylhand = get(gca,'ylabel');
                    set(xlhand,'fontsize',50);
                    set(ylhand,'fontsize',50);
                else
                    set(plot_handles{plot_id},'ydata',norm_data(:, 2*state_id));
                end
                
                hold on;
                plot_id=plot_id+1;
                plot_title=sprintf('Jacobian');
                if frame_id==start_id
                    subplot(1,2,2);
                    plot_handles{plot_id}=plot(jacobian_data(:, 2*state_id-1),...
                        jacobian_data(:, 2*state_id), 'LineWidth', line_width,...
                        'Color', col_rgb{strcmp(col_names,'red')});
                    if(plot_num_jac)
                        hold on;
                        plot_handles{plot_id}=plot(jacobian_num_data(:, 2*state_id-1),...
                            jacobian_num_data(:, 2*state_id), 'LineWidth', line_width,...
                            'Color', col_rgb{strcmp(col_names,'green')});
                        legend('analytical', 'numerical');
                    end
                    title(plot_title, 'interpreter','none'), grid on;
                    xlabel(['$T_x$'], 'interpreter','latex');
                    ylabel([sprintf('$\\frac{\\partial F_{%s}}{\\partial T_x}$', am_name)], 'interpreter','latex');
                    xlhand = get(gca,'xlabel');
                    ylhand = get(gca,'ylabel');
                    set(xlhand,'fontsize',50);
                    set(ylhand,'fontsize',50);
                else
                    set(plot_handles{plot_id},'ydata',jacobian_data(:, 2*state_id));
                end
                plot_id=plot_id+1;
            end
        else
            norm_data=plot_data{strcmp(data_types,'norm')};
            jacobian_data=plot_data{strcmp(data_types,'jac')};
            hessian_data=plot_data{strcmp(data_types,'hess')};
            hessian2_data=plot_data{strcmp(data_types,'hess2')};          
            jacobian_num_data=plot_data{strcmp(data_types,'jac_num')};
            hessian_num_data=plot_data{strcmp(data_types,'hess_num')};
            if plot_misc_hessian
                ihessian2_data=plot_data{strcmp(data_types,'ihess2')};
                chessian2_data=plot_data{strcmp(data_types,'chess2')};
                mjhessian2_data=plot_data{strcmp(data_types,'mjhess2')};
                mjhhessian2_data=plot_data{strcmp(data_types,'mjhhess2')};
            end            
            plot_id=1;
            axis_id=1;
            
            for state_id = state_ids
                diag_fig=diag_figs{state_id};
                %         set (diag_fig, 'Units', 'normalized', 'Position', [0,0,1,1]);
                x_label = sprintf('param %d frame %d', state_id, frame_id);
                set(diag_fig, 'Name', x_label);
                
                % matlab_jacobian=gradient(norm_data(:, 2*state_id));
                % matlab_hessian=gradient(jacobian_data(:, 2*state_id));
                
                
                plot_title=sprintf('Norm');
                if frame_id==start_id
                    figure(diag_fig), hold off;
                    axes_handles{axis_id}=subplot(2,2,1);
                    plot_handles{plot_id}=plot(norm_data(:, 2*state_id-1), norm_data(:, 2*state_id));
                    title(plot_title), grid on, xlabel(x_label), ylabel('Norm');
                else
                    set(plot_handles{plot_id},'ydata',norm_data(:, 2*state_id));
                    %             set(axes_handles{axis_id},'title',plot_title);
                end
                axis_id=axis_id+1;
                plot_id=plot_id+1;
                
                plot_title=sprintf('Jacobian');
                
                if frame_id==start_id
                    axes_handles{axis_id}=subplot(2,2,2);
                    hold off;
                    plot_handles{plot_id}=plot(jacobian_data(:, 2*state_id-1), jacobian_data(:, 2*state_id),...
                        'Color', col_rgb{strcmp(col_names,'red')});
                else
                    set(plot_handles{plot_id},'ydata',jacobian_data(:, 2*state_id));
                    %             set(axes_handles{axis_id},'title',plot_title);
                end
                axis_id=axis_id+1;
                plot_id=plot_id+1;
                
                
                if frame_id==start_id
                    hold on;
                    plot_handles{plot_id}=plot(jacobian_num_data(:, 2*state_id-1), jacobian_num_data(:, 2*state_id),...
                        'Color', col_rgb{strcmp(col_names,'green')});
                    title(plot_title), grid on, xlabel(x_label), ylabel('Jacobian'), legend('analytical', 'numerical');
                else
                    set(plot_handles{plot_id},'ydata',jacobian_num_data(:, 2*state_id));
                end
                plot_id=plot_id+1;
                %  plot(jacobian_num_data(:, 2*state_id-1), matlab_jacobian,...
                %'Color', col_rgb{strcmp(col_names,'blue')});
                
                
                plot_title=sprintf('Hessian');
                if frame_id==start_id
                    axes_handles{axis_id}=subplot(2,2,3);
                    hold off;
                    plot_handles{plot_id}=plot(hessian_data(:, 2*state_id-1), hessian_data(:, 2*state_id),...
                        'Color', col_rgb{strcmp(col_names,'red')});
                else
                    set(plot_handles{plot_id},'ydata',hessian_data(:, 2*state_id));
                    %             set(axes_handles{axis_id},'title',plot_title);
                end
                axis_id=axis_id+1;
                plot_id=plot_id+1;
                
                hold on;
                if frame_id==start_id
                    plot_handles{plot_id}=plot(hessian_num_data(:, 2*state_id-1), hessian_num_data(:, 2*state_id),...
                        'Color', col_rgb{strcmp(col_names,'green')});
                    title(plot_title), grid on, xlabel(x_label), ylabel('Hessian'), legend('analytical', 'numerical');
                else
                    set(plot_handles{plot_id},'ydata',hessian_num_data(:, 2*state_id));
                end
                plot_id=plot_id+1;
                %             plot(hessian_num_data(:, 2*state_id-1), matlab_hessian,...
                %                 'Color', col_rgb{strcmp(col_names,'blue')});
                
                plot_title=sprintf('Hessian2');
                if frame_id==start_id
                    axes_handles{axis_id}=subplot(2,2,4);
                    hold off;
                    plot_handles{plot_id}=plot(hessian2_data(:, 2*state_id-1), hessian2_data(:, 2*state_id),...
                        'Color', col_rgb{strcmp(col_names,'red')});
                    plot_legend={'analytical'};
                else
                    set(plot_handles{plot_id},'ydata',hessian2_data(:, 2*state_id));
                    %             set(axes_handles{axis_id},'title',plot_title);
                end
                axis_id=axis_id+1;
                plot_id=plot_id+1;
                
                hold on;
                if plot_misc_hessian             
                    if frame_id==start_id
                        plot_handles{plot_id}=plot(ihessian2_data(:, 2*state_id-1), ihessian2_data(:, 2*state_id),...
                            'Color', col_rgb{strcmp(col_names,'cyan')});
                        plot_legend=[plot_legend, {'initial'}];
                    else
                        set(plot_handles{plot_id},'ydata',ihessian2_data(:, 2*state_id));
                    end
                    plot_id=plot_id+1;

                    if frame_id==start_id
                        plot_handles{plot_id}=plot(chessian2_data(:, 2*state_id-1), chessian2_data(:, 2*state_id),...
                            'Color', col_rgb{strcmp(col_names,'blue')});
                        plot_legend=[plot_legend, {'current'}];
                    else
                        set(plot_handles{plot_id},'ydata',chessian2_data(:, 2*state_id));
                    end
                    plot_id=plot_id+1;

                    if frame_id==start_id
                        plot_handles{plot_id}=plot(mjhessian2_data(:, 2*state_id-1), mjhessian2_data(:, 2*state_id),...
                            'Color', col_rgb{strcmp(col_names,'orange')});
                        plot_legend=[plot_legend, {'mj'}];
                    else
                        set(plot_handles{plot_id},'ydata',mjhessian2_data(:, 2*state_id));
                    end
                    plot_id=plot_id+1;
                    
                    if frame_id==start_id
                        plot_handles{plot_id}=plot(mjhhessian2_data(:, 2*state_id-1), mjhhessian2_data(:, 2*state_id),...
                            'Color', col_rgb{strcmp(col_names,'black')});
                        plot_legend=[plot_legend, {'mjh'}];
                    else
                        set(plot_handles{plot_id},'ydata',mjhhessian2_data(:, 2*state_id));
                    end
                    plot_id=plot_id+1;
                end
                
                if frame_id==start_id
                    plot_handles{plot_id}=plot(hessian_num_data(:, 2*state_id-1), hessian_num_data(:, 2*state_id),...
                        'Color', col_rgb{strcmp(col_names,'green')});
                    plot_legend=[plot_legend, {'numerical'}];
                    title(plot_title), grid on, xlabel(x_label), ylabel('Hessian2'), legend(plot_legend);
                else
                    set(plot_handles{plot_id},'ydata',hessian_num_data(:, 2*state_id));
                end
                plot_id=plot_id+1;              
            end
        end
        if show_img
            fseek(img_fid, img_pos(curr_frame_id), 'bof');
            img_bin=fread(img_fid, [img_width img_height], 'uint8', 'a');
            img_bin=uint8(img_bin');
            set(0,'CurrentFigure',img_fig);
            imshow(img_bin);
        end
        last_frame_id=curr_frame_id;
    end
    if ~pause_exec
        frame_id=frame_id+speed_factor*frame_id_diff;
    end
    pause(0.1);
end
for i=1:n_data
    fclose(bin_diag_fids{i});
end
