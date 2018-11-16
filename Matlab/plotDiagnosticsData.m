close all;
clear all;
colRGBDefs;
getParamLists;

root_dir='../C++/MTF_LIB/log';

actor_id = 0;
seq_id = 38;
plot_id = 0;
update_type = 1;
am_name = 'ssd';
ssm_name = '2';
start_id=150;
end_id=200;
binary_input=1;

actor = actors{actor_id+1};
seq_name = sequences{actor_id + 1}{seq_id + 1};

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 14);
set(0,'DefaultAxesFontWeight', 'bold');

data_types={
    'norm',...
    'jac',...
    'hess',...
    'hess2',...
    'ihess2',...
    'chess2',...
    'mjhess2',...
    'mjhhess2',...
    'jac_num',...
    'hess_num',...
    'nhess_num'
    };
data_types2={
    'hess2_nn',...
    'hess2_linear',...
    'hess2_bicubic',...
    'hess2_bspl'
    };

n_data=length(data_types);
plot_data=cell(n_data, 1);
data_fnames=cell(n_data, 1);

bin_dist_fids=cell(n_data, 1);

if binary_input
    for i=1:n_data
        bin_data_fname = sprintf('%s/diagnostics/%s/%s_%s_%d_%s_%d_%d.txt',...
            root_dir, seq_name, am_name, ssm_name, update_type, data_types{i}, start_id, end_id);
        bin_dist_fids(i)=fopen(bin_data_fname);
        diag_res=fread(dist_fid, 1 , 'uint32', 'a');
        state_size=fread(dist_fid, 1 , 'uint32', 'a');
        if isempty(file_end_id) || isempty(file_start_id)
            error('Diagnostic data file for %s is empty: \n%s', data_types{i}, bin_data_fname);
        else
            fprintf('diag_res: %d\n', diag_res);
            fprintf('state_size: %d\n', state_size);
        end
        surf_x_len=fread(dist_fid, 1 , 'uint32', 'a');
        surf_y_len=fread(dist_fid, 1 , 'uint32', 'a');
        surf_x=fread(dist_fid, surf_x_len , 'float64', 'a');
        surf_y=fread(dist_fid, surf_y_len , 'float64', 'a');
    end
end
                
frame_id=start_id;
while frame_id<=end_id
    if plot_id==0
        for i=1:n_data
            if binary_input
                data_fnames{i} = sprintf('%s/diagnostics/%s/%s_%s_%d_%s_%d_%d.txt',...
                    root_dir, seq_name, am_name, ssm_name, update_type, data_types{i}, start_id, end_id);
                dist_fid=fopen(dist_fname);
                diag_res=fread(dist_fid, 1 , 'uint32', 'a');
                ssm_state_size=fread(dist_fid, 1 , 'uint32', 'a');
                if isempty(file_end_id) || isempty(file_start_id)
                    error('Distance data file is empty: \n%s', dist_fname);
                end
                surf_x_len=fread(dist_fid, 1 , 'uint32', 'a');
                surf_y_len=fread(dist_fid, 1 , 'uint32', 'a');
                surf_x=fread(dist_fid, surf_x_len , 'float64', 'a');
                surf_y=fread(dist_fid, surf_y_len , 'float64', 'a');
            else
                data_fnames{i} = sprintf('%s/diagnostics/%s/%s_%s_%d_%s_%d.txt',...
                    root_dir, seq_name, am_name, ssm_name, update_type, data_types{i}, frame_id);
                fprintf('Reading %s data from: %s\n', data_types{i}, data_fnames{i});
                data_struct=importdata(data_fnames{i});
                plot_data{i}=data_struct.data;
            end
        end
        state_size = size(plot_data{1}, 2) / 2
    elseif plot_id==1
        ssm_fname = sprintf('%s/diagnostics/%s/%s_%s_%d_ssm_%d.txt',...
            root_dir, seq_name, am_name, ssm_name, update_type, frame_id);
        fprintf('Reading ssm data from: %s\n', ssm_fname);
        ssm_data=importdata(ssm_fname);
        ssm_data = ssm_data.data;
        
        state_size = size(ssm_data, 2) / 2
    elseif plot_id==2
        fprintf('Reading NN hessian2 data from: %s\n', hessian2_nn_fname);
        fprintf('Reading Linear hessian2 data from: %s\n', hessian2_linear_fname);
        fprintf('Reading Bicubic hessian2 data from: %s\n', hessian2_bicubic_fname);
        fprintf('Reading BSpline hessian2 data from: %s\n', hessian2_bspl_fname);
        fprintf('Reading numerical hessian data from: %s\n', hessian_num_fname);
        
        hessian2_nn_data=importdata(hessian2_nn_fname);
        hessian2_nn_data = hessian2_nn_data.data;
        hessian2_linear_data=importdata(hessian2_linear_fname);
        hessian2_linear_data = hessian2_linear_data.data;
        hessian2_bicubic_data=importdata(hessian2_bicubic_fname);
        hessian2_bicubic_data = hessian2_bicubic_data.data;
        hessian2_bspl_data=importdata(hessian2_bspl_fname);
        hessian2_bspl_data = hessian2_bspl_data.data;
        hessian_num_data=importdata(hessian_num_fname);
        hessian_num_data = hessian_num_data.data;
        
        state_size = size(hessian2_nn_data, 2) / 2
    elseif plot_id==3
        feat_norm_fname = sprintf('%s/diagnostics/%s/%s_%s_%d_feat_norm_%d.txt',...
            root_dir, seq_name, am_name, ssm_name, update_type, frame_id);
        fprintf('Reading feature norm data from: %s\n', feat_norm_fname);
        feat_norm_data=importdata(feat_norm_fname);
        feat_norm_data = feat_norm_data.data;
        state_size = size(feat_norm_data, 2) / 2
    end
    close all;
    for state_id = 1:state_size
        curr_fig=figure;
        set (curr_fig, 'Units', 'normalized', 'Position', [0,0,1,1]);
        x_label = sprintf('param %d', state_id);
        set(curr_fig, 'Name', x_label);
        if plot_id==0
            norm_data=plot_data{strcmp(data_types,'norm')};
            jacobian_data=plot_data{strcmp(data_types,'jac')};
            hessian_data=plot_data{strcmp(data_types,'hess')};
            hessian2_data=plot_data{strcmp(data_types,'hess2')};
            ihessian2_data=plot_data{strcmp(data_types,'ihess2')};
            chessian2_data=plot_data{strcmp(data_types,'chess2')};
            mjhessian2_data=plot_data{strcmp(data_types,'mjhess2')};
            mjhhessian2_data=plot_data{strcmp(data_types,'mjhhess2')};
            
            jacobian_num_data=plot_data{strcmp(data_types,'jac_num')};
            hessian_num_data=plot_data{strcmp(data_types,'hess_num')};
            
            % matlab_jacobian=gradient(norm_data(:, 2*state_id));
            % matlab_hessian=gradient(jacobian_data(:, 2*state_id));
            
            subplot(2,2,1);
            plot(norm_data(:, 2*state_id-1), norm_data(:, 2*state_id));
            title('Norm'), grid on, xlabel(x_label), ylabel('Norm');
            
            subplot(2,2,2), hold on;
            plot(jacobian_data(:, 2*state_id-1), jacobian_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'red')})
            plot(jacobian_num_data(:, 2*state_id-1), jacobian_num_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'green')});
            %  plot(jacobian_num_data(:, 2*state_id-1), matlab_jacobian,...
            %'Color', col_rgb{strcmp(col_names,'blue')});
            plot_title=sprintf('Jacobian :: Frame %d', frame_id);
            title(plot_title), grid on, xlabel(x_label), ylabel('Jacobian'), legend('analytical', 'numerical');
            
            subplot(2,2,3), hold on;
            plot(hessian_data(:, 2*state_id-1), hessian_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'red')});
            plot(hessian_num_data(:, 2*state_id-1), hessian_num_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'green')});
            %             plot(hessian_num_data(:, 2*state_id-1), matlab_hessian,...
            %                 'Color', col_rgb{strcmp(col_names,'blue')});
            plot_title=sprintf('Hessian :: Frame %d', frame_id);
            title(plot_title), grid on, xlabel(x_label), ylabel('Hessian'), legend('analytical', 'numerical');
            
            subplot(2,2,4), hold on;
            plot(hessian2_data(:, 2*state_id-1), hessian2_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'red')});
            plot_legend={'analytical'};
            
            plot(ihessian2_data(:, 2*state_id-1), ihessian2_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'cyan')});
            plot_legend=[plot_legend, {'initial'}];
            
            plot(chessian2_data(:, 2*state_id-1), chessian2_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'blue')});
            plot_legend=[plot_legend, {'current'}];
            
            plot(mjhessian2_data(:, 2*state_id-1), mjhessian2_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'orange')});
            plot_legend=[plot_legend, {'mj'}];
            
            plot(mjhhessian2_data(:, 2*state_id-1), mjhhessian2_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'black')});
            plot_legend=[plot_legend, {'mjh'}];
            
            %             plot(hessian_num_data(:, 2*state_id-1), hessian_num_data(:, 2*state_id),...
            %                 'Color', col_rgb{strcmp(col_names,'green')});
            %             plot_legend=[plot_legend, {'numerical'}];
            
            plot_title=sprintf('Hessian2 :: Frame %d', frame_id);
            title(plot_title), grid on, xlabel(x_label), ylabel('Hessian2'), legend(plot_legend);
            
        elseif plot_id==1
            plot(ssm_data(:, 2*state_id-1), ssm_data(:, 2*state_id));
            title('SSM Corner change'), xlabel(x_label), ylabel('Corner change');
        elseif plot_id==2
            hold on;
            plot(hessian2_nn_data(:, 2*state_id-1), hessian2_nn_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'red')});
            plot(hessian2_linear_data(:, 2*state_id-1), hessian2_linear_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'green')});
            plot(hessian2_bicubic_data(:, 2*state_id-1), hessian2_bicubic_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'blue')});
            plot(hessian2_bspl_data(:, 2*state_id-1), hessian2_bspl_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'orange')});
            plot(hessian_num_data(:, 2*state_id-1), hessian_num_data(:, 2*state_id),...
                'Color', col_rgb{strcmp(col_names,'black')});
            title('Hessian'), grid on, xlabel(x_label), ylabel('Hessian'),...
                legend('NN', 'Linear', 'Analytical Bicubic', 'BSpline', 'Numerical');
        elseif plot_id==3
            plot(feat_norm_data(:, 2*state_id-1), feat_norm_data(:, 2*state_id));
            title('Feature Norm'), grid on, xlabel(x_label), ylabel('Feature Norm');
        end
    end
    k = waitforbuttonpress;
    frame_id=frame_id+1;
end

% for state_id = 1:state_size
%     figure;
%     x_label = sprintf('param %d', state_id);
%     plot(norm_data(:, 2*state_id-1), norm_data(:, 2*state_id));
%     title('Norm'), xlabel(x_label), ylabel('Norm');
%
%     figure;
%     plot(jacobian_data(:, 2*state_id-1), jacobian_data(:, 2*state_id),...
%         jacobian_num_data(:, 2*state_id-1), jacobian_num_data(:, 2*state_id));
%     title('Jacobian'), xlabel(x_label), ylabel('Jacobian'), legend('analytical', 'numerical');
%
%     figure;
%     plot(hessian_data(:, 2*state_id-1), hessian_data(:, 2*state_id),...
%         hessian_num_data(:, 2*state_id-1), hessian_num_data(:, 2*state_id));
%     title('Hessian'), xlabel(x_label), ylabel('Hessian'), legend('analytical', 'numerical');
% end

