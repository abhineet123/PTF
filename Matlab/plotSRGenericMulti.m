clear all;
% close all;

getParamLists;
colRGBDefs;

db_root_dir = '../../Datasets';
sr_root_dir = '../C++/MTF/log/success_rates';

subplot_positions_with_xlabel_small={
    [0.0414915966386555 0.0884676145339652 0.343487394957983 0.894154818325434],...
    [0.417016806722689 0.0963665086887836 0.326680672268907 0.88783570300158],...
    [0.788758068444766 0.0189873417721519 0.213405797101449 0.963607594936709]
};
subplot_positions_without_xlabel_small={
    [0.0435924369747899 0.054140127388535 0.334033613445378 0.929936305732484],...
    [0.410797101449275 0.0525477707006369 0.330799537206187 0.928343949044586],...
    [0.788758068444768 0.0207006369426752 0.213405797101449 0.961783439490446]
};

subplot_positions_with_xlabel_large={
    [0.0472689075630252 0.10126582278481 0.327731092436974 0.876582278481013],...
    [0.410797101449275 0.109177215189873 0.320295335525515 0.870253164556962],...
    [0.788758068444766 0.0189873417721519 0.213405797101449 0.963607594936709]
};

subplot_positions_without_xlabel_large={
    [0.0472689075630252 0.10126582278481 0.327731092436974 0.876582278481013],...
    [0.410797101449275 0.109177215189873 0.320295335525515 0.870253164556962],...
    [0.788758068444766 0.0189873417721519 0.213405797101449 0.963607594936709]
};

subplot_positions_with_xlabel_large_new={
    [0.0472689075630252 0.0754098360655738 0.327731092436974 0.902438265200249],...
    [0.410797101449275 0.0765027322404372 0.320295335525515 0.902927647506398],...
    [0.788758068444766 0.0189873417721519 0.213405797101449 0.963607594936709]
};

subplot_positions_without_xlabel_large_new={
    [0.0472689075630252 0.0754098360655738 0.327731092436974 0.902438265200249],...
    [0.410797101449275 0.0765027322404372 0.320295335525515 0.902927647506398],...
    [0.788758068444766 0.0189873417721519 0.213405797101449 0.963607594936709]
};

plot_titles={};
plot_data_descs={};

desc_keys={'actor_id', 'seq_idxs_ids', 'plot_type', 'error_type', 'enable_subseq',...
    'file_name', 'mtf_sm', 'mtf_am', 'mtf_ssm', 'opt_gt_ssm', 'iiw',...
    'legend', 'color', 'line_style'};
bar_desc_keys={'actor_id', 'stats_file_name', 'stats_col', 'value', 'label', 'color', 'line_style'};

scatter_markers = ['o', '+', '*', 's', 'x',...
    'd', '^', 'v', '>', '<', 'p', 'h'];


plot_combined_data = 1;
seq_idxs_ids = 0;
n_ams = 0;

line_width = 3;

title_as_text_box = 1;

out_dir = 'plots';
save_plot = 0;
save_fmt='bmp';

n_subseq = 10;
mcd_err_thresh = 20;
jaccard_err_thresh = 0.90;
plot_data_type = 3;

n_runs = 1;
min_err_thr = 1;

overriding_error_type = -2;
read_from_bin = 1;

% add the name of datasets (and any subsets thereof) being plotted into the title
actor_in_title = 1;
plot_type_in_title = 0;

bar_width=0.5;
bar_line_width = 2;
bar_line_style = '-';
annotate_bars = 1;
annotation_font_size = 20;
annotation_col = [0, 0, 0];
% annotation_col = [];
horz_bar_plot = 1;
plot_average_in_bar = 1;
bar_with_legend = 0;
show_bar_legend = 1;
annotate_bar_legend = 1;
annotate_with_ratio = 1;
scatter_size = 128;
scatter_line_width=2;
intra_bar_gap = 0.65;
inter_bar_gap_ratio = 1.00;

reinit_frame_skip = 5;
reinit_err_thresh = 20;
plot_area_under_sr = 0;
show_area_in_legend = 1;
show_failures_in_legend = 1;
normalize_failures = 0;

plot_font_size = 24;
legend_font_size = 18;
col_legend = 0;
enable_xlabel = 1;
automatic_placement = 0;

ytick_precision = 20;
ytick_gap = 0.05;
x_min = 0;
x_max = 20;
y_min = 0;
y_max = 0.90;
adaptive_axis_range = 0;

% 0: SR without reinitialization
% 1: total number of failures
% 2: average error on successfull frames
% 3: average number of frames between consecutive failures
% 4: fraction of frames tracked successfully
% 5: Scatter plot with SR area on x axis and no. of failures on y axis
plot_types = [0];
reinit_at_each_frame = 0;
reset_at_each_frame = 0;
reset_to_init = 1;

% load generic plot configurations
% genericConfigsAM_gd;
% genericConfigsAM_stochastic;
% genericConfigsSM_robust;
genericConfigsSSM_thesis;

% plot_ids = [3042];
base_id = 3030;
plot_ids = [base_id, base_id + 1, base_id + 2];

% genericConfigsAM;
% genericConfigsSM;
% genericConfigsSSM;


% scatter AM
% plot_ids = [7002, 9002, 10002];
% plot_ids = [11002, 8002, 4002];
% plot_ids = [5002, 6002, 2002];
% plot_ids = [3002, 12002, 13002];


% scatter SM
% plot_ids = [1032, 2012, 3032];
% plot_ids = [4032, 5032, 6002];
% plot_ids = [2002, 3022, 4102];
% plot_ids = [5102, 5202, 6102];

% speed SM
% plot_ids = [1500, 2500, 3500];
% plot_ids = [4500, 5500, 6500];
% plot_ids = [2500, 3700, 4500];
% plot_ids = [5900, 5901, 6600];

% settings for synthetic sequences
syn_ssm = 'c8';
syn_ssm_sigma_ids = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28];
% syn_ssm_sigma_ids = [94, 95, 96, 97, 98, 99, 100, 101, 102, 103];

syn_ssm_sigmas = 1:10;
syn_ilm = '0';
syn_am_sigma_id = 9;
syn_add_noise = 1;
syn_noise_mean = 0;
syn_noise_sigma = 10;
syn_frame_id = 0;
syn_err_thresh = 4;
syn_plot_type = 0;

axis_label_x = 'Error Threshold';
axis_label_y = 'Success Rate';

% CRV
% plot_ids = [4911];
% plot_ids = [451, 461, 481];
% plot_ids = [491, 361, 381];
% plot_ids = [2571, 2572, 2573;
%     2574, 2575, 2576];
% plot_ids = [2511, 2512, 2513];

% plot_ids = [177, 170, 179];
% plot_ids = [181, 183, 172];
% plot_ids = [1762, 174, 176];

% ECCV
% plot_ids = [132,133,134];
% plot_ids = [135,136,137];
% plot_ids = [138,139,140];

% plot_ids = [1, 2, 3];
% plot_ids = [4, 5, 6];
% plot_ids = [7, 8, 9];

% plot_ids = [252, 253, 254];
% plot_ids = [255, 256, 257];


% plot_ids = [100, 110, 120];
% plot_ids = [101, 111, 121];

% plot_ids = [1762];
% plot_ids = [6, 7, 8, 9, 10];

n_rows=size(plot_ids, 1);
n_cols=size(plot_ids, 2);
plot_rows = n_rows;
plot_cols = n_cols;
if plot_area_under_sr
    plot_cols = plot_cols*2;
end
start_t = cputime;
for plot_type_ = plot_types
    
    set(0,'DefaultAxesFontName', 'Times New Roman');
    set(0,'DefaultAxesFontSize', plot_font_size);
    set(0,'DefaultAxesFontWeight', 'bold');
    
    plot_fig=figure;
    % grid minor;
    set (plot_fig, 'Units', 'normalized', 'Position', [0,0.03,1.00,0.88]);
    
    if plot_combined_data
        display('Using combined SR data');
    end
    
    
    subplot_id=1;
    for row_id=1:n_rows
        for col_id=1:n_cols
            fprintf('Generating subplot: %d, %d\n', row_id, col_id);
            plot_id=plot_ids(row_id, col_id);
            plot_data_desc=plot_data_descs{plot_data_type, plot_id};
            if isempty(plot_data_desc)
                error('Invalid plot id specified: %d', plot_id);
            end                   
          
            if automatic_placement
                if plot_font_size>16
                    subplot_pos_id = mod(subplot_id-1, length(subplot_positions_with_xlabel_large_new)) + 1;
                    if subplot_pos_id==0
                        subplot_pos_id=1;
                    end
                    if enable_xlabel
                        subplot_pos=subplot_positions_with_xlabel_large_new{subplot_pos_id};
                    else
                        subplot_pos=subplot_positions_without_xlabel_large_new{subplot_pos_id};
                    end                    
                else
                    subplot_pos_id=mod(subplot_id-1, length(subplot_positions_with_xlabel_small)) + 1;
                    if subplot_pos_id==0
                        subplot_pos_id=1;
                    end
                    if enable_xlabel
                        subplot_pos=subplot_positions_with_xlabel_small{subplot_pos_id};
                    else
                        subplot_pos=subplot_positions_without_xlabel_small{subplot_pos_id};
                    end
                end
                fprintf('Setting subplot %d position to:', subplot_id),disp(subplot_pos);
                % set(gca, 'Position', subplot_pos);
                subplot('position', subplot_pos);
            else
                subplot(plot_rows, plot_cols, subplot_id);
            end
            hold on, grid on;
            
            plot_title=plot_titles{plot_data_type, plot_id};
            
            n_lines=length(plot_data_desc);
            
            data_sr=cell(n_lines, 1);
            line_data=cell(n_lines, 1);
            sr_area_data=zeros(2, n_lines);
            
            if plot_data_desc{1}('actor_id')<0
                % bar plot
                plotSRBar;
                subplot_id=subplot_id+1;
                continue;                
            end  
            
            min_sr = inf;
            max_sr = -inf;
            plot_legend={};
            ax1=gca;
            root_dir=sr_root_dir;
            if overriding_error_type==2
                min_err_thr=0;
                reinit_err_thresh = jaccard_err_thresh;
                x_max = jaccard_err_thresh;
            end
            
            plot_type=plot_data_desc{1}('plot_type');
            if plot_type_==5 || isempty(plot_type) || plot_type<0
                plot_type=plot_type_;                
            end
            if plot_type == 5
                plotSRScatter;
                subplot_id=subplot_id+1;
                continue;
            end
            reinit_on_failure=plot_type;
            if reinit_at_each_frame
                reinit_on_failure = 0;
                root_dir=sprintf('%s/reinit',root_dir);
            elseif reset_at_each_frame
                reinit_on_failure = 0;
                if reset_to_init
                    root_dir=sprintf('%s/reset_to_init',root_dir);
                else
                    root_dir=sprintf('%s/reset',root_dir);
                end              
            elseif reinit_on_failure
                if reinit_err_thresh==int32(reinit_err_thresh)
                    root_dir=sprintf('%s/reinit_%d_%d',root_dir, reinit_err_thresh, reinit_frame_skip);
                else
                    root_dir=sprintf('%s/reinit_%4.2f_%d',root_dir, reinit_err_thresh, reinit_frame_skip);
                end
                ax2 = ax1;
                failure_data=zeros(n_lines, 1);
            end           
            plot_synthetic_sr = 0;
            for line_id=1:n_lines                
                desc=plot_data_desc{line_id};
                actor_ids=desc('actor_id');
                n_actors=length(actor_ids);
                opt_gt_ssm = desc('opt_gt_ssm');
                enable_subseq = desc('enable_subseq');
                error_type=desc('error_type');
                curr_n_runs = n_runs;
                if overriding_error_type>=0
                    error_type=overriding_error_type;
                end
                
                if reinit_on_failure
                    enable_subseq = 0;
                end
                if length(actor_ids)>1
                    plot_combined_data=1;
                end
                data_sr{line_id}=[];
                total_frames=0;
                scuccessful_frames=[];
                if reinit_on_failure
                    total_valid_frames=0;
                    total_failures=0;
                    total_error=0;
                end
                seq_idxs_ids=desc('seq_idxs_ids');
                if length(seq_idxs_ids)==1
                    % use the same seq_idxs_ids for all actors
                    seq_idxs_ids=repmat(seq_idxs_ids, n_actors, 1);
                elseif length(seq_idxs_ids)~= n_actors
                    error('Invalid seq_idxs_ids provided as it should be either a scalar or a vector of the same size as n_actors: %d', n_actors);
                end
                
                file_name = desc('file_name');
                for actor_ids_id=1:n_actors
                    actor_id=actor_ids(actor_ids_id);
                    actor=actors{actor_id+1};
                    data_fname=sprintf('%s/%s/sr',root_dir, actor);
                    %if overriding_seq_id>=0
                        %data_fname=sprintf('%s_%s', data_fname, sequences{actor_id+1}{overriding_seq_id+1});
                        %seq_idxs = [overriding_seq_id];
                    %end
                    if strcmp(actor, 'Synthetic')
                        plotSRSynthetic;
                        continue;
                    else
                        actor_n_frames=importdata(sprintf('%s/%s/n_frames.txt', db_root_dir, actor));
                        seq_idxs=actor_idxs{actor_id+1}{seq_idxs_ids(actor_ids_id)+1};
                        if ~isempty(file_name)
                            data_fname=sprintf('%s_%s', data_fname, file_name);
                            if ~isempty(desc('mtf_sm'))
                                curr_n_runs=desc('mtf_sm');
                            end
                        else
                            data_fname=sprintf('%s_%s_%s_%s_%d', data_fname,...
                                desc('mtf_sm'), desc('mtf_am'), desc('mtf_ssm'), desc('iiw'));
                        end
                        if(opt_gt_ssm ~= '0')
                            data_fname=sprintf('%s_%s', data_fname, opt_gt_ssm);
                        end
                        if(enable_subseq)
                            data_fname=sprintf('%s_subseq_%d', data_fname, n_subseq);
                        end
                        if error_type
                            data_fname=sprintf('%s_%s', data_fname, error_types{error_type + 1});
                        end
                        if curr_n_runs > 1
                            data_fname=sprintf('%s_%s_runs', data_fname, curr_n_runs);
                        end
                        if read_from_bin
                            data_fname=sprintf('%s.bin', data_fname);
                        else
                            data_fname=sprintf('%s.txt', data_fname);
                        end

                        fprintf('Reading data for plot line %d actor %d from: %s\n',...
                            line_id, actor_id, data_fname);
                        if read_from_bin
                            data_fid=fopen(data_fname);
                            data_rows=fread(data_fid, 1, 'uint32', 'a');  
                            data_cols=fread(data_fid, 1, 'uint32', 'a');
                            actor_data_sr=fread(data_fid, [data_cols, data_rows], 'float64', 'a');    
                            actor_data_sr = actor_data_sr';
                            fclose(data_fid);
                        else
                            actor_data_sr=importdata(data_fname);
                        end
                        if reinit_on_failure
                            % exclude the 0s in the first column and
                            % include combined data in last column
                            % reinit_seq_idxs=[seq_idxs+1 size(actor_data_sr, 2)];
                            reinit_seq_idxs = seq_idxs + 1;
                            frames_per_failure=actor_data_sr(end, reinit_seq_idxs);
                            failure_counts=actor_data_sr(end-1, reinit_seq_idxs);
                            avg_err=actor_data_sr(end-2, reinit_seq_idxs);

                            valid_frames=round((failure_counts+1).*frames_per_failure);

                            % actor_total_failures=failure_counts(end);
                            actor_total_failures=sum(failure_counts);
                            % actor_valid_frames=round((actor_total_failures+1)*cmb_frames_per_failure);
                            actor_valid_frames=sum(valid_frames);
                            % cmd_avg_err=avg_err(end);
                            actor_total_error=sum(valid_frames.*avg_err);

                            total_failures = total_failures + actor_total_failures;
                            total_valid_frames = total_valid_frames + actor_valid_frames;
                            total_error = total_error + actor_total_error;

                            failure_data(line_id, 1) = actor_total_failures;
                            % remove the last 3 lines specific to reinit data
                            actor_data_sr(end-2:end, :)=[];
                        end
                        % first frame in each sequence where tracker is initialized
                        % is not considered for computing the total tracked frames

                        if enable_subseq
                            actor_subseq_n_frames=importdata(sprintf('%s/%s/subseq_n_frames_%d.txt',...
                                db_root_dir, actor, n_subseq));
                            seq_n_frames = actor_subseq_n_frames(seq_idxs).';
                            actor_total_frames = sum(seq_n_frames);
                        else
                            % actor_total_frames=sum(actor_n_frames.data)-length(actor_n_frames.data);
                            seq_n_frames = actor_n_frames.data(seq_idxs).';
                            actor_total_frames = sum(seq_n_frames)- length(seq_idxs);
                        end
                        total_frames = total_frames + actor_total_frames;

                        % actor_combined_sr = actor_data_sr(:, end);
                        % actor_successful_frames = actor_combined_sr.*actor_total_frames;
                        seq_sr = actor_data_sr(:, seq_idxs + 1); % first column contains the thresholds
                        seq_successful_frames = repmat(seq_n_frames, size(seq_sr, 1), 1).*seq_sr;
                        actor_successful_frames = sum(seq_successful_frames, 2);
                        if isempty(scuccessful_frames)
                            scuccessful_frames = actor_successful_frames;
                        else
                            scuccessful_frames = scuccessful_frames + actor_successful_frames;
                        end

                        if isempty(data_sr{line_id})
                            % assume that the error thresholds are same for all
                            % datasets
                            % data_sr{line_id}=actor_data_sr(:, 1:end-1);
                            data_sr{line_id}=actor_data_sr(:, [1 seq_idxs + 1]);
                        else
                            % omit the first and last columns ontaining the error
                            % thresholds and the combined SR respectively
                            % data_sr{line_id}=horzcat(data_sr{line_id}, actor_data_sr(:, 2:end-1));
                            data_sr{line_id}=horzcat(data_sr{line_id}, actor_data_sr(:, seq_idxs + 1));
                        end
                    end
                end
                if plot_synthetic_sr
                    x_min = syn_ssm_sigmas(1);
                    x_max = syn_ssm_sigmas(end);
                    sr_area_data(1, line_id) = trapz(syn_ssm_sigmas,line_data{line_id});
                    plot(syn_ssm_sigmas, line_data{line_id},...
                        'Parent', ax1,...
                        'Color', col_rgb{strcmp(col_names,desc('color'))},...
                        'LineStyle', desc('line_style'),...
                        'LineWidth', line_width);
                    if adaptive_axis_range
                        max_line_data=max(line_data{line_id});
                        if max_line_data>max_sr
                            max_sr=max_line_data;
                        end
                        min_line_data=min(line_data{line_id});
                        if min_line_data < min_sr
                            min_sr = min_line_data;
                        end
                        if syn_plot_type ~= 0
                            ytick_gap = (max_sr - min_sr)/10.0;
                        end
                    end
                    if syn_plot_type==0
                        axis_label_y = sprintf('Success Rate with threshold %4.2f',...
                            syn_err_thresh);
                    else
                        axis_label_y = 'Average Error';
                    end
                    axis_label_x = '\sigma_{syn}';
                else
                    if reinit_on_failure
                        if plot_type==1
                            reinit_data = total_failures;
                        elseif plot_type==2
                            overall_avg_error = total_error / total_valid_frames;
                            reinit_data = overall_avg_error;
                        elseif plot_type==3
                            total_frames_per_failure = total_frames / total_failures;
                            reinit_data=total_valid_frames / total_failures;
                        elseif plot_type==4
                            reinit_data = total_valid_frames / (total_frames-total_failures);
                        else
                            error('Invalid plot_type: %d', plot_type);
                        end
                        bar(line_id, reinit_data,...
                            'Parent', ax2,...
                            'BarWidth', bar_width,...
                            'LineStyle', desc('line_style'),...
                            'LineWidth', bar_line_width,...
                            'FaceColor', col_rgb{strcmp(col_names,desc('color'))},...
                            'EdgeColor', col_rgb{strcmp(col_names,'black')});
                        if annotate_bars
                            annotation('textbox',...
                                [0 0 0.3 0.15],...
                                'String',sprintf('%d', round(failure_data(line_id))),...
                                'FontSize',20,...
                                'FontWeight','bold',...
                                'FontName','Times New Roman',...
                                'LineStyle','-',...
                                'EdgeColor','none',...
                                'LineWidth',2,...
                                'BackgroundColor','none',...
                                'Color',[0 0 0],...
                                'FitBoxToText','on');
                        end
                        if line_id==1
                            hold on;
                        end
                    end
                    data_sr{line_id}=horzcat(data_sr{line_id}, scuccessful_frames./total_frames);
                    err_thr=data_sr{line_id}(:, 1);
                    if plot_combined_data
                        line_data{line_id}=data_sr{line_id}(:, end);
                    else
                        % first column has error thresholds and last one has
                        % combined SR
                        line_data{line_id} = mean(data_sr{line_id}(:, 2:end-1), 2);
                    end
                    if ~reinit_on_failure
                        if min_err_thr>0
                            valid_idx=err_thr>=min_err_thr;
                            err_thr=err_thr(valid_idx);
                            line_data{line_id}=line_data{line_id}(valid_idx);
                        end
                        sr_area_data(1, line_id) = trapz(err_thr,line_data{line_id});
                        plot(err_thr, line_data{line_id},...
                            'Parent',ax1,...
                            'Color', col_rgb{strcmp(col_names,desc('color'))},...
                            'LineStyle', desc('line_style'),...
                            'LineWidth', line_width);
                        if adaptive_axis_range
                            max_line_data=max(line_data{line_id});
                            if max_line_data>max_sr
                                max_sr=max_line_data;
                            end
                            min_line_data=min(line_data{line_id});
                            if min_line_data < min_sr
                                min_sr = min_line_data;
                            end
                        end
                    end
                end
                if ~isempty(desc('legend'))
                    if ~reinit_on_failure && show_area_in_legend
                        curr_legend=sprintf('%s:%6.3f', desc('legend'), sr_area_data(1, line_id));
                    elseif plot_type==1 && show_failures_in_legend
                        curr_legend=sprintf('%s:%d', desc('legend'), total_failures);
                    else
                        curr_legend=desc('legend');
                    end
                    plot_legend=[plot_legend {curr_legend}];
                end
            end
            if col_legend
                h_legend=columnlegend(2,plot_legend,'NorthWest', 'boxon');
            else
                h_legend=legend(ax1, plot_legend);
            end
            set(h_legend,'FontSize',legend_font_size);
            set(h_legend,'FontWeight','bold');
            if reinit_on_failure
                labels=cell(n_lines, 1);
                %             ax2 = axes;
                %             bar plot of failure counts
                %             bar_handles=bar(failure_data, 'Parent',ax2);
                for line_id=1:n_lines
                    desc=plot_data_desc{line_id};
                    %                 bar(line_id, failure_data(line_id),...
                    %                     'Parent',ax2,...
                    %                     'BarWidth', 0.2,...
                    %                     'LineWidth', 5,...
                    %                     'FaceColor', 'None',...
                    %                     'EdgeColor', col_rgb{strcmp(col_names,desc('color'))});
                    %                 if line_id==1
                    %                     hold on;
                    %                 end
                    %                 set(bar_handles(line_id), 'FaceColor', col_rgb{strcmp(col_names,desc('color'))});
                    labels{line_id}=desc('legend');
                    %                 annotation('textbox',...
                    %                     [0 0 0.3 0.15],...
                    %                     'String',num2str(round(failure_data(line_id))),...
                    %                     'FontSize',20,...
                    %                     'FontWeight','bold',...
                    %                     'FontName','Times New Roman',...
                    %                     'LineStyle','-',...
                    %                     'EdgeColor','none',...
                    %                     'LineWidth',2,...
                    %                     'BackgroundColor','none',...
                    %                     'Color',[0 0 0],...
                    %                     'FitBoxToText','on');
                end
                ax1_pos = get(ax1, 'Position'); % position of first axes
                %             ax1_ar = get(ax1, 'PlotBoxAspectRatio'); % position of first axes
                set(ax2, 'Position', ax1_pos);
                %             set(ax2, 'PlotBoxAspectRatio', ax1_ar);
                
                set(ax2, 'XAxisLocation', 'bottom');
                set(ax2, 'YAxisLocation', 'left');
                set(ax2, 'Color', 'None');
                set(ax2, 'XLim', [0, n_lines+1]);
                set(ax2, 'XTick', 1:n_lines);
                set(ax2, 'XTickLabel', []);
                set(ax2,'box','off');
                if reinit_on_failure==1
                    y_label= 'Number of Failures';
                elseif reinit_on_failure==2
                    y_label= 'Average alignment error';
                elseif reinit_on_failure==3
                    y_label= 'Average Frames between Failures';
                elseif reinit_on_failure==4
                    y_label='Fraction of frames tracked successfully';
                end
                if plot_type_in_title
                    plot_title=sprintf('%s :: %s', plot_title, y_label);
                end
                ylabel(ax2, y_label);
            else
                if adaptive_axis_range
                    y_min=floor(min_sr*ytick_precision)/ytick_precision;
                    y_max=ceil(max_sr*ytick_precision)/ytick_precision;
                    fprintf('min_sr: %f\t max_sr: %f\n', min_sr, max_sr);
                    fprintf('y_min: %f\t y_max: %f\n', y_min, y_max);
                end
                if y_min~=y_max
                    set(ax1,'YLim', [y_min y_max]);
                    set(ax1,'YTick', y_min:ytick_gap:y_max);
                end
                if x_min~=x_max
                    set(ax1,'XLim', [x_min x_max]);
                end                
                if reinit_on_failure
                    set(ax1, 'XAxisLocation', 'bottom');
                    set(ax1, 'YAxisLocation', 'right');
                end
                %         set(ax1,'Color', 'r');
                if enable_xlabel
                    xlabel(ax1, axis_label_x);
                end
                ylabel(ax1, axis_label_y);
                if plot_type_in_title
                    plot_title=sprintf('%s :: SR Plot', plot_title);
                end
            end
            if actor_in_title
                % assuming that all lines have the same seq_idxs_ids
                seq_idxs_ids=plot_data_desc{1}('seq_idxs_ids');
                plot_title = sprintf('%s (',plot_title);
                iter_id=1;
                for actor_id=plot_data_desc{1}('actor_id')
                    plot_title = sprintf('%s %s', plot_title, actors{actor_id+1});
                    plot_data_desc{line_id};
                    if seq_idxs_ids(iter_id) ~= 0
                        % add label of subset if entire dataset is not being used
                        plot_title = sprintf('%s (%s)', plot_title, actor_idx_types{actor_id+1}{seq_idxs_ids(iter_id) + 1});
                        iter_id=iter_id+1;
                    end
                end
                plot_title = sprintf('%s )', plot_title);
                if enable_subseq
                    plot_title = sprintf('%s (SubSeq %d)', plot_title, n_subseq);
                end
            end
            if error_type
                plot_title = sprintf('%s [%s error]', plot_title, error_types{error_type + 1});
            end
            if title_as_text_box
                annotation('textbox',...
                    [0 0 0.3 0.15],...
                    'String',plot_title,...
                    'FontSize',24,...
                    'FontWeight','bold',...
                    'FontName','Times New Roman',...
                    'LineStyle','-',...
                    'EdgeColor','none',...
                    'LineWidth',2,...
                    'BackgroundColor','none',...
                    'Color',[0 0 0],...
                    'FitBoxToText','on');
            else
                title(plot_title, 'interpreter', 'none');
            end
            subplot_id=subplot_id+1;
            if plot_area_under_sr
                subplot(plot_rows, plot_cols, subplot_id);
%                 bar(line_id, sr_area_data(line_id),...
%                     'BarWidth', bar_width,...
%                     'LineStyle', desc('line_style'),...
%                     'LineWidth', bar_line_width,...
%                     'FaceColor', col_rgb{strcmp(col_names,desc('color'))},...
%                     'EdgeColor', col_rgb{strcmp(col_names,'black')});
                b_area=barh(sr_area_data);
                for line_id=1:n_lines
                    desc=plot_data_desc{line_id};
                    labels{line_id}=desc('legend');
                    set(b_area(line_id), 'LineStyle', desc('line_style'));
                    set(b_area(line_id), 'FaceColor', col_rgb{strcmp(col_names,desc('color'))});
                    set(b_area(line_id), 'EdgeColor', col_rgb{strcmp(col_names,'black')});
                end
                set(gca, 'YTick', 1:n_lines);
                set(gca, 'YTickLabel', labels, 'DefaultTextInterpreter', 'none');
                subplot_id = subplot_id + 1;
            end            
        end
    end
    set(plot_fig, 'Name', plot_title);
    if save_plot
        out_fname=strrep(plot_title, '::', '_');
        out_fname=strrep(out_fname, ' ', '_');
        out_fname=strrep(out_fname, '/', '_');
        out_path=sprintf('%s/%s.%s', out_dir, out_fname, save_fmt);
        fprintf('Saving plot to: %s\n', out_path);
        %         set (gcf, 'PaperUnits', 'normalized', 'PaperOrientation', 'portrait', 'PaperPosition', [0, 0, 1.8, 0.8]);
        saveas(gcf, out_path, save_fmt);
    end
end
end_t = cputime;
fprintf('Time taken: %f\n', end_t-start_t);


