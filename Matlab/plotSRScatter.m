if reinit_err_thresh==int32(reinit_err_thresh)
    reinit_root_dir=sprintf('%s/reinit_%d_%d',root_dir, reinit_err_thresh, reinit_frame_skip);
else
    reinit_root_dir=sprintf('%s/reinit_%4.2f_%d',root_dir, reinit_err_thresh, reinit_frame_skip);
end
failure_data=zeros(n_lines, 1);
total_failures=zeros(n_lines, 1);
total_error=zeros(n_lines, 1);
total_valid_frames=zeros(n_lines, 1);
sr_area=zeros(n_lines, 1);
scatter_cols=zeros(n_lines, 3);

plot_legend = {};
hold on;
for line_id=1:n_lines
    desc=plot_data_desc{line_id};
    actor_ids=desc('actor_id');
    n_actors=length(actor_ids);
    opt_gt_ssm = desc('opt_gt_ssm');
    enable_subseq = desc('enable_subseq');
    error_type=desc('error_type');
    if overriding_error_type>=0
        error_type=overriding_error_type;
    end   

    if length(actor_ids)>1
        plot_combined_data=1;
    end
    
    data_sr{line_id}=[];
    total_frames=0;
    scuccessful_frames=[];
    
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
        
        sr_data_fname=sprintf('%s/%s/sr',root_dir, actor);
        reinit_data_fname=sprintf('%s/%s/sr',reinit_root_dir, actor);
        
        %if overriding_seq_id>=0
        %data_fname=sprintf('%s_%s', data_fname, sequences{actor_id+1}{overriding_seq_id+1});
        %seq_idxs = [overriding_seq_id];
        %end
        actor_n_frames=importdata(sprintf('%s/%s/n_frames.txt', db_root_dir, actor));
        seq_idxs=actor_idxs{actor_id+1}{seq_idxs_ids(actor_ids_id)+1};
        if ~isempty(file_name)
            sr_data_fname=sprintf('%s_%s', sr_data_fname, file_name);
            reinit_data_fname=sprintf('%s_%s', reinit_data_fname, file_name);
            if ~isempty(desc('mtf_sm'))
                n_runs=desc('mtf_sm');
            end
        else
            sr_data_fname=sprintf('%s_%s_%s_%s_%d', sr_data_fname,...
                desc('mtf_sm'), desc('mtf_am'), desc('mtf_ssm'), desc('iiw'));
            reinit_data_fname=sprintf('%s_%s_%s_%s_%d', reinit_data_fname,...
                desc('mtf_sm'), desc('mtf_am'), desc('mtf_ssm'), desc('iiw'));
        end
        if(opt_gt_ssm ~= '0')
            sr_data_fname=sprintf('%s_%s', sr_data_fname, opt_gt_ssm);
            reinit_data_fname=sprintf('%s_%s', reinit_data_fname, opt_gt_ssm);
        end
        if(enable_subseq)
            sr_data_fname=sprintf('%s_subseq_%d', sr_data_fname, n_subseq);
        end
        if error_type
            sr_data_fname=sprintf('%s_%s', sr_data_fname, error_types{error_type + 1});
            reinit_data_fname=sprintf('%s_%s', reinit_data_fname, error_types{error_type + 1});
        end
        if n_runs > 1
            sr_data_fname=sprintf('%s_%s_runs', sr_data_fname, n_runs);
            reinit_data_fname=sprintf('%s_%s_runs', reinit_data_fname, n_runs);
        end
        if read_from_bin
            sr_data_fname=sprintf('%s.bin', sr_data_fname);
            reinit_data_fname=sprintf('%s.bin', reinit_data_fname);
        else
            sr_data_fname=sprintf('%s.txt', sr_data_fname);
            reinit_data_fname=sprintf('%s.txt', reinit_data_fname);
        end
        
        fprintf('Reading reinit data for plot line %d actor %d from: %s\n',...
            line_id, actor_id, reinit_data_fname);
        fprintf('Reading SR data for plot line %d actor %d from: %s\n',...
            line_id, actor_id, sr_data_fname);
        if read_from_bin
            reinit_data_fid=fopen(reinit_data_fname);
            reinit_data_rows=fread(reinit_data_fid, 1, 'uint32', 'a');
            reinit_data_cols=fread(reinit_data_fid, 1, 'uint32', 'a');
            reinit_actor_data=fread(reinit_data_fid, [reinit_data_cols, reinit_data_rows], 'float64', 'a');
            reinit_actor_data = reinit_actor_data';
            fclose(reinit_data_fid);
            
            sr_data_fid=fopen(sr_data_fname);
            sr_data_rows=fread(sr_data_fid, 1, 'uint32', 'a');
            sr_data_cols=fread(sr_data_fid, 1, 'uint32', 'a');
            sr_actor_data=fread(sr_data_fid, [sr_data_cols, sr_data_rows], 'float64', 'a');
            sr_actor_data = sr_actor_data';
            fclose(sr_data_fid);
        else
            reinit_actor_data=importdata(reinit_data_fname);
            sr_data_fname=importdata(sr_data_fname);
        end
        % exclude the 0s in the first column and
        % include combined data in last column
        % reinit_seq_idxs=[seq_idxs+1 size(actor_data_sr, 2)];
        reinit_seq_idxs = seq_idxs + 1;
        frames_per_failure=reinit_actor_data(end, reinit_seq_idxs);
        failure_counts=reinit_actor_data(end-1, reinit_seq_idxs);
        avg_err=reinit_actor_data(end-2, reinit_seq_idxs);
        
        valid_frames=round((failure_counts+1).*frames_per_failure);
        
        % actor_total_failures=failure_counts(end);
        actor_total_failures=sum(failure_counts);
        % actor_valid_frames=round((actor_total_failures+1)*cmb_frames_per_failure);
        actor_valid_frames=sum(valid_frames);
        % cmd_avg_err=avg_err(end);
        actor_total_error=sum(valid_frames.*avg_err);
        
        total_failures(line_id, 1) = total_failures(line_id, 1) + actor_total_failures;
        total_valid_frames(line_id, 1) = total_valid_frames(line_id, 1) + actor_valid_frames;
        total_error(line_id, 1) = total_error(line_id, 1) + actor_total_error;
        
        failure_data(line_id, 1) = actor_total_failures;
        % remove the last 3 lines specific to reinit data
        reinit_actor_data(end-2:end, :)=[];
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
        seq_sr = sr_actor_data(:, seq_idxs + 1); % first column contains the thresholds
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
            data_sr{line_id}=sr_actor_data(:, [1 seq_idxs + 1]);
        else
            % omit the first and last columns ontaining the error
            % thresholds and the combined SR respectively
            % data_sr{line_id}=horzcat(data_sr{line_id}, actor_data_sr(:, 2:end-1));
            data_sr{line_id}=horzcat(data_sr{line_id}, sr_actor_data(:, seq_idxs + 1));
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
    if min_err_thr>0
        valid_idx=err_thr>=min_err_thr;
        err_thr=err_thr(valid_idx);
        line_data{line_id}=line_data{line_id}(valid_idx);
    end
    sr_area(line_id, 1) = trapz(err_thr,line_data{line_id});
    plot_legend=[plot_legend {desc('legend')}];    
    scatter_cols(line_id, :) = col_rgb{strcmp(col_names,desc('color'))};
    marker_id = mod(line_id-1, length(scatter_markers)) + 1;     
    
    if normalize_failures
        total_failures(line_id, 1) = total_failures(line_id, 1)/total_frames;
    end
        
    scatter(sr_area(line_id, 1), total_failures(line_id, 1), scatter_size,...
        'Marker', scatter_markers(marker_id),...
        'MarkerFaceColor', scatter_cols(line_id, :),...
        'MarkerEdgeColor', scatter_cols(line_id, :),...
        'LineWidth', scatter_line_width);
end
% scatter(ax1, sr_area, total_failures, scatter_size, scatter_cols, 'filled');
plot_legend = cellstr(plot_legend);
if col_legend
    h_legend=columnlegend(2,plot_legend,'NorthWest', 'boxon');
else
    h_legend=legend(ax1, plot_legend);
end
set(h_legend,'FontSize',legend_font_size);
set(h_legend,'FontWeight','bold');
xlabel(ax1, 'Area under SR Curve');
ylabel(ax1, 'Number of Failures');

