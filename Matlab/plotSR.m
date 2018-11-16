clear all;
colRGBDefs;
getParamLists;

actor_id = 1;
mtf_am = 'miIN100r100i';
mtf_ssm = '8';
opt_gt_ssm = '0';
iiw = 1;
plot_type = 2;
plot_cmbd = 1;
use_opt_gt = 0;
plot_learning = 0;
plot_pf = 0;
seq_id = -1;

if ~strcmp(opt_gt_ssm, '0')
    use_opt_gt=1;
    fprintf('Using optimized ground truth with ssm: %s\n', opt_gt_ssm);
end
mtf_sms={
    'fclk',...
    'iclk',...
    'falk',...
    'ialk',...
    'nesm',...
    'aesm',...
    'nnic'
    };
mtf_sm_iiw={0, 0, 1, 1, 0, 1, 0};

sm_ids=[1, 2, 5];
% mtf_sm_labels=struct('name', mtf_sms, 'iiw', mtf_sm_iiw);

n_sms = length(sm_ids);

sm_sr=cell(n_sms, 1);
sm_sr_mean=cell(n_sms, 1);

plot_cols={
    'orange',...
    'green',...
    'blue',...
    'red',...
    'magenta',...
    'purple',...
    'dark_green'
    };
plot_legend=upper(mtf_sms(sm_ids));

actor = actors{actor_id+1};

if seq_id<0
    if strcmp(actor, 'UCSB')
        min_seq_id = 0;
        max_seq_id = 95;
    %     plot_title=sprintf('SM Success Rates for Normalized SSD/8DOF on UCSB Dataset');
        plot_title=sprintf('SM Success Rates for %s/%sDOF on %s Dataset', mtf_am, mtf_ssm, actor);
    %     plot_title=sprintf('SM Success Rates for RSCV/2DOF on UCSB Dataset (2DOF GT)');
    elseif strcmp(actor, 'TMT')
        if plot_type==0
            min_seq_id = 0;
            max_seq_id = 49;
            plot_title=sprintf('Success Rate %s NL', actor);
        elseif plot_type==1
            min_seq_id = 50;
            max_seq_id = 97;
            plot_title=sprintf('Success Rate %s DL', actor);
        elseif plot_type==2
            min_seq_id = 0;
            max_seq_id = 97;
    %     plot_title=sprintf('SM Success Rates for NCC/2DOF SSM on TMT Dataset (2DOF GT)');
        plot_title=sprintf('SM Success Rates for %s/%sDOF on %s Dataset', mtf_am, mtf_ssm, actor);
    %         plot_title=sprintf('SM Success Rates for Normalized SSD/8DOF on TMT Dataset');
        else
            error('Invalid plot type: %d', plot_type);
        end
    else
        error('Invalid actor: %s', actor);
    end
    min_seq_id = min_seq_id+2;
    max_seq_id = max_seq_id+2;   
else
    seq_name = sequences{actor_id + 1}{seq_id + 1};
    actor_name = actors{actor_id + 1};
    plot_title=sprintf('SM Success Rates for %s/%sDOF on %s: %s',...
        upper(mtf_am), mtf_ssm, actor_name, seq_name);
end

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 14);
set(0,'DefaultAxesFontWeight', 'bold');

figure, hold on, grid on, title(plot_title, 'interpreter', 'none');
line_width=3;

for i=1:n_sms
    sm_id=sm_ids(i);
    mtf_sm=mtf_sms{sm_id};
    iiw=mtf_sm_iiw{sm_id};
    if use_opt_gt
        data_fname=sprintf('success_rates/sr_%s_%s_%s_%s_%d_%s.txt', actor, mtf_sm, mtf_am, mtf_ssm, iiw, opt_gt_ssm)
    else
        data_fname=sprintf('success_rates/sr_%s_%s_%s_%s_%d.txt', actor, mtf_sm, mtf_am, mtf_ssm, iiw)
    end
    sm_sr{sm_id}=importdata(data_fname);
    
    if seq_id<0     
        sm_sr_mean{sm_id}=mean(sm_sr{sm_id}(:, min_seq_id:max_seq_id), 2);
    else
        sm_sr_mean{sm_id}=sm_sr{sm_id}(:,  seq_id+2);
    end
    err_thr=sm_sr{sm_id}(:, 1);
    plot(err_thr, sm_sr_mean{sm_id}, 'Color', col_rgb{strcmp(col_names,plot_cols{sm_id})},...
        'LineStyle', '-', 'LineWidth', line_width);    
end

learning_cols={
    'sandy_brown',...
    'olive_drab',...
    'black'
    };

if plot_learning
    learning_trackers={
        'tld',...
        'kcf',...
        'dsst'
        };
    learning_legend=upper(learning_trackers);
    plot_legend=[plot_legend, learning_legend];

    n_learning=length(learning_trackers);
    learning_sr=cell(n_learning, 1);
    learning_sr_mean=cell(n_learning, 1);
    for tracker_id=1:n_learning
        tracker=learning_trackers{tracker_id};
        if use_opt_gt
            data_fname=sprintf('success_rates/sr_%s_%s_rscv_8_1_%s.txt', actor, tracker, opt_gt_ssm)
        else
            data_fname=sprintf('success_rates/sr_%s_%s_rscv_8_1.txt', actor, tracker)
        end
        learning_sr{tracker_id}=importdata(data_fname);
        if seq_id<0     
            learning_sr_mean{tracker_id}=mean(learning_sr{tracker_id}(:, min_seq_id:max_seq_id), 2);
        else
            learning_sr_mean{tracker_id}=learning_sr{tracker_id}(:,  seq_id+2);
        end   
        plot(err_thr, learning_sr_mean{tracker_id}, 'Color', col_rgb{strcmp(col_names,learning_cols{tracker_id})},...
            'LineStyle', '-', 'LineWidth', line_width);
    end
    if plot_pf
        data_fname=sprintf('success_rates/sr_%s_pf_rscv_2_1.txt', actor)
        pf_sr=importdata(data_fname);
        if seq_id<0     
            pf_sr_mean=mean(pf_sr(:, min_seq_id:max_seq_id), 2);
        else
            pf_sr_mean=pf_sr(:,  seq_id+2);
        end        
        plot(err_thr, pf_sr_mean, 'Color', col_rgb{strcmp(col_names,'cyan')},...
            'LineStyle', '-', 'LineWidth', line_width);
        plot_legend=[plot_legend, {'PF'}];
    end
end 

h_legend=legend(plot_legend);
set(h_legend,'FontSize',10);

xlabel('Error Threshold');
ylabel('Success Rate');
% axis([0 20 0 1]);

if plot_cmbd
    plot_title=strcat(plot_title, '(Combined)');
    figure, hold on, grid on, title(plot_title);
    line_width = 2.5;
    
    sm_sr_combined=cell(n_sms, 1);
    for i=1:n_sms
        sm_id=sm_ids(i);
        sm_sr_combined{sm_id}=sm_sr{sm_id}(:, end);
         plot(err_thr, sm_sr_combined{sm_id}, 'Color', col_rgb{strcmp(col_names,plot_cols{sm_id})},...
             'LineStyle', '-', 'LineWidth', line_width);         
    end
    if plot_learning
        for tracker_id=1:n_learning
            learning_sr_cmbd=learning_sr{tracker_id}(:, end);
            plot(err_thr, learning_sr_cmbd, 'Color', col_rgb{strcmp(col_names,learning_cols{tracker_id})},...
                'LineStyle', '-', 'LineWidth', line_width);
        end      
    end
    legend(plot_legend);
    xlabel('error threshold');
    ylabel('success rate');
    axis([0 20 0 1]);
end

