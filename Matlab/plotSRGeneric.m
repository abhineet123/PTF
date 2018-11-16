clear all;
% close all;

getParamLists;
colRGBDefs;

root_dir='../C++/MTF/log';

opt_gt_ssm = '0';
plot_combined_data = 1;

mean_idx_id = 0;
n_ams = 0;

plot_rows = 1;
plot_cols = 1;

y_min = 0;
y_max = 1;

plot_font_size = 30;
legend_font_size = 20;

plot_titles={};
plot_data_descs={};


desc_keys={'actor_id', 'mean_idx_id', 'mtf_sm', 'mtf_am', 'mtf_ssm', 'iiw',...
    'legend', 'color', 'line_style'};

%load all generic plot configurations
% genericConfigs;
genericConfigsCRVSM;
genericConfigsCRVAM;

adaptive_axis_range = 1;

plot_id = 166;


plot_title=plot_titles{plot_id};
plot_data_desc=plot_data_descs{plot_id};

n_lines=length(plot_data_desc);

data_sr=cell(n_ams, 1);
line_data=cell(n_ams, 1);

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', plot_font_size);
set(0,'DefaultAxesFontWeight', 'bold');

plot_fig=figure;

hold on, grid on, title(plot_title, 'interpreter', 'none');
% grid minor;
line_width = 3;
set(plot_fig, 'Name', plot_title);
set (plot_fig, 'Units', 'normalized', 'Position', [0,0,1,1]);

if plot_combined_data
    display('Using combined SR data');
end

min_sr = 1.0;
max_sr = 0.0;

plot_legend={};
for line_id=1:n_lines    
    desc=plot_data_desc{line_id};
    actor_id=desc('actor_id');
    actor=actors{actor_id+1};    
    mean_idxs=actor_idxs{actor_id+1}{desc('mean_idx_id')+1};
    if(opt_gt_ssm=='0')
        data_fname=sprintf('%s/success_rates/sr_%s_%s_%s_%s_%d.txt', root_dir, actor,...
            desc('mtf_sm'), desc('mtf_am'), desc('mtf_ssm'), desc('iiw'));
    else
    data_fname=sprintf('%s/success_rates/sr_%s_%s_%s_%s_%d_%s.txt', root_dir, actor,...
        desc('mtf_sm'), desc('mtf_am'), desc('mtf_ssm'), desc('iiw'), opt_gt_ssm);
    end
    fprintf('Reading data for plot line %d from: %s\n', line_id, data_fname);
    data_sr{line_id}=importdata(data_fname);
    err_thr=data_sr{line_id}(:, 1);
    if plot_combined_data
        line_data{line_id}=data_sr{line_id}(:, end);
    else
        line_data{line_id}=mean(data_sr{line_id}(:, mean_idxs+1), 2);
    end
    plot(err_thr, line_data{line_id}, 'Color', col_rgb{strcmp(col_names,desc('color'))},...
    'LineStyle', desc('line_style'), 'LineWidth', line_width);
    if ~isempty(desc('legend'))
        plot_legend=[plot_legend {desc('legend')}];
    end
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

h_legend=legend(plot_legend);
set(h_legend,'FontSize',legend_font_size);
set(h_legend,'FontWeight','bold');

if adaptive_axis_range
    y_min=floor(min_sr*10)/10;
    y_max=ceil(max_sr*10)/10;
    fprintf('min_sr: %f\t max_sr: %f\n', min_sr, max_sr);
    fprintf('y_min: %f\t y_max: %f\n', y_min, y_max);
end

axis([0 20 y_min y_max]);
% axis([0 20 0.4 1]);
set(gca,'YTick', y_min:0.1:y_max);

xlabel('Error Threshold');
ylabel('Success Rate');

