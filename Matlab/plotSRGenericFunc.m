function plotSRGenericFunc(plot_title, plot_data_desc, actors, actor_idxs,...
    col_rgb, col_names, y_min, y_max, root_dir, plot_combined_data)	
n_lines=length(plot_data_desc);

data_sr=cell(n_lines, 1);
line_data=cell(n_lines, 1);

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 18);
set(0,'DefaultAxesFontWeight', 'bold');


plot_fig=figure;
hold on, grid on, title(plot_title, 'interpreter', 'none');
line_width = 3;
set(plot_fig, 'Name', plot_title);
set (plot_fig, 'Units', 'normalized', 'Position', [0,0,1,1]);

plot_legend={};
for line_id=1:n_lines    
    desc=plot_data_desc{line_id};
    actor_id=desc('actor_id');
    actor=actors{actor_id+1}; 
	actor_idx = actor_idxs{actor_id+1};	
    mean_idxs=actor_idx{desc('mean_idx_id')+1};
    data_fname=sprintf('%s/success_rates/sr_%s_%s_%s_%s_%d.txt', root_dir, actor,...
        desc('mtf_sm'), desc('mtf_am'), desc('mtf_ssm'), desc('iiw'));
    fprintf('Reading data for plot line %d from: %s with mean_idx_id: %d\n',...
        line_id, data_fname, desc('mean_idx_id'));
    data_sr{line_id}=importdata(data_fname);
    err_thr=data_sr{line_id}(:, 1);
    line_data{line_id}=mean(data_sr{line_id}(:, mean_idxs+1), 2);
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
end

h_legend=legend(plot_legend);
set(h_legend,'FontSize',16);
set(h_legend,'FontWeight','bold');

axis([0 20 y_min y_max]);
% axis([0 20 0.4 1]);

xlabel('Error Threshold');
ylabel('Success Rate');

end

