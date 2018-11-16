clear all;

getParamLists;
colRGBDefs;

root_dir='../C++/MTF_LIB/log';

mtf_sm = 'nesm';
mtf_ssm = '8';
iiw = 0;

actor_id = 1;
mean_idx_id = 0;
n_ams = 0;

mtf_ams={
    'ssd',...
    'rscv',...
    'ncc',...
    'ssimCN50r30i4u',...  
    'rssimCN50r30i4u',...
    'rssimMJN50r30i4u',...  
    'rssimMJHN50r30i4u'     
    };

if n_ams<=0 || n_ams>length(mtf_ams)
    n_ams=length(mtf_ams);
end

actor=actors{actor_id+1};
mean_idxs=actor_idxs{actor_id+1}{mean_idx_id+1};
mean_idx_type=actor_idx_types{actor_id+1}{mean_idx_id+1};

plot_legend=mtf_ams;

plot_cols={
    'orange',...
    'green',...
    'blue',...
    'red',...
    'black',...
    'cyan',...
    'chartreuse',...
    'olive_drab',...
    'bisque',...
    'moccasin',...
    'seashell',...
    'lemon_chiffon'
    };

am_sr=cell(n_ams, 1);
am_sr_mean=cell(n_ams, 1);

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 18);
set(0,'DefaultAxesFontWeight', 'bold');

for am_id=1:n_ams
    mtf_am=mtf_ams{am_id}
    data_fname=sprintf('%s/success_rates/sr_%s_%s_%s_%s_%d.txt', root_dir, actor, mtf_sm, mtf_am, mtf_ssm, iiw)
    am_sr{am_id}=importdata(data_fname);
    am_sr_mean{am_id}=mean(am_sr{am_id}(:, mean_idxs+1), 2);
end

err_thr=am_sr{1}(:, 1);



plot_title=sprintf('AM Success Rates for %s on %s (%s)', upper(mtf_sm), actor, mean_idx_type);

plot_fig=figure;
hold on, grid on, title(plot_title, 'interpreter', 'none');
line_width = 3;
set(plot_fig, 'Name', plot_title);
set (plot_fig, 'Units', 'normalized', 'Position', [0,0,1,1]);

for am_id=1:n_ams
    plot(err_thr, am_sr_mean{am_id}, 'Color', col_rgb{strcmp(col_names,plot_cols{am_id})},...
        'LineStyle', '-', 'LineWidth', line_width);
end

h_legend=legend(plot_legend);
set(h_legend,'FontSize',16);
set(h_legend,'FontWeight','bold');

axis([0 20 0.15 0.8]);
% axis([0 20 0.4 1]);

xlabel('Error Threshold');
ylabel('Success Rate');

