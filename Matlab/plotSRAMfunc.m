function plotSRAMfunc(mtf_ams, mtf_sm, mtf_ssm, iiw,...
    root_dir, n_ams, actor, mean_idxs, mean_idx_type,...
    col_rgb, col_names)

plot_legend=upper(mtf_ams)

plot_cols={
    'orange',...
    'green',...
    'blue',...
    'red',...
    'black',...
    'cyan',...
    'chartreuse',...
    'olive_drab'
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

axis([0 20 0 1]);

xlabel('Error Threshold');
ylabel('Success Rate');

end

