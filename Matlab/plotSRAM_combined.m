clear all;

colRGBDefs;

mtf_ssm = '8';
iiw = 0;
mtf_sms={'nesm', 'fclk', 'nnic'};

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 14);
set(0,'DefaultAxesFontWeight', 'bold');


plot_title=sprintf('AM Success Rates with 8DOF');

figure;
line_width = 3;

for i=1:3    
    mtf_sm=mtf_sms{i};    

    actor = 'Human';
    rscv_sr=importdata(sprintf('sr_%s_%s_rscv_%s_%d.txt', actor, mtf_sm, mtf_ssm, iiw));
    scv_sr=importdata(sprintf('sr_%s_%s_scv_%s_%d.txt', actor, mtf_sm, mtf_ssm, iiw));
    ssd_sr=importdata(sprintf('sr_%s_%s_ssd_%s_%d.txt', actor, mtf_sm, mtf_ssm, iiw));
    ncc_sr=importdata(sprintf('sr_%s_%s_ncc_%s_%d.txt', actor, mtf_sm, mtf_ssm, iiw));
    actor = 'VTD';
    vtd_rscv_sr=importdata(sprintf('sr_%s_%s_rscv_%s_%d.txt', actor, mtf_sm, mtf_ssm, iiw));
    vtd_scv_sr=importdata(sprintf('sr_%s_%s_scv_%s_%d.txt', actor, mtf_sm, mtf_ssm, iiw));
    vtd_ssd_sr=importdata(sprintf('sr_%s_%s_ssd_%s_%d.txt', actor, mtf_sm, mtf_ssm, iiw));
    vtd_ncc_sr=importdata(sprintf('sr_%s_%s_ncc_%s_%d.txt', actor, mtf_sm, mtf_ssm, iiw));

    min_seq_id = 0;
    max_seq_id = 97;
    min_seq_id = min_seq_id+2;
    max_seq_id = max_seq_id+2;
    err_thr=rscv_sr(:, 1);
    rscv_sr_mean=mean(rscv_sr(:, min_seq_id:max_seq_id), 2);
    scv_sr_mean=mean(scv_sr(:, min_seq_id:max_seq_id), 2);
    ssd_sr_mean=mean(ssd_sr(:, min_seq_id:max_seq_id), 2);
    ncc_sr_mean=mean(ncc_sr(:, min_seq_id:max_seq_id), 2);

    min_seq_id = 0;
    max_seq_id = 95;
    min_seq_id = min_seq_id+2;
    max_seq_id = max_seq_id+2;
    vtd_err_thr=vtd_rscv_sr(:, 1);
    vtd_rscv_sr_mean=mean(vtd_rscv_sr(:, min_seq_id:max_seq_id), 2);
    vtd_scv_sr_mean=mean(vtd_scv_sr(:, min_seq_id:max_seq_id), 2);
    vtd_ssd_sr_mean=mean(vtd_ssd_sr(:, min_seq_id:max_seq_id), 2);
    vtd_ncc_sr_mean=mean(vtd_ncc_sr(:, min_seq_id:max_seq_id), 2);
    
    subplot(1, 3, i), hold on, grid on, title(upper(mtf_sm));
    plot(err_thr, rscv_sr_mean, 'Color', col_rgb{strcmp(col_names,'orange')},...
        'LineStyle', '-', 'LineWidth', line_width);
    plot(err_thr, scv_sr_mean, 'Color', col_rgb{strcmp(col_names,'green')},...
        'LineStyle', '-', 'LineWidth', line_width);
    plot(err_thr, ssd_sr_mean, 'Color', col_rgb{strcmp(col_names,'blue')},...
        'LineStyle', '-', 'LineWidth', line_width);
    plot(err_thr, ncc_sr_mean, 'Color', col_rgb{strcmp(col_names,'red')},...
        'LineStyle', '-', 'LineWidth', line_width);
    
    plot(vtd_err_thr, vtd_rscv_sr_mean, 'Color', col_rgb{strcmp(col_names,'orange')},...
        'LineStyle', '--', 'LineWidth', line_width);
    plot(vtd_err_thr, vtd_scv_sr_mean, 'Color', col_rgb{strcmp(col_names,'green')},...
        'LineStyle', '--', 'LineWidth', line_width);
    plot(vtd_err_thr, vtd_ssd_sr_mean, 'Color', col_rgb{strcmp(col_names,'blue')},...
        'LineStyle', '--', 'LineWidth', line_width);
    plot(vtd_err_thr, vtd_ncc_sr_mean, 'Color', col_rgb{strcmp(col_names,'red')},...
        'LineStyle', '--', 'LineWidth', line_width);
    xlabel('Error Threshold');
    ylabel('Success Rate');    
    axis([0 20 0 1]);
end
suplabel(plot_title, 't');
plot_legend={'RSCV', 'SCV', 'SSD', 'NCC'};
h_legend=legend(plot_legend);
set(h_legend,'FontSize',10);
axis([0 20 0 1]); 




