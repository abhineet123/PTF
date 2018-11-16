clear all;

colRGBDefs;

mtf_sm = 'nesm';
mtf_am = 'rscv';
iiw = 0;

plot_title=sprintf('SSM Success Rates for NESM (corresponding GT)');

actor = 'TMT';
lhom_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_l8_%d.txt', actor, mtf_sm, mtf_am, iiw));
chom_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_c8_%d.txt', actor, mtf_sm, mtf_am, iiw));
hom_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_8_%d.txt', actor, mtf_sm, mtf_am, iiw));
aff_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_6_%d_6.txt', actor, mtf_sm, mtf_am, iiw));
sim_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_4_%d_4.txt', actor, mtf_sm, mtf_am, iiw));
iso_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_3_%d_3.txt', actor, mtf_sm, mtf_am, iiw));
trans_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_2_%d_2.txt', actor, mtf_sm, mtf_am, iiw));
actor = 'VTD';
vtd_lhom_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_l8_%d.txt', actor, mtf_sm, mtf_am, iiw));
vtd_chom_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_c8_%d.txt', actor, mtf_sm, mtf_am, iiw));
vtd_hom_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_8_%d.txt', actor, mtf_sm, mtf_am, iiw));
vtd_aff_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_6_%d_6.txt', actor, mtf_sm, mtf_am, iiw));
vtd_sim_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_4_%d_4.txt', actor, mtf_sm, mtf_am, iiw));
vtd_iso_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_3_%d_3.txt', actor, mtf_sm, mtf_am, iiw));
vtd_trans_sr=importdata(sprintf('success_rates/sr_%s_%s_%s_2_%d_2.txt', actor, mtf_sm, mtf_am, iiw));

min_seq_id = 0;
max_seq_id = 97;
min_seq_id = min_seq_id+2;
max_seq_id = max_seq_id+2;
err_thr=hom_sr(:, 1);
lhom_sr_mean=mean(lhom_sr(:, min_seq_id:max_seq_id), 2);
chom_sr_mean=mean(chom_sr(:, min_seq_id:max_seq_id), 2);
hom_sr_mean=mean(hom_sr(:, min_seq_id:max_seq_id), 2);
aff_sr_mean=mean(aff_sr(:, min_seq_id:max_seq_id), 2);
sim_sr_mean=mean(sim_sr(:, min_seq_id:max_seq_id), 2);
iso_sr_mean=mean(iso_sr(:, min_seq_id:max_seq_id), 2);
trans_sr_mean=mean(trans_sr(:, min_seq_id:max_seq_id), 2);

min_seq_id = 0;
max_seq_id = 95;
min_seq_id = min_seq_id+2;
max_seq_id = max_seq_id+2;
vtd_err_thr=vtd_hom_sr(:, 1);
vtd_lhom_sr_mean=mean(vtd_lhom_sr(:, min_seq_id:max_seq_id), 2);
vtd_chom_sr_mean=mean(vtd_chom_sr(:, min_seq_id:max_seq_id), 2);
vtd_hom_sr_mean=mean(vtd_hom_sr(:, min_seq_id:max_seq_id), 2);
vtd_aff_sr_mean=mean(vtd_aff_sr(:, min_seq_id:max_seq_id), 2);
vtd_sim_sr_mean=mean(vtd_sim_sr(:, min_seq_id:max_seq_id), 2);
vtd_iso_sr_mean=mean(vtd_iso_sr(:, min_seq_id:max_seq_id), 2);
vtd_trans_sr_mean=mean(vtd_trans_sr(:, min_seq_id:max_seq_id), 2);


set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 14);
set(0,'DefaultAxesFontWeight', 'bold');

h_fig=figure;

line_width = 3;

% plot(err_thr, rscv_sr_mean, 'Color', [1 0 0], 'LineStyle', '-', 'LineWidth', line_width);
% plot(err_thr, scv_sr_mean, '-g','LineWidth', line_width);
% plot(err_thr, ssd_sr_mean, '-b','LineWidth', line_width);
% plot(err_thr, ncc_sr_mean, '-k','LineWidth', line_width);

subplot(1, 2, 1), hold on, grid on;
plot(err_thr, lhom_sr_mean, 'Color', col_rgb{strcmp(col_names,'magenta')},...
    'LineStyle', '--', 'LineWidth', line_width);
plot(err_thr, chom_sr_mean, 'Color', col_rgb{strcmp(col_names,'cyan')},...
    'LineStyle', ':', 'LineWidth', line_width);
plot(err_thr, hom_sr_mean, 'Color', col_rgb{strcmp(col_names,'orange')},...
    'LineStyle', '-', 'LineWidth', line_width);
plot(err_thr, aff_sr_mean, 'Color', col_rgb{strcmp(col_names,'black')},...
    'LineStyle', '-', 'LineWidth', line_width);
plot(err_thr, sim_sr_mean, 'Color', col_rgb{strcmp(col_names,'green')},...
    'LineStyle', '-', 'LineWidth', line_width);
plot(err_thr, iso_sr_mean, 'Color', col_rgb{strcmp(col_names,'blue')},...
    'LineStyle', '-', 'LineWidth', line_width);
plot(err_thr, trans_sr_mean, 'Color', col_rgb{strcmp(col_names,'red')},...
    'LineStyle', '-', 'LineWidth', line_width);
title('TMT Dataset');
xlabel('Error Threshold');
ylabel('Success Rate');
axis([0 20 0 1]);

subplot(1, 2, 2), hold on, grid on;
plot(vtd_err_thr, vtd_lhom_sr_mean, 'Color', col_rgb{strcmp(col_names,'magenta')},...
    'LineStyle', '-', 'LineWidth', line_width);
plot(vtd_err_thr, vtd_chom_sr_mean, 'Color', col_rgb{strcmp(col_names,'cyan')},...
    'LineStyle', '-', 'LineWidth', line_width);
plot(vtd_err_thr, vtd_hom_sr_mean, 'Color', col_rgb{strcmp(col_names,'orange')},...
    'LineStyle', '-', 'LineWidth', line_width);
plot(vtd_err_thr, vtd_aff_sr_mean, 'Color', col_rgb{strcmp(col_names,'black')},...
    'LineStyle', '-', 'LineWidth', line_width);
plot(vtd_err_thr, vtd_sim_sr_mean, 'Color', col_rgb{strcmp(col_names,'green')},...
    'LineStyle', '-', 'LineWidth', line_width);
plot(vtd_err_thr, vtd_iso_sr_mean, 'Color', col_rgb{strcmp(col_names,'blue')},...
    'LineStyle', '-', 'LineWidth', line_width);
plot(vtd_err_thr, vtd_trans_sr_mean, 'Color', col_rgb{strcmp(col_names,'red')},...
    'LineStyle', '-', 'LineWidth', line_width);
title('UCSB Dataset');
xlabel('Error Threshold');
ylabel('Success Rate');
axis([0 20 0 1]);

% legend('TMT-rscv', 'TMT-scv', 'TMT-ssd', 'TMT-ncc',...
%     'UCSB-rscv', 'UCSB-scv', 'UCSB-ssd', 'UCSB-ncc');
% mtit(gca, plot_title);
suplabel(plot_title, 't');
plot_legend={'Lie 8DOF (SL3)', 'Corner 8DOF','8DOF', '6DOF', '4DOF', '3DOF', '2DOF'};
h_legend=legend(plot_legend);
set(h_legend,'FontSize',10);
axis([0 20 0 1]);



