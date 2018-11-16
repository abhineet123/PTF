clear all;
colRGBDefs;
actors={
    'Human',...
    'VTD'
    };
actor_names={
    'TMT',...
    'UCSB'
    };


mtf_am = 'rscv';
mtf_ssm = '2';
opt_gt_ssm = '2';
iiw = 0;
plot_type = 2;
plot_cmbd = 0;
use_opt_gt = 0;
plot_learning = 1;
plot_pf = 1;


if ~strcmp(opt_gt_ssm, '0')
    use_opt_gt=1;
    fprintf('Using optimized ground truth with ssm: %s\n', opt_gt_ssm);
end

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 14);
set(0,'DefaultAxesFontWeight', 'bold');

figure;
line_width=3;

for actor_id = 1:2
    actor = actors{actor_id}; 
   
    if use_opt_gt
        fclk_sr=importdata(sprintf('sr_%s_fclk_%s_%s_%d_%s.txt', actor, mtf_am, mtf_ssm, iiw, opt_gt_ssm));
        iclk_sr=importdata(sprintf('sr_%s_iclk_%s_%s_%d_%s.txt', actor, mtf_am, mtf_ssm, iiw, opt_gt_ssm));
        falk_sr=importdata(sprintf('sr_%s_falk_%s_%s_%d_%s.txt', actor, mtf_am, mtf_ssm, 1, opt_gt_ssm));
        ialk_sr=importdata(sprintf('sr_%s_ialk_%s_%s_%d_%s.txt', actor, mtf_am, mtf_ssm, 1, opt_gt_ssm));
        nesm_sr=importdata(sprintf('sr_%s_nesm_%s_%s_%d_%s.txt', actor, mtf_am, mtf_ssm, iiw, opt_gt_ssm));
        aesm_sr=importdata(sprintf('sr_%s_aesm_%s_%s_%d_%s.txt', actor, mtf_am, mtf_ssm, 1, opt_gt_ssm));
        nnic_sr=importdata(sprintf('sr_%s_nnic_%s_%s_%d_%s.txt', actor, mtf_am, mtf_ssm, iiw, opt_gt_ssm));
        
        if plot_learning
            tld_sr=importdata(sprintf('sr_%s_tld_rscv_8_1_%s.txt', actor, opt_gt_ssm));
            kcf_sr=importdata(sprintf('sr_%s_kcf_rscv_8_1_%s.txt', actor, opt_gt_ssm));
            dsst_sr=importdata(sprintf('sr_%s_dsst_rscv_8_1_%s.txt', actor, opt_gt_ssm));
            pf_sr=importdata(sprintf('sr_%s_pf_rscv_2_1_%s.txt', actor, opt_gt_ssm));
        end
    else
        fclk_sr=importdata(sprintf('sr_%s_fclk_%s_%s_%d.txt', actor, mtf_am, mtf_ssm, iiw));
        iclk_sr=importdata(sprintf('sr_%s_iclk_%s_%s_%d.txt', actor, mtf_am, mtf_ssm, iiw));
        falk_sr=importdata(sprintf('sr_%s_falk_%s_%s_%d.txt', actor, mtf_am, mtf_ssm, 1));
        ialk_sr=importdata(sprintf('sr_%s_ialk_%s_%s_%d.txt', actor, mtf_am, mtf_ssm, 1));
        nesm_sr=importdata(sprintf('sr_%s_nesm_%s_%s_%d.txt', actor, mtf_am, mtf_ssm, iiw));
        aesm_sr=importdata(sprintf('sr_%s_aesm_%s_%s_%d.txt', actor, mtf_am, mtf_ssm, 1));
        nnic_sr=importdata(sprintf('sr_%s_nnic_%s_%s_%d.txt', actor, mtf_am, mtf_ssm, iiw));
        if plot_learning
            tld_sr=importdata(sprintf('sr_%s_tld_rscv_8_1.txt', actor));
            kcf_sr=importdata(sprintf('sr_%s_kcf_rscv_8_1.txt', actor));
            dsst_sr=importdata(sprintf('sr_%s_dsst_rscv_8_1.txt', actor));
            if plot_pf
                pf_sr=importdata(sprintf('sr_%s_pf_rscv_2_1.txt', actor));
            end
        end
    end
    
    n_seq = size(fclk_sr, 2)-2;
    fprintf('actor_id: %d actor: %s n_seq: %d\n', actor_id, actor, n_seq);
    
    min_seq_id = 2;
    max_seq_id = n_seq+1;
    
    err_thr=fclk_sr(:, 1);
    fclk_sr_mean=mean(fclk_sr(:, min_seq_id:max_seq_id), 2);
    iclk_sr_mean=mean(iclk_sr(:, min_seq_id:max_seq_id), 2);
    falk_sr_mean=mean(falk_sr(:, min_seq_id:max_seq_id), 2);
    ialk_sr_mean=mean(ialk_sr(:, min_seq_id:max_seq_id), 2);
    nesm_sr_mean=mean(nesm_sr(:, min_seq_id:max_seq_id), 2);
    aesm_sr_mean=mean(aesm_sr(:, min_seq_id:max_seq_id), 2);
    nnic_sr_mean=mean(nnic_sr(:, min_seq_id:max_seq_id), 2);
    if plot_learning
        tld_sr_mean=mean(tld_sr(:, min_seq_id:max_seq_id), 2);
        kcf_sr_mean=mean(kcf_sr(:, min_seq_id:max_seq_id), 2);
        dsst_sr_mean=mean(dsst_sr(:, min_seq_id:max_seq_id), 2);
        if plot_pf
            pf_sr_mean=mean(pf_sr(:, min_seq_id:max_seq_id), 2);
        end
    end
    
    subplot(1, 3, actor_id), hold on, grid on, title(upper(actor_names{actor_id}));
    plot(err_thr, fclk_sr_mean, 'Color', col_rgb{strcmp(col_names,'orange')},...
        'LineStyle', '-', 'LineWidth', line_width);
    plot(err_thr, iclk_sr_mean, 'Color', col_rgb{strcmp(col_names,'green')},...
        'LineStyle', '-', 'LineWidth', line_width);
    plot(err_thr, falk_sr_mean, 'Color', col_rgb{strcmp(col_names,'blue')},...
        'LineStyle', '-', 'LineWidth', line_width);
    plot(err_thr, ialk_sr_mean, 'Color', col_rgb{strcmp(col_names,'red')},...
        'LineStyle', '-', 'LineWidth', line_width);
    plot(err_thr, nesm_sr_mean, 'Color', col_rgb{strcmp(col_names,'magenta')},...
        'LineStyle', '-', 'LineWidth', line_width);
    plot(err_thr, aesm_sr_mean, 'Color', col_rgb{strcmp(col_names,'purple')},...
        'LineStyle', '-', 'LineWidth', line_width);
    plot(err_thr, nnic_sr_mean, 'Color', col_rgb{strcmp(col_names,'dark_green')},...
        'LineStyle', '-', 'LineWidth', line_width);
    
    if plot_learning
        if plot_pf
            plot(err_thr, pf_sr_mean, 'Color', col_rgb{strcmp(col_names,'cyan')},...
                'LineStyle', '-', 'LineWidth', line_width);
        end
        plot(err_thr, tld_sr_mean, 'Color', col_rgb{strcmp(col_names,'sandy_brown')},...
            'LineStyle', '--', 'LineWidth', line_width);
        plot(err_thr, kcf_sr_mean, 'Color', col_rgb{strcmp(col_names,'olive_drab')},...
            'LineStyle', '--', 'LineWidth', line_width);
        plot(err_thr, dsst_sr_mean, 'Color', col_rgb{strcmp(col_names,'black')},...
            'LineStyle', '--', 'LineWidth', line_width);
    end
       
    xlabel('Error Threshold');
    ylabel('Success Rate');
    axis([0 20 0 1]);
end
plot_legend={'FCLK', 'ICLK', 'FALK', 'IALK', 'ESM', 'AESM', 'NNIC'};
if plot_learning
    if plot_pf
        plot_legend=[plot_legend, {'PF'}];
    end
    plot_legend=[plot_legend, {'TLD', 'KCF', 'DSST'}];
end
plot_title=sprintf('SM Success Rates for RSCV/8DOF');
suplabel(plot_title, 't');
h_legend=legend(plot_legend);
set(h_legend,'FontSize',10);
% speed_img=imread('speed_2dof.png');
% subplot(1, 3, 3), image(speed_img), set(gca,'xtick',[],'ytick',[]);


