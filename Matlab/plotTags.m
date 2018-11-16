clear all;

actor = 'Human';
mtf_am = 'rscv';
mtf_ssm = '8';
iiw = 0;

plot_type = 2;
rel_plot = 1;

nl_tag_count=[16515, 8145, 4679, 7195, 2927, 534, 2589, 5587]';
dl_tag_count=[16890, 7167, 5442, 2310, 2578, 680, 1645, 5713]';
total_tag_count = nl_tag_count + dl_tag_count;

if plot_type==0  
    min_seq_id = 0;
    max_seq_id = 49;
    plot_title='Tag Count NL';    
elseif plot_type==1
    min_seq_id = 50;
    max_seq_id = 97;
    plot_title='Tag Count DL';
elseif plot_type==2
    min_seq_id = 0;
    max_seq_id = 97;
    plot_title='SM Success Rates for Different Challeges with RSCV/8DOF ';
else
    error('Invalid plot type: %d', plot_type);
end
    


fclk_tags=importdata(sprintf('tags_fclk_%s_%s_%d.txt', mtf_am, mtf_ssm, iiw));
iclk_tags=importdata(sprintf('tags_iclk_%s_%s_%d.txt',  mtf_am, mtf_ssm, iiw));
falk_tags=importdata(sprintf('tags_falk_%s_%s_%d.txt', mtf_am, mtf_ssm, 1));
ialk_tags=importdata(sprintf('tags_ialk_%s_%s_%d.txt',  mtf_am, mtf_ssm, 1));
%esm_tags=importdata(sprintf('tags_esm_%s_%s_%d.txt',  mtf_am, mtf_ssm, iiw));
nesm_tags=importdata(sprintf('tags_nesm_%s_%s_%d.txt',  mtf_am, mtf_ssm, iiw));
aesm_tags=importdata(sprintf('tags_aesm_%s_%s_%d.txt',  mtf_am, mtf_ssm, 1));
nnic_tags=importdata(sprintf('tags_nnic_%s_%s_%d.txt',  mtf_am, mtf_ssm, iiw));

min_seq_id = min_seq_id+1;
max_seq_id = max_seq_id+1;

n_sm = 7;
n_tags = 8;
tag_ids=1:n_tags;

fclk_tags_sum=sum(fclk_tags(:, min_seq_id:max_seq_id), 2);
iclk_tags_sum=sum(iclk_tags(:, min_seq_id:max_seq_id), 2);
falk_tags_sum=sum(falk_tags(:, min_seq_id:max_seq_id), 2);
ialk_tags_sum=sum(ialk_tags(:, min_seq_id:max_seq_id), 2);
nesm_tags_sum=sum(nesm_tags(:, min_seq_id:max_seq_id), 2);
aesm_tags_sum=sum(aesm_tags(:, min_seq_id:max_seq_id), 2);
nnic_tags_sum=sum(nnic_tags(:, min_seq_id:max_seq_id), 2);
%esm_tags_sum=sum(esm_tags(:, min_seq_id:max_seq_id), 2);

combined_tags_sum=zeros(n_tags, n_sm);
combined_tags_sum(:, 1)=fclk_tags_sum;
combined_tags_sum(:, 2)=iclk_tags_sum;
combined_tags_sum(:, 3)=falk_tags_sum;
combined_tags_sum(:, 4)=ialk_tags_sum;
combined_tags_sum(:, 5)=nesm_tags_sum;
combined_tags_sum(:, 6)=aesm_tags_sum;
combined_tags_sum(:, 7)=nnic_tags_sum;
%combined_tags_sum(:, 7)=esm_tags_sum;

if rel_plot
    if plot_type==0  
        combined_tags_sum = combined_tags_sum ./ repmat(nl_tag_count, 1, size(combined_tags_sum,2));   
    elseif plot_type==1
        combined_tags_sum = combined_tags_sum ./ repmat(dl_tag_count, 1, size(combined_tags_sum,2));
    elseif plot_type==2
        combined_tags_sum = combined_tags_sum ./ repmat(total_tag_count, 1, size(combined_tags_sum,2));
    else
        error('Invalid plot type: %d', plot_type);
    end
end
set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 14);
set(0,'DefaultAxesFontWeight', 'bold');

figure, hold on, grid on, title(plot_title);
bar(tag_ids, combined_tags_sum);
legend('FCLK', 'ICLK', 'FALK', 'IALK', 'ESM', 'AESM', 'NNIC');
ax = gca;
set(ax,'XTickLabel',{'0', 'TR','RO','PR','SR','OC','TX','BL', 'SC'})
xlabel('challenges');
ylabel('success count');

