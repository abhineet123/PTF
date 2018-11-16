clear all;
% close all;

getParamLists;
colRGBDefs;

root_dir='../C++/MTF/log';
y_min = 0;
y_max = 1;
mean_idx_ids = -1;

plot_titles={};
plot_data_descs={};

desc_keys={'actor_id', 'mean_idx_id', 'mtf_sm', 'mtf_am', 'mtf_ssm', 'opt_gt_ssm', 'iiw',...
    'legend', 'color', 'line_style'};

genericConfigsCRVSM2;
genericConfigsCRVAM2;
genericConfigsCRVSSM;

plot_combined_data = 0;

plot_id = 185;

plot_data_desc=plot_data_descs{plot_id};
plot_title=plot_titles{plot_id};


n_lines=length(plot_data_desc);
if n_lines==0
    error('Invalid plot id provided: %d\n', plot_id);
end

actor_id=plot_data_desc{1}('actor_id');
actor_idx = actor_idxs{actor_id+1};
actor_idx_type=actor_idx_types{actor_id+1}
if mean_idx_ids<0
    mean_idx_ids=0:length(actor_idx)-1
end

for mean_idx_id=mean_idx_ids     
    mean_idxs=actor_idx{mean_idx_id+1};
    mean_idx_type=actor_idx_type{mean_idx_id+1};
    plot_title_final=sprintf('%s (%s)', plot_title, mean_idx_type);
    for line_id=1:n_lines 
        plot_data_desc{line_id}('actor_id') = actor_id;
        plot_data_desc{line_id}('mean_idx_id') = mean_idx_id;
    end
    plotSRGenericFunc(plot_title_final, plot_data_desc, actors, actor_idxs,...
    col_rgb, col_names, y_min, y_max, root_dir, plot_combined_data);
end


