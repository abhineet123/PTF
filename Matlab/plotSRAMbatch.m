clear all;

getParamLists;
colRGBDefs;

root_dir='../C++/MTF_LIB/log';

mtf_sm = 'nesm';
mtf_ssm = '8';
iiw = 0;

actor_id = 0;
mean_idx_ids = -1;
n_ams = 0;


mtf_ams={
    'miMJN50r30i8b',...
    'miMJHN50r30i8b',...
    'miCN50r30i8b',...
    'miIN50r30i8b'
    };
% mtf_ams={   
%     'ssdMJN50r30i8b',...
%     'ssdMJHN50r30i8b',...
%     'ssdCN50r30i8b',...
%     'ssdIN50r30i8b' 
%     };

if mean_idx_ids<0
    mean_idx_ids=0:length(actor_idxs{actor_id+1})-1
end

if n_ams<=0 || n_ams>length(mtf_ams)
    n_ams=length(mtf_ams);
end
actor=actors{actor_id+1};
for mean_idx_id=mean_idx_ids      
    mean_idxs=actor_idxs{actor_id+1}{mean_idx_id+1};
    mean_idx_type=actor_idx_types{actor_id+1}{mean_idx_id+1};
    plotSRAMfunc(mtf_ams, mtf_sm, mtf_ssm, iiw,...
        root_dir, n_ams, actor, mean_idxs, mean_idx_type,...
        col_rgb, col_names);
end


