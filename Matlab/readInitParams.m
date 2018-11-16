clear all;

init_params=importdata('../Tools/paramPairs.txt');
param_names=init_params.textdata;
param_vals=init_params.data;
am_name = param_vals(strcmp(param_names,'mtf_am'))
ssm_name = param_vals(strcmp(param_names,'mtf_ssm'))
update_type = param_vals(strcmp(param_names,'diag_update'))

% inc_id = param_vals(strcmp(param_names,'inc_id'));
% appearance_id = param_vals(strcmp(param_names,'appearance_id'));
% tracker_id = param_vals(strcmp(param_names,'tracker_id'));
% start_id = param_vals(strcmp(param_names,'start_id')) ;
% filter_id = param_vals(strcmp(param_names,'filter_id')) ;
% kernel_size = param_vals(strcmp(param_names,'kernel_size')) ;
% end_id = param_vals(strcmp(param_names,'end_id')) ;
% pw_opt_id = param_vals(strcmp(param_names,'opt_id')) ;
% pw_sel_opt = param_vals(strcmp(param_names,'selective_opt')) ;
% pw_opt_dof = param_vals(strcmp(param_names,'dof')) ;
% show_img = param_vals(strcmp(param_names,'show_img'));