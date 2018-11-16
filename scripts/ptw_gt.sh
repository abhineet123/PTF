#!/bin/bash -v
set -x
DB_ROOT_PATH="/home/abhineet/Secondary/Datasets/"
SETTINGS="mtf_sm fc mtf_ssm 8 mtf_res 50 mtf_ilm 0 res_from_size 0 enable_nt 0 max_iters 30 epsilon 1e-4 db_root_path $DB_ROOT_PATH write_tracking_data 1 pre_proc_type 1  overwrite_gt 0 show_tracking_error 1 tracking_err_type 0 reinit_with_new_obj 0 reinit_at_each_frame 0 reinit_gt_from_bin 1 reinit_frame_skip 5 reinit_err_thresh 20 use_opt_gt 0 pause_after_frame 0 show_cv_window 0 init_frame_id 0 start_frame_id 0 frame_gap 1 read_obj_from_gt 1 invalid_state_check 1 invalid_state_err_thresh 0 img_resize_factor 1"
FC_SETTINGS="sec_ord_hess 0 fc_chained_warp 1 fc_hess_type 1 leven_marq 0 lm_delta_init 0.01 lm_delta_update 10 fc_write_ssm_updates 0 fc_show_grid 0 fc_show_patch 0 fc_patch_resize_factor 4 enable_learning 0 fc_debug_mode 0"
SCV_SETTINGS="scv_use_bspl 0 scv_n_bins 256 scv_preseed 0 scv_pou 1 scv_weighted_mapping 0 scv_mapped_gradient 0 scv_affine_mapping 1 scv_once_per_frame 0 scv_approx_dist_feat 0"
LSCV_SETTINGS="lscv_sub_regions 3 lscv_spacing 10 lscv_show_subregions 0"
HOM_SETTINGS="hom_corner_based_sampling 1 hom_normalized_init 1"
ACTORS=(TMT UCSB LinTrack PAMI LinTrackShort METAIO CMT VOT VOT16 VTB VIVID TrakMark TMT_FINE)

N_SEQS=(109 96 3 28 24 210)
ACTOR_IDS=(5)
EXEC_NAME=runMTF
TRACKING_DATA_FNAME=rkl_ncc_8_0
N_ACTORS=${#ACTOR_IDS[@]}
ACTOR_IDS_IDX=0
while [ $ACTOR_IDS_IDX -lt ${N_ACTORS} ]; do	
	ACTOR_ID=${ACTOR_IDS[$ACTOR_IDS_IDX]}
	N_SEQ=${N_SEQS[$ACTOR_ID]}
	ACTOR=${ACTORS[$ACTOR_ID]}
	SEQ_ID=49
	saveIFS=$IFS
	while [ $SEQ_ID -lt ${N_SEQ} ]; do		
		$EXEC_NAME actor_id $ACTOR_ID source_id $SEQ_ID		
		let SEQ_ID=SEQ_ID+1 		
	done
	let ACTOR_IDS_IDX=ACTOR_IDS_IDX+1
done
