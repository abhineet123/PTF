#!/bin/bash -v
# Generate SSM updates for TMT, UCSB and LinTrack using Homography

SETTINGS="db_root_path /home/abhineet/Secondary/Datasets/ write_tracking_data 1 tracking_err_type 0 reinit_with_new_obj 0 reinit_gt_from_bin 1 reinit_frame_skip 5 reinit_err_thresh 20 use_opt_gt 0 pause_after_frame 0 show_cv_window 0 write_gt_ssm_params 0"
HOM_SETTINGS="hom_corner_based_sampling 1 hom_normalized_init 1"
ACTOR_IDS=(0 1 2)
N_SEQS=(109 96 3)
for ACTOR_ID in "${!ACTOR_IDS[@]}"; do
	SEQ_ID=0
	N_SEQ=${N_SEQS[$ACTOR_ID]}
	echo "ACTOR_ID: $ACTOR_ID"	
	echo "N_SEQ: $N_SEQ"	
	while [ $SEQ_ID -lt ${N_SEQS[$ACTOR_ID]} ]; do
		showGroundTruth actor_id $ACTOR_ID source_id $SEQ_ID tracking_data_fname pf1k_ncc50r30i4u_8_0 $SETTINGS mtf_ssm 8 reinit_from_gt 1 $HOM_SETTINGS 	
		let SEQ_ID=SEQ_ID+1 
	done
done
