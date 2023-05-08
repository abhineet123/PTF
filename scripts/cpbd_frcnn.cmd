@echo off
REM crop patch by detection

set curr_dir=%cd%
set tf_api_dir=%~dp0\..\..\..\..\Acamp\acamp_code\tf_api
echo %curr_dir%
echo %tf_api_dir%
echo %PYTHONPATH%
echo %1
echo %2

cd /D %tf_api_dir% & python3 tf_api_test.py ckpt_path=pre_trained_models/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb labels_path=data/mscoco_label_map.pbtxt root_dir=%curr_dir% n_frames=0 batch_size=1 show_img=1 save_video=0 vis_width=1280 vis_height=960 n_classes=90 write_det=0 classes_to_include=person, n_objs_to_include=1 save_patches=1 extend_vertically=0  patch_out_root_dir=%curr_dir%/#patches_src root_dir=%curr_dir%/%1 n_objs_to_include=%2 extend_vertically=%3 patch_ar=%4 gpu=0

cd /D %curr_dir%
REM cd /D H:\UofA\Acamp\code\tf_api
REM set PYTHONPATH=%PYTHONPATH%;H:\UofA\Acamp\code\tf_api & 
REM patch_ar=0.89
REM extend_vertically=1