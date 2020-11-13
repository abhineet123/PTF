@echo off
set NLM=^
set NL=^^^%NLM%%NLM%^%NLM%%NLM%
:x
set /p fname="%NL%Enter log folder name:%NL%"
tensorboard --logdir=%fname% --samples_per_plugin images=100
goto x