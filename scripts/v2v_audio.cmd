@echo off
ffmpeg  -y -i %1 -i %2 -c copy %3
