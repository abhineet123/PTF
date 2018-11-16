clear all;
root_dir='.';
% root_dir='../../Datasets';

% getParamLists;
% actor_id = 1;
% seq_id = 46;
% actor = actors{actor_id+1};
% seq_name = sequences{actor_id + 1}{seq_id + 1};

seq_name='comparing_ssm_tmt_stacked_grid_25x25_24_100';

start_time = 0;
end_time = 0;
desired_fps = 1000;
out_fps = 24;
quality = 100;
input_format='avi';
output_fmt='mp4';
% profile='Motion JPEG AVI';
profile='MPEG-4';

filename=seq_name;   
if ~isempty(input_format)
    file_path=sprintf('%s/%s.%s',root_dir,filename,input_format)
else
    file_path=sprintf('%s/%s',root_dir,filename)
end
inputVideo = VideoReader(file_path);
fprintf('Reading input video from: %s\n',file_path);

inputVideo.CurrentTime = start_time;

out_fname=sprintf('%s_%d_%d_%d_%d.%s', seq_name, out_fps, quality,...
    start_time, end_time, output_fmt);
fprintf('Writing output video to: %s\n', out_fname);
outputVideo = VideoWriter(fullfile(root_dir, out_fname), profile);
outputVideo.FrameRate = out_fps;
outputVideo.Quality = quality;
open(outputVideo);

fps_ratio = out_fps/inputVideo.FrameRate;

frame_id=0;    
while hasFrame(inputVideo)
    frame_id=frame_id+1;
    img = readFrame(inputVideo);
    writeVideo(outputVideo,img);  
    if end_time>start_time && inputVideo.CurrentTime>end_time
        break;
    end
    if mod(frame_id, 10)==0
        fprintf('\tDone processing %d frames\n',frame_id);
    end   
end
close(outputVideo);
