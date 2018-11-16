clear all;
root_dir='./vid';
% root_dir='../../Datasets';

% getParamLists;
% actor_id = 1;
% seq_id = 46;
% actor = actors{actor_id+1};
% seq_name = sequences{actor_id + 1}{seq_id + 1};

seq_names={
    'single_tracker_book_cropped',...
    'single_tracker_bus_mug_cropped',...
    'single_tracker_non_planar_cropped',...
    'multi_tracker_cropped',...
    };
stack_order = 0;
n_frames = 0;
resize_factor = 0.5;
start_time = 0;
end_time = 0;
desired_fps = 1000;
out_fps = 30;
quality = 100;
input_format='mp4';
output_fmt='mp4';
% profile='Motion JPEG AVI';
profile='MPEG-4';
n_seq=length(seq_names);
inputVideos=cell(n_seq, 1);
out_fname='stacked';
for seq_id = 1:n_seq
    filename=seq_names{seq_id};
    if ~isempty(input_format)
        file_path=sprintf('%s/%s.%s',root_dir,filename,input_format);
    else
        file_path=sprintf('%s/%s',root_dir,filename);
    end
    fprintf('Reading input video %d from: %s\n', seq_id, file_path);
    inputVideos{seq_id} = VideoReader(file_path);
    out_fname=sprintf('%s__%s', out_fname, filename);
end

out_fname=sprintf('%s__%d_%d_%d_%d.%s', out_fname, out_fps, quality,...
    start_time, end_time, output_fmt);
fprintf('Writing output video to: %s\n', out_fname);
outputVideo = VideoWriter(fullfile(root_dir, out_fname), profile);
outputVideo.FrameRate = out_fps;
outputVideo.Quality = quality;
open(outputVideo);

frame_id=0;
input_ended = 0;
while ~input_ended
    frame_id=frame_id+1;
    current_frame_list=cell(n_seq, 1);
    for seq_id = 1:n_seq
        current_frame_list{seq_id} = readFrame(inputVideos{seq_id});
        if resize_factor ~= 1
            current_frame_list{seq_id}=imresize(current_frame_list{seq_id},...
                resize_factor);
        end
        if ~hasFrame(inputVideos{seq_id})
            input_ended = 1;
        end
    end
    stacked_frame = stackImages(current_frame_list, stack_order);
    writeVideo(outputVideo, stacked_frame);
    if end_time>start_time && inputVideo.CurrentTime>end_time
        break;
    end
    if mod(frame_id, 10)==0
        fprintf('\tDone processing %d frames\n',frame_id);
    end
    if n_frames > 0 && frame_id >= n_frames
        break;
    end
end
% for seq_id = 1:n_seq
%     close(inputVideos{seq_id});
% end
close(outputVideo);

