function combineVideos
% clear all;
% workingDir='../../../Reports/Thesis/Presentation/tracking_videos';
workingDir='H:\UofA\Acamp\code\object_detection\videos';
% workingDir='../../../../206';

read_from_list = 1;

if read_from_list
    videos=importdata(sprintf('%s/list.txt',workingDir))
else
    videos = {
        'YUN00001_0_239_640_640_640_640',...
        'YUN00001_0_239_640_640_640_640',...
        'YUN00001_0_239_640_640_640_640',...
        'YUN00001_0_239_640_640_640_640',...
        'YUN00001_0_239_640_640_640_640',...
        };
end

width = 640;
height = 360;
n_channels = 3;

fps = 24;
quality = 100;
vid_fmt='mp4';
% profile='Motion JPEG AVI';
profile='MPEG-4';


out_fname=sprintf('grizzly_bear_1_1_4.%s', vid_fmt);
fprintf('Writing to: %s\n', out_fname);

outputVideo = VideoWriter(fullfile(workingDir,out_fname), profile);
outputVideo.FrameRate = fps;
outputVideo.Quality = quality;
open(outputVideo);

n_videos = length(videos);
aspect_ratio = double(width) / double(height);


for video_id = 1:n_videos
    
    vid_name=videos{video_id};
%     vid_path = vid_name;
    vid_path = fullfile(workingDir, vid_name);
    fprintf('Processing file: %s\n',vid_path);
    
    video = VideoReader(vid_path);
    
    frame_id=0;
    while hasFrame(video)
        frame_id=frame_id+1;
        fprintf('%s :: Done processing %d frames\n',vid_name, frame_id);
        img = readFrame(video);
        src_height = int32(size(img, 1));
        src_width = int32(size(img, 2));
        
        src_aspect_ratio = double(src_width) / double(src_height);
        
        if src_aspect_ratio == aspect_ratio
            dst_width = src_width;
            dst_height = src_height;
            start_row = 0;
            start_col = 0;
        elseif src_aspect_ratio > aspect_ratio
            dst_width = src_width;
            dst_height = int32(src_width / aspect_ratio);
            start_row = int32((dst_height - src_height) / 2.0);
            start_col = 0;
        else
            dst_height = src_height;
            dst_width = int32(src_height * aspect_ratio);
            start_col = int32((dst_width - src_width) / 2.0);
            start_row = 0;
        end
        
        start_row = int32(start_row);
        start_col = int32(start_col);
        
        end_row = start_row + src_height;
        end_col = start_col + src_width;
        
        dst_img = uint8(zeros(dst_height, dst_width, n_channels));
        dst_img(start_row+1:end_row, start_col+1:end_col, :) = img;
        
        dst_img=imresize(dst_img, [height width]);
        
        writeVideo(outputVideo,dst_img);
        frame_id=frame_id+1;
    end
end
close(outputVideo);
% clear all;
end
