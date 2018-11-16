function toVideo
% clear all;
% workingDir='../../../Reports/Thesis/Presentation/tracking_videos';
workingDir='J:/deeplab';
% workingDir='../../../../206';

getParamLists;

% actor_id = 1;
% seq_id = 46;
% actor = actors{actor_id+1};
% seq_name = sequences{actor_id + 1}{seq_id + 1};

% nl_cereal_s3
% nl_bookI_s3
% nl_bookII_s3
% nl_bookIII_s3
% nl_bus
seq_name='YUN00001_0_239_640_640_640_640';

start_frame_id = 1;
end_frame_id = 0;

src_dir=seq_name;

fps = 24;
quality = 100;
img_fmt='jpg';
vid_fmt='mp4';
% profile='Motion JPEG AVI';
profile='MPEG-4';


imageNames = dir(fullfile(workingDir, src_dir,sprintf('*.%s', img_fmt)));
imageNames = {imageNames.name}';
if end_frame_id<=0 || end_frame_id>length(imageNames)
    end_frame_id=length(imageNames);
end
if end_frame_id < start_frame_id
    fprintf('No input frames found\n');
    return
end
out_fname=sprintf('%s_%d_%d_%d_%d.%s', src_dir, fps, quality,...
    start_frame_id, end_frame_id, vid_fmt);
fprintf('Writing to: %s\n', out_fname);
outputVideo = VideoWriter(fullfile(workingDir,out_fname), profile);
outputVideo.FrameRate = fps;
outputVideo.Quality = quality;
open(outputVideo);

frame_id=start_frame_id;
img = imread(fullfile(workingDir, src_dir,imageNames{frame_id}));
writeVideo(outputVideo,img);
img_size=size(img);

while frame_id <= end_frame_id
    fprintf('Frame: %d/%d\n', frame_id, end_frame_id);
    img = imread(fullfile(workingDir, src_dir,imageNames{frame_id}));
    if size(img, 1) ~= img_size(1) || size(img, 2) ~= img_size(2)
        img=imresize(img, [img_size(1) img_size(2)]);
    end
    %     size(img)
    writeVideo(outputVideo,img);
    frame_id=frame_id+1;
end
close(outputVideo);
% clear all;
end
