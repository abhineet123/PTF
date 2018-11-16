root_dir='.';
seq_names={
	'a',...
	'b',...
	'c',...
	'd',...
	'e',...
    };
n_frames = 0;
out_fps = 30;
quality = 100;
input_format='mp4';
output_fmt='mp4';
% profile='Motion JPEG AVI';
profile='MPEG-4';
n_seq=length(seq_names);
out_fname='merged';
out_fname=sprintf('%s_%d_%d.%s', out_fname, out_fps, quality, output_fmt);
fprintf('Writing output video to: %s\n', out_fname);
outputVideo = VideoWriter(fullfile(root_dir, out_fname), profile);
outputVideo.FrameRate = out_fps;
outputVideo.Quality = quality;
open(outputVideo);
frame_id=0;
for seq_id = 1:n_seq
    filename=seq_names{seq_id};
    if ~isempty(input_format)
        file_path=sprintf('%s/%s.%s', root_dir,filename,input_format);
    else
        file_path=sprintf('%s/%s',root_dir,filename);
    end
    fprintf('Reading input video %d from: %s\n', seq_id, file_path);
    inputVideo = VideoReader(file_path);
	while ~hasFrame(inputVideo)
		frame_id=frame_id+1;
		current_frame = readFrame(inputVideo);
		writeVideo(outputVideo, current_frame);
		if mod(frame_id, 10)==0
			fprintf('\tDone processing %d frames\n',frame_id);
		end
		if n_frames > 0 && frame_id >= n_frames
			break;
		end
	end
% 	close(inputVideo);
end    
close('all');


