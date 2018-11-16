% root_dir='.';
root_dir='H:\UofA\Acamp\code\object_detection\videos';

read_from_list = 1;

if read_from_list
    file_list=importdata(sprintf('%s/list.txt',root_dir))    
    format='';
else
    file_list={'M2U00021'}
    format='MPG';
end
for file_id=1:length(file_list)
    filename=file_list{file_id};   
    if ~isempty(format)
        file_path=sprintf('%s/%s.%s',root_dir,filename,format)
        image_dir=sprintf('%s/%s',root_dir,filename);
    else
        file_path=sprintf('%s/%s',root_dir,filename)
        [pathstr,name,ext] = fileparts(file_path); 
        image_dir=sprintf('%s/%s',root_dir,name);
    end
    if(~exist(image_dir, 'dir'))
        mkdir(image_dir);
    end
    video = VideoReader(file_path);
    fprintf('Processing file: %s\n',file_path);
    frame_id=0;
    while hasFrame(video)
        frame_id=frame_id+1;
        fprintf('\tDone processing %d frames\n',frame_id);
        img = readFrame(video);
        imwrite(img, sprintf('%s/image%06d.jpg',image_dir, frame_id), 'Quality', 100);
    end
end


