% close all;
clear all;
colRGBDefs;
getParamLists;

img_root_dir=sprintf('../../Image Data');
root_dir='../C++/MTF_LIB/log';

actor_id = 0;
seq_id = 3;
plot_id = 0;
update_type = 1;
opt_type = 1;
am_name = 'ssim';
ssm_name = '2';

actor = actors{actor_id+1};
seq_name = sequences{actor_id + 1}{seq_id + 1};
show_img = 0;

src_img_fname=sprintf('%s/%s.bin', img_root_dir, seq_name);

pause_exec=1;
end_exec=0;
speed_factor=1;
frame_id_diff=1;

if show_img
    if ~exist(src_img_fname,'file')
        error('Image data file is not present:\n%s', src_img_fname);
    end
    img_fid=fopen(src_img_fname);
    img_width=fread(img_fid, 1, 'uint32', 'a');
    img_height=fread(img_fid, 1, 'uint32', 'a');
    if isempty(img_width) || isempty(img_height)
        error('Image data file is empty: \n%s', src_img_fname);
    end
    img_size=img_width*img_height;
    img_start_pos=ftell(img_fid);
    
    img_pos=zeros(file_end_id, 1);
    for frame_id=1:file_end_id
        img_pos(frame_id)=img_start_pos + img_size*(frame_id-1);
    end
    
    img_fig=figure;
    set(img_fig, 'Position', [1000, 50, 800, 600]);
    set(img_fig, 'Name', seq_name);
end

set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 20);
set(0,'DefaultAxesFontWeight', 'bold');

bin_data_fname = sprintf('%s/diagnostics/%s/lscv_data.bin',...
    root_dir, seq_name);
if ~exist(bin_data_fname,'file')
    error('LSCV data file is not present:\n%s', bin_data_fname);
end
lscv_file_id=fopen(bin_data_fname);
n_bins=fread(lscv_file_id, 1 , 'uint32', 'a');
if isempty(n_bins) 
    error('LSCV data file is empty\n');
end

main_fig=figure;
% hist_fig=figure;
pmainfig=get(main_fig);
x_data=0:n_bins-1;

start_id = 1;
end_id = 400;

frame_id=start_id;
while frame_id<=end_id
    joint_hist=fread(lscv_file_id, [n_bins n_bins] , 'float64', 'a');
    intensity_map=fread(lscv_file_id, [n_bins 1] , 'float64', 'a');
    intensity_map_linear=fread(lscv_file_id, [n_bins 1] , 'float64', 'a');
    
    if isempty(joint_hist) || isempty(intensity_map) || isempty(intensity_map_linear)
        break;
    end       
    
    joint_hist=flipud(joint_hist);
    
    plot_title=sprintf('Frame %d', frame_id);
    set(0,'CurrentFigure',main_fig);       
    
    subplot(1, 2, 1);
    
%     imagesc(joint_hist);     
    
    joint_hist_img = mat2gray(joint_hist);
    imshow(joint_hist_img);
    
    subplot(1, 2, 2);
    hold off;
    plot(x_data, intensity_map, 'r-');    
    hold on, grid on;
    plot(x_data, intensity_map_linear, 'g--');    
    legend({'intensity_map', 'intensity_map_linear'}, 'interpreter', 'none');
    title(plot_title, 'interpreter','none');
    
    if show_img
        fseek(img_fid, img_pos(curr_frame_id), 'bof');
        img_bin=fread(img_fid, [img_width img_height], 'uint8', 'a');
        img_bin=uint8(img_bin');
        set(0,'CurrentFigure',img_fig);
        imshow(img_bin);
    end
    pause(0.1);
    frame_id=frame_id+1;
end
fclose(lscv_file_id);
