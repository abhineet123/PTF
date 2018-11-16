img_fid=fopen('frames_gs.bin');
start_id=219;
end_id=219;
img_width=800;
img_height=600;
nframes=531;
img_fig=figure;
img_size=img_width*img_height;
if start_id>1
    fseek(img_fid, img_size*(start_id-1), 'bof');
end
for i=start_id:end_id
    img_bin=fread(img_fid, [img_width img_height], 'uint8', 'a');
    img_bin=uint8(img_bin');
    figure(img_fig), imshow(img_bin);
    getframe(img_fig);    
end