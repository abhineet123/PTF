fname='xv_frame_py.txt';
fname2='init_template_py.txt';
fname3='init_template.txt';
img=importdata(fname2);
img_height=size(img, 1)
img_width=size(img, 2);
new_width=img_width/3

% img=reshape(img',[1, img_height*img_width]);
% redChannel = reshape(img(1:3:end), [new_width, img_height]);
% greenChannel = reshape(img(2:3:end), [new_width, img_height]);
% blueChannel = reshape(img(3:3:end), [new_width, img_height]);
% rgbImage = cat(3, redChannel', greenChannel', blueChannel');

redChannel = img(:, 1:3:end);
greenChannel = img(:, 2:3:end);
blueChannel = img(:, 3:3:end);
rgbImage = cat(3, redChannel, greenChannel, blueChannel);

imshow(uint8(rgbImage));