function frm = GetOneFrm(x,y,width,height, im)
% given image center point (x,y), width and height, get image subarea
% im = rgb2gray(im);
w = width/2;
h = height/2;
frm = im(y-w+1:y+w,x-h+1:x+h,:);
end