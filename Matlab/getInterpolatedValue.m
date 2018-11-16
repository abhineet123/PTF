function [ val ] = getInterpolatedValue(x_vec, y_vec, z_arr, x, y)

diff_x_vec=x_vec-x;
diff_y_vec=y_vec-y;

diff_x_vec(diff_x_vec>0)=-inf;
diff_y_vec(diff_y_vec>0)=-inf;
[~,idx1]=max(diff_x_vec);
[~,idy1]=max(diff_y_vec);







img_size=size(img);
pts_x1=floor(pts_x);
pts_x2=pts_x1+1;
diff_x=pts_x-pts_x1;



pts_y1=floor(pts_y);
pts_y2=pts_y1+1;
diff_y=pts_y-pts_y1;

p11=img(sub2ind(img_size, pts_y1, pts_x1));
p12=img(sub2ind(img_size, pts_y2, pts_x1));
p21=img(sub2ind(img_size, pts_y1, pts_x2));
p22=img(sub2ind(img_size, pts_y2, pts_x2));

p1=((1-diff_x).*p11) + (diff_x.*p21);
p2=((1-diff_x).*p12) + (diff_x.*p22);

pixel_vals=((1-diff_y).*p1)+(diff_y.*p2);

end




