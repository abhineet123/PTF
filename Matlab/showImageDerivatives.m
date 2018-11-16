clear all;
close all;

root_dir='../../Datasets';
actor='TMT';
seq_name='nl_cereal_s3';
frame_id=1;
interp_method='cubic';

border_size=1;
grad_eps=1e-10;
hess_eps=1e-4;

img_path=sprintf('%s/%s/%s/frame%05d.jpg', root_dir, actor, seq_name, frame_id);
img=imread(img_path);
img_gs=rgb2gray(img);

img_height=size(img_gs, 1);
img_width=size(img_gs, 2);

img_x=1:img_width;
img_y=1:img_height;

min_x=10;
max_x=600;
min_y=10;
max_y=560;

deriv_x=min_x:max_x;
deriv_y=min_y:max_y;
% deriv_x=border_size:img_width-border_size;
% deriv_y=border_size:img_height-border_size;

grad_inc_x=deriv_x + grad_eps;
grad_dec_x=deriv_x - grad_eps;
grad_inc_y=deriv_y + grad_eps;
grad_dec_y=deriv_y - grad_eps;


% [deriv_x_grid, deriv_y_grid]=meshgrid(deriv_x, deriv_y);
pix_val=double(img_gs(deriv_y, deriv_x));

[grad_inc_x_gridx, grad_inc_x_gridy]=meshgrid(grad_inc_x, deriv_y);
grad_inc_x_pix_val=interp2(img_x, img_y, double(img_gs), grad_inc_x_gridx, grad_inc_x_gridy, interp_method);
[grad_dec_x_gridx, grad_dec_x_gridy]=meshgrid(grad_dec_x, deriv_y);
grad_dec_x_pix_val=interp2(img_x, img_y, double(img_gs), grad_dec_x_gridx, grad_dec_x_gridy, interp_method);
img_grad_x=(grad_inc_x_pix_val - grad_dec_x_pix_val)./(2*grad_eps);

[grad_inc_y_gridx, grad_inc_y_gridy]=meshgrid(deriv_x, grad_inc_y);
grad_inc_y_pix_val=interp2(img_x, img_y, double(img_gs), grad_inc_y_gridx, grad_inc_y_gridy, interp_method);
[grad_dec_y_gridx, grad_dec_y_gridy]=meshgrid(deriv_x, grad_dec_y);
grad_dec_y_pix_val=interp2(img_x, img_y, double(img_gs), grad_dec_y_gridx, grad_dec_y_gridy, interp_method);
img_grad_y=(grad_inc_y_pix_val - grad_dec_y_pix_val)./(2*grad_eps);

img_grad_x_norm=img_grad_x./max(max(img_grad_x));
img_grad_y_norm=img_grad_y./max(max(img_grad_y));

% figure, imshow(img_grad_x_norm), title('img_grad_x', 'interpret', 'none');
% figure, imshow(img_grad_y_norm), title('img_grad_y', 'interpret', 'none');

% figure, surf(deriv_x, deriv_y, img_grad_x,...
%     'FaceColor', 'interp', 'FaceAlpha', 0.5),...
%     title('img_grad_x', 'interpret', 'none');
% 
% figure, surf(deriv_x, deriv_y, img_grad_y,...
%     'FaceColor', 'interp', 'FaceAlpha', 0.5),...
%     title('img_grad_y', 'interpret', 'none');

hess_inc_x=deriv_x + hess_eps;
hess_dec_x=deriv_x - hess_eps;
hess_inc_y=deriv_y + hess_eps;
hess_dec_y=deriv_y - hess_eps;

[hess_inc_x_gridx, hess_inc_x_gridy]=meshgrid(hess_inc_x, deriv_y);

hess_inc_x_pix_val=interp2(img_x, img_y, double(img_gs), hess_inc_x_gridx, hess_inc_x_gridy, interp_method);
[hess_dec_x_gridx, hess_dec_x_gridy]=meshgrid(hess_dec_x, deriv_y);
hess_dec_x_pix_val=interp2(img_x, img_y, double(img_gs), hess_dec_x_gridx, hess_dec_x_gridy, interp_method);

img_hess_xx=(hess_inc_x_pix_val + hess_dec_x_pix_val - (pix_val.*2))./(hess_eps*hess_eps);

[hess_inc_y_gridx, hess_inc_y_gridy]=meshgrid(deriv_x, hess_inc_y);

hess_inc_y_pix_val=interp2(img_x, img_y, double(img_gs), hess_inc_y_gridx, hess_inc_y_gridy, interp_method);
[hess_dec_y_gridx, hess_dec_y_gridy]=meshgrid(deriv_x, hess_dec_y);
hess_dec_y_pix_val=interp2(img_x, img_y, double(img_gs), hess_dec_y_gridx, hess_dec_y_gridy, interp_method);
img_hess_yy=(hess_inc_y_pix_val + hess_dec_y_pix_val - (pix_val.*2))./(hess_eps*hess_eps);

img_hess_xx_norm=img_hess_xx./max(max(img_hess_xx));
img_hess_yy_norm=img_hess_yy./max(max(img_hess_yy));

figure, imshow(img_hess_xx_norm), title('img_hess_xx', 'interpret', 'none');
figure, imshow(img_hess_yy_norm), title('img_hess_yy', 'interpret', 'none');
% figure, imshow(uint8(pix_val)), title('pix_val', 'interpret', 'none');

