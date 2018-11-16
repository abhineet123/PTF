clear all;
close all;

root_dir='../../Datasets';
actor='TMT';
seq_name='nl_cereal_s3';
frame_id=1;
interp_method='cubic';

eps_base=1:10;
eps_exp=1:10;

n_base=length(eps_base);
n_exp=length(eps_exp);
n_eps=n_exp*n_base;

plot_x=zeros(n_eps, 1);
plot_y=zeros(n_eps, 1);

img_path=sprintf('%s/%s/%s/frame%05d.jpg', root_dir, actor, seq_name, frame_id);
img=imread(img_path);
img_gs=rgb2gray(img);

img_height=size(img_gs, 1);
img_width=size(img_gs, 2);

img_x=1:img_width;
img_y=1:img_height;

min_x=17;
max_x=253;
min_y=300;
max_y=560;

deriv_x=min_x:max_x;
deriv_y=min_y:max_y;
pix_val=double(img_gs(deriv_y, deriv_x));

plot_id=1;
for i=1:n_exp
    curr_exp=eps_exp(i);
    for j=1:n_base
        curr_base=eps_base(j);
        hess_eps=curr_base * 10.^-curr_exp;
        plot_x(plot_id)=hess_eps;      

        hess_inc_x=deriv_x + hess_eps;
        hess_dec_x=deriv_x - hess_eps;
        
        [hess_inc_x_gridx, hess_inc_x_gridy]=meshgrid(hess_inc_x, deriv_y);
        hess_inc_x_pix_val=interp2(img_x, img_y, double(img_gs), hess_inc_x_gridx, hess_inc_x_gridy, interp_method);
        [hess_dec_x_gridx, hess_dec_x_gridy]=meshgrid(hess_dec_x, deriv_y);
        hess_dec_x_pix_val=interp2(img_x, img_y, double(img_gs), hess_dec_x_gridx, hess_dec_x_gridy, interp_method);
        img_hess_xx=(hess_inc_x_pix_val + hess_dec_x_pix_val - (pix_val.*2))./(hess_eps*hess_eps);        
        plot_y(plot_id)=img_hess_xx(1, 1);
        plot_id=plot_id+1;
    end
end       

figure, plot(plot_x, plot_y), title('img_hess_xx', 'interpret', 'none'), grid on;

