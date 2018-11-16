%   File: 'generates_hessian.m'
%
%   Author(s):  Rogerio Richa
%   Created on: 2011
% 
%   (C) Copyright 2006-2011 Johns Hopkins University (JHU), All Rights
%   Reserved.
% 
% --- begin cisst license - do not edit ---
% 
% This software is provided "as is" under an open source license, with
% no warranty.  The complete license can be found in license.txt and
% http://www.cisst.org/cisst/license.txt.
% 
% --- end cisst license ---

function Hessian = generates_hessian(Template, size_template_x, size_template_y, Nx, n_bins, size_bin)

% Quantizing input images
Template = Template./size_bin;
Warped = Template; 

% Computing Template first and second-order gradients
[Ix, Iy] = gradient(Template);
[Ixx, Ixy] = gradient(Ix);
[Iyx, Iyy] = gradient(Iy);

Ix = Ix(:);
Iy = Iy(:);
Ixx = Ixx(:);
Ixy = Ixy(:);
Iyx = Iyx(:);
Iyy = Iyy(:);

% Auxiliary matrices
[imx imy] = meshgrid(1:size_template_x*2, 1:size_template_y*2);
imx = imx(:);
imy = imy(:);

% Reshaping Template
Template = Template(:);
Warped = Warped(:);

% computes histograms
p_joint = zeros(n_bins+3, n_bins+3);

computes_p_joint_slow;

p_ref = sum(p_joint);
p_cur = sum(p_joint');

% Initializing g and Hess
g = zeros(8, 1);
Hessian = zeros(8);

dp = zeros(n_bins+3, n_bins+3, 8);
spline_cur = zeros(Nx, n_bins+3);
spline_ref_der = zeros(Nx, n_bins+3);
spline_ref_sec_der = zeros(Nx, n_bins+3);

% Computing kernel derivatives for every pixel on image
for k = 0:n_bins+2
    spline_cur(:,k+1) = comp_spline(k - 1 - Warped);
    spline_ref_der(:,k+1) = comp_spline_der(k - 1 - Template);
    spline_ref_sec_der(:, k+1) = comp_spline_sec_der(k - 1- Template);
end

% Computing image gradients
cte = zeros(Nx, 2, 8);

cte(:,1,1) = ones(Nx,1);
cte(:,2,1) = zeros(Nx,1);

cte(:,1,2) = zeros(Nx,1);
cte(:,2,2) = ones(Nx,1);

cte(:,1,3) = imy;
cte(:,2,3) = zeros(Nx,1);

cte(:,1,4) = zeros(Nx,1);
cte(:,2,4) = imx;

cte(:,1,5) = imx;
cte(:,2,5) = -imy;

cte(:,1,6) = -imx;
cte(:,2,6) = -2.*imy;

cte(:,1,7) = -imx.*imx;
cte(:,2,7) = -imy.*imx;

cte(:,1,8) = -imx.*imy;
cte(:,2,8) = -imy.*imy;

% Populating vector of gradients g
for k = 0:n_bins+2 
    for j = 0:n_bins+2 
        
        cte_part = (log_r(exp(1)*p_joint(k+1, j+1) / (p_ref(j+1))) );
        
        dp(k+1, j+1, 1) = (1/(Nx))*sum(spline_cur(:,k+1).*-spline_ref_der(:,j+1).*(cte(:,1,1).*Ix + cte(:,2,1).*Iy));
        dp(k+1, j+1, 2) = (1/(Nx))*sum(spline_cur(:,k+1).*-spline_ref_der(:,j+1).*(cte(:,1,2).*Ix + cte(:,2,2).*Iy));
        dp(k+1, j+1, 3) = (1/(Nx))*sum(spline_cur(:,k+1).*-spline_ref_der(:,j+1).*(cte(:,1,3).*Ix + cte(:,2,3).*Iy));
        dp(k+1, j+1, 4) = (1/(Nx))*sum(spline_cur(:,k+1).*-spline_ref_der(:,j+1).*(cte(:,1,4).*Ix + cte(:,2,4).*Iy));
        dp(k+1, j+1, 5) = (1/(Nx))*sum(spline_cur(:,k+1).*-spline_ref_der(:,j+1).*(cte(:,1,5).*Ix + cte(:,2,5).*Iy));
        dp(k+1, j+1, 6) = (1/(Nx))*sum(spline_cur(:,k+1).*-spline_ref_der(:,j+1).*(cte(:,1,6).*Ix + cte(:,2,6).*Iy));
        dp(k+1, j+1, 7) = (1/(Nx))*sum(spline_cur(:,k+1).*-spline_ref_der(:,j+1).*(cte(:,1,7).*Ix + cte(:,2,7).*Iy));
        dp(k+1, j+1, 8) = (1/(Nx))*sum(spline_cur(:,k+1).*-spline_ref_der(:,j+1).*(cte(:,1,8).*Ix + cte(:,2,8).*Iy));
                
        g(1) = g(1) + cte_part*dp(k+1, j+1, 1);
        g(2) = g(2) + cte_part*dp(k+1, j+1, 2);
        g(3) = g(3) + cte_part*dp(k+1, j+1, 3);
        g(4) = g(4) + cte_part*dp(k+1, j+1, 4);
        g(5) = g(5) + cte_part*dp(k+1, j+1, 5);
        g(6) = g(6) + cte_part*dp(k+1, j+1, 6);
        g(7) = g(7) + cte_part*dp(k+1, j+1, 7);
        g(8) = g(8) + cte_part*dp(k+1, j+1, 8);
    end
end

% Populating Hessian matrix
for k = 0:n_bins+2 
    for j = 0:n_bins+2
        
        cte_2 = (1/(Nx))*(1+log_r(p_joint(k+1, j+1)/ p_ref(j+1) ));
        
        cte_inv_prob = ( inv_prob(p_joint(k+1, j+1)) - inv_prob(p_ref(j+1)) ) ;
        
        for l = 1:8
            for m = 1:8
                Hessian(m,l) = Hessian(m,l) +  dp(k+1, j+1, l)*dp(k+1, j+1, m)*cte_inv_prob ...
                    + cte_2*sum(spline_cur(:,k+1).*(spline_ref_sec_der(:,j+1).*(cte(:,1,m).*Ix + cte(:,2,m).*Iy).*(cte(:,1,l).*Ix + cte(:,2,l).*Iy) ...
                    - spline_ref_der(:,j+1).*(Ixx.*cte(:,1,m).*cte(:,1,l) + Ixy.*cte(:,2,m).*cte(:,1,l) + Iyx.*cte(:,1,m).*cte(:,2,l) + Iyy.*cte(:,2,m).*cte(:,2,l))));
            end
        end
        
    end
end










