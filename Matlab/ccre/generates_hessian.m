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

% Computing joint distribution
p_joint = zeros(n_bins+3, n_bins+3);
computes_p_joint_slow;

% Computing cumulative distribution
p_zao = zeros(size(p_joint));

for k = 1:n_bins+3
    for l = 1:n_bins+3    
        p_zao(k, l) = sum(p_joint(k, l:end));
    end
end

% Marginals
soma_pzao = sum(p_zao);

p_ref = sum(p_joint);
p_cur = sum(p_joint');

% Initialization
G = zeros(8, 1);
Hess1 = zeros(8);
Hess2 = zeros(8);
Hess3 = zeros(8);

dp = zeros(n_bins+3, n_bins+3, 8);
spline_cur = zeros(Nx, n_bins+3);
spline_ref_der = zeros(Nx, n_bins+3);

% Computing kernel and derivatives for every pixel in image
for k = 0:n_bins+2
    spline_cur(:,k+1) = comp_spline(k - 1 - Template);
    spline_ref_der(:,k+1) = comp_spline_der(k - 1 - Template);
end

spline_ref = spline_cur;

% Computing image derivatives
cte = zeros(numel(Ix), 2, 8);

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

% Populating gradient vector g
for k = 0:n_bins+2 
    for j = 0:n_bins+2

        cte_part = ( log_r( p_zao(k+1,j+1)*inv_prob(p_cur(k+1)*soma_pzao(j+1))));
            
        dp(k+1, j+1, 1) = (1/(Nx))*sum(spline_cur(:,k+1).*spline_cur(:,j+1).*(cte(:,1,1).*Ix + cte(:,2,1).*Iy));
        dp(k+1, j+1, 2) = (1/(Nx))*sum(spline_cur(:,k+1).*spline_cur(:,j+1).*(cte(:,1,2).*Ix + cte(:,2,2).*Iy));
        dp(k+1, j+1, 3) = (1/(Nx))*sum(spline_cur(:,k+1).*spline_cur(:,j+1).*(cte(:,1,3).*Ix + cte(:,2,3).*Iy));
        dp(k+1, j+1, 4) = (1/(Nx))*sum(spline_cur(:,k+1).*spline_cur(:,j+1).*(cte(:,1,4).*Ix + cte(:,2,4).*Iy));
        dp(k+1, j+1, 5) = (1/(Nx))*sum(spline_cur(:,k+1).*spline_cur(:,j+1).*(cte(:,1,5).*Ix + cte(:,2,5).*Iy));
        dp(k+1, j+1, 6) = (1/(Nx))*sum(spline_cur(:,k+1).*spline_cur(:,j+1).*(cte(:,1,6).*Ix + cte(:,2,6).*Iy));
        dp(k+1, j+1, 7) = (1/(Nx))*sum(spline_cur(:,k+1).*spline_cur(:,j+1).*(cte(:,1,7).*Ix + cte(:,2,7).*Iy));
        dp(k+1, j+1, 8) = (1/(Nx))*sum(spline_cur(:,k+1).*spline_cur(:,j+1).*(cte(:,1,8).*Ix + cte(:,2,8).*Iy));
        
        G(1) = G(1) + cte_part.*dp(k+1, j+1, 1);
        G(2) = G(2) + cte_part.*dp(k+1, j+1, 2);
        G(3) = G(3) + cte_part.*dp(k+1, j+1, 3);
        G(4) = G(4) + cte_part.*dp(k+1, j+1, 4);
        G(5) = G(5) + cte_part.*dp(k+1, j+1, 5);
        G(6) = G(6) + cte_part.*dp(k+1, j+1, 6);
        G(7) = G(7) + cte_part.*dp(k+1, j+1, 7);
        G(8) = G(8) + cte_part.*dp(k+1, j+1, 8);
    end
end

% Populating Hessian matrix
aux = zeros(n_bins+1,n_bins+1,8);
for j = 0:n_bins+2   
    for i = 0:n_bins+2 
        
        for m = 1:8 
            aux(i+1,j+1,m) = (1/Nx)*sum(spline_ref(:,j+1).*spline_cur(:,k+1).*(cte(:,1,m).*Ix + cte(:,2,m).*Iy));
        end
        
    end
end

aux2 = zeros(n_bins+1, 8);
for j = 1:n_bins+3
    for m = 1:8
        aux2(j, m) = sum(aux(:,j,m));
    end
end

 for j = 0:n_bins+2  
    for k = 0:n_bins+2
        
        for l = 1:8
            for m = 1:8
                % Third term in eq 16
                Hess1(l, m) = Hess1(l, m) - inv_prob(soma_pzao(j+1))*...
                                         ((1/Nx)*sum(spline_ref(:,j+1).*spline_cur(:,k+1).*(cte(:,1,m).*Ix + cte(:,2,m).*Iy)))*...
                                           aux2(j+1, l);
                
                % Second term in eq 16                 
                Hess2(l, m) = Hess2(l, m) + inv_prob(p_zao(k+1,j+1))*...
                    (1/Nx)*sum(spline_cur(:,k+1).*spline_ref(:,j+1).*(cte(:,1,m).*Ix + cte(:,2,m).*Iy))*...
                    (1/Nx)*sum(spline_cur(:,k+1).*spline_ref(:,j+1).*(cte(:,1,l).*Ix + cte(:,2,l).*Iy));
                
                % First term in eq 16 
                Hess3(l, m) = Hess3(l, m) + log_r( p_zao(k+1,j+1)*inv_prob(p_cur(k+1)*soma_pzao(j+1)) )*...
                                (1/Nx)*sum(spline_cur(:,k+1).*(-spline_ref_der(:,j+1).*(cte(:,1,m).*Ix + cte(:,2,m).*Iy).*(cte(:,1,l).*Ix + cte(:,2,l).*Iy) + ...
                                 spline_ref(:,j+1).*(Ixx.*cte(:,1,m).*cte(:,1,l) + Ixy.*cte(:,2,m).*cte(:,1,l) + Iyx.*cte(:,1,m).*cte(:,2,l) + Iyy.*cte(:,2,m).*cte(:,2,l)) ));
                
            end
        end        
    end
 end
 
% Hessian is a sum of the three terms
Hessian = (Hess1 + Hess2 + Hess3);