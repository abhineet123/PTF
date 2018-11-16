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


function [Hessian cte] = generates_hessian(Warped, Template, size_template_x, size_template_y, Nx)

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

% Initializing vector of gradients g and Hessian matrix
g = zeros(8, 1);
Hessian = zeros(8);

% Other inits
a_der = g;
b_der = a_der;
cte = zeros(Nx, 8);
cte2 = zeros(Nx, 2, 8);

% Computing NCC coefficient
b_bar = mean(Warped);
b = sqrt(sum((Warped-b_bar).^2));

c_bar = mean(Template);
c = sqrt(sum((Template-c_bar).^2));

a = sum((Warped - b_bar).*(Template - c_bar));

% Populating g

% 1a Derivada
cte(:,1) = Ix;
cte(:,2) = Iy;
cte(:,3) = imy.*Ix;
cte(:,4) = imx.*Iy;
cte(:,5) = -imy.*Iy + imx.*Ix;
cte(:,6) = -imx.*Ix-2*imy.*Iy;
cte(:,7) = (-imx.*Ix-imy.*Iy).*imx;
cte(:,8) = (-imx.*Ix-imy.*Iy).*imy;

% 2a Derivada
cte2(:,1,1) = ones(Nx,1);
cte2(:,2,1) = zeros(Nx,1);

cte2(:,1,2) = zeros(Nx,1);
cte2(:,2,2) = ones(Nx,1);

cte2(:,1,3) = imy;
cte2(:,2,3) = zeros(Nx,1);

cte2(:,1,4) = zeros(Nx,1);
cte2(:,2,4) = imx;

cte2(:,1,5) = imx;
cte2(:,2,5) = -imy;

cte2(:,1,6) = -imx;
cte2(:,2,6) = -2.*imy;

cte2(:,1,7) = -imx.*imx;
cte2(:,2,7) = -imy.*imx;

cte2(:,1,8) = -imx.*imy;
cte2(:,2,8) = -imy.*imy;

for k = 1:8
    a_der(k) = sum((Template - c_bar).*(cte(:,k) - mean(cte(:,k))));
    b_der(k) = (1/b)*sum((Warped - b_bar).*(cte(:,k) - mean(cte(:,k))));
    g(k) = (1/(b*c))*a_der(k) - (a/(b^2*c))*b_der(k); 
end

% Populating Hessian matrix
for k = 1:8
    for l = 1:8
        Hessian(k,l) = 1/(b*c)*sum((Warped - b_bar).*((Ixx.*cte2(:,1,l)+Ixy.*cte2(:,2,l)).*cte2(:,1,k)+(Iyx.*cte2(:,1,l) + Iyy.*cte2(:,2,l)).*cte2(:,2,k) - mean((Ixx.*cte2(:,1,l)+Ixy.*cte2(:,2,l)).*cte2(:,1,k)+(Iyx.*cte2(:,1,l) + Iyy.*cte2(:,2,l)).*cte2(:,2,k)))) ...
                        - (1/(b*b*c))*a_der(l)*b_der(k) - (1/(b*b*c))*a_der(k)*b_der(l);
    end
end

