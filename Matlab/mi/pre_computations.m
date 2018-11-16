%   File: 'pre_computations.m'
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

% Pre-computing image gradients
[imx imy] = meshgrid(1:size_template_x*2, 1:size_template_y*2);
imx = imx(:);
imy = imy(:);

[Ix, Iy] = gradient(Template./size_bin);
Ix = Ix(:);
Iy = Iy(:);

cte_t = zeros(numel(Ix), 8);

cte_t(:,1) = Ix;
cte_t(:,2) = Iy;
cte_t(:,3) = imy.*Ix;
cte_t(:,4) = imx.*Iy;
cte_t(:,5) = -imy.*Iy + imx.*Ix;
cte_t(:,6) = -imx.*Ix-2*imy.*Iy;
cte_t(:,7) = (-imx.*Ix-imy.*Iy).*imx;
cte_t(:,8) = (-imx.*Ix-imy.*Iy).*imy;

% Pre-computing Hessian matrix and its inverse
Hessian = generates_hessian(Template, size_template_x, size_template_y, Nx, n_bins, size_bin);
Hessian_inverse = eye(8)/Hessian;
 
Template = Template(:)./size_bin;

% Computing Kernel and derivatives
spline_ref = computes_gradient(Template, Nx, n_bins);

spline_ref_der = zeros(Nx, n_bins+3);
for k = 0:n_bins+2
    spline_ref_der(:,k+1) = comp_spline_der(k-1-Template);
end
