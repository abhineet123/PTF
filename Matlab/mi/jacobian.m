%   File: 'jacobian.m'
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

function J = jacobian(grad_x, grad_y, Nx, size_template_x, size_template_y)

[imx,imy] = meshgrid(1:size_template_x, 1:size_template_y);

x = reshape(imx, Nx, 1);
y = reshape(imy, Nx, 1);

Ix = reshape(grad_x, Nx, 1);
Iy = reshape(grad_y, Nx, 1);

xIx = x.*Ix; yIx = y.*Ix; xIy = x.*Iy; yIy = y.*Iy; temp = -xIx-yIy;

J = [Ix, Iy, yIx, xIy, -yIy + xIx, temp-yIy, temp.*x, temp.*y];

return