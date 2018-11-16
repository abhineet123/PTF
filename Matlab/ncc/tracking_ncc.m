%   File: 'tracking_ncc.m'
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

function [H Warped_original] = tracking_ncc(ICur_gray, H, Hessian_inverse, cte, Template, Nx, size_template_x, size_template_y, maxIters, epsilon)

% Auxiliary matrices
[imx imy] = meshgrid(1:size_template_x*2, 1:size_template_y*2);
imx = imx(:);
imy = imy(:);

% Pre-computing mean and std of Template 
c_bar = mean(Template);
c = sqrt(sum((Template - c_bar).^2));

% Runs for a maximum number of 'maxIters' iterations
for iters = 1:maxIters
    
    % Computes warped image and gradient
    Warped_original = warp(double(ICur_gray), H, size_template_x*2, size_template_y*2);
    Warped = Warped_original(:);

    % Initialization
    g = zeros(8, 1);
    a_der = g;
    b_der = a_der;    
    
    % Computes NCC gradient
    b_bar = mean(Warped);
    b = sqrt(sum((Warped - b_bar).^2));    

    a = sum((Warped - b_bar).*(Template - c_bar));
    
    % Populating gradient vector g
    for k = 1:8
        a_der(k) = sum((Template - c_bar).*(cte(:,k) - mean(cte(:,k))));
        b_der(k) = (1/b)*sum((Warped - b_bar).*(cte(:,k) - mean(cte(:,k))));

        g(k) = (1/(b*c))*a_der(k) - (a/(b^2*c))*b_der(k);
    end
    
    % Update
    p = -Hessian_inverse*g;
    
    A = [p(5),p(3),p(1); p(4),-p(5)-p(6),p(2); p(7),p(8),p(6)];
    H = H*expm(A);
    
    if sum(abs(p))<epsilon
        break;
    end
end

iters

return
