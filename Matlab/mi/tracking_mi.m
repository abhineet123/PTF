%   File: 'tracking_mi.m'
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

function [H Warped_original] = tracking_mi(ICur_gray, H, inverse, Template, size_bin, n_bins, Nx, size_template_x, size_template_y, cte_r, spline_ref, spline_ref_der, max_iters, epsilon)
 
% Simple inits
[imx imy] = meshgrid(1:size_template_x*2, 1:size_template_y*2);
imx = reshape(imx, numel(imx), 1);
imy = reshape(imy, numel(imy), 1);

% Runs for a maximum number of 'maxIters' iterations
for iters = 1:max_iters
    
    % Computes warped image and gradient
    Warped_original = warp(double(ICur_gray), H, size_template_x*2, size_template_y*2);
    Warped = Warped_original(:)./size_bin;

%     % Forwards compositional approach    
%     [Ix, Iy] = gradient(Warped_original./size_bin);
%     Ix = Ix(:);
%     Iy = Iy(:);
        
    % Computes joint histogram
    p_joint = computes(Warped', Template', Nx, n_bins);

    p_ref = sum(p_joint);
    p_cur = sum(p_joint');

%     % Forwards compositional approach
%     % Computing Image derivatives
%     cte = zeros(numel(Ix), 8);
% 
%     cte(:,1) = Ix;
%     cte(:,2) = Iy;
%     cte(:,3) = imy.*Ix;
%     cte(:,4) = imx.*Iy;
%     cte(:,5) = -imy.*Iy + imx.*Ix;
%     cte(:,6) = -imx.*Ix-2*imy.*Iy;
%     cte(:,7) = (-imx.*Ix-imy.*Iy).*imx;
%     cte(:,8) = (-imx.*Ix-imy.*Iy).*imy;

    % Initializing g 
    G = zeros(8, 1);
    dp = zeros(n_bins+3, n_bins+3, 8);

%     % Forwards compositional approach
%     % Computing Kernel derivatives
%     spline_cur_der = zeros(Nx, n_bins+3);
%  
%     for k = 0:n_bins+2
%         spline_cur_der(:,k+1) = comp_spline_der(k-1-Warped);
%     end 
% 
%     % Populating g
%     for k = 0:n_bins+2
%         for j = 0:n_bins+2 
%             
%             constante = -(1/(Nx))*(log_r(exp(1)*p_joint(k+1, j+1) / p_cur(k+1) ) );
%             
%             dp(k+1, j+1, 1) = constante.*sum(spline_ref(:,j+1).*spline_cur_der(:,k+1).*cte(:,1));
%             dp(k+1, j+1, 2) = constante.*sum(spline_ref(:,j+1).*spline_cur_der(:,k+1).*cte(:,2));
%             dp(k+1, j+1, 3) = constante.*sum(spline_ref(:,j+1).*spline_cur_der(:,k+1).*cte(:,3));
%             dp(k+1, j+1, 4) = constante.*sum(spline_ref(:,j+1).*spline_cur_der(:,k+1).*cte(:,4));
%             dp(k+1, j+1, 5) = constante.*sum(spline_ref(:,j+1).*spline_cur_der(:,k+1).*cte(:,5));
%             dp(k+1, j+1, 6) = constante.*sum(spline_ref(:,j+1).*spline_cur_der(:,k+1).*cte(:,6));
%             dp(k+1, j+1, 7) = constante.*sum(spline_ref(:,j+1).*spline_cur_der(:,k+1).*cte(:,7));
%             dp(k+1, j+1, 8) = constante.*sum(spline_ref(:,j+1).*spline_cur_der(:,k+1).*cte(:,8));
%             
%             G(1) = G(1) + dp(k+1, j+1, 1);
%             G(2) = G(2) + dp(k+1, j+1, 2);
%             G(3) = G(3) + dp(k+1, j+1, 3);
%             G(4) = G(4) + dp(k+1, j+1, 4);
%             G(5) = G(5) + dp(k+1, j+1, 5);
%             G(6) = G(6) + dp(k+1, j+1, 6);
%             G(7) = G(7) + dp(k+1, j+1, 7);
%             G(8) = G(8) + dp(k+1, j+1, 8);
%         end
%     end

    % Inverse compositional approach    
    spline_cur = computes_gradient(Warped, Nx, n_bins);

    % Populating g
    for k = 0:n_bins+2
        for j = 0:n_bins+2 
            
            constante = -(1/(Nx))*(log_r(exp(1)*p_joint(k+1, j+1) / p_ref(j+1) ) );
            
            dp(k+1, j+1, 1) = constante.*sum(spline_cur(:,k+1).*spline_ref_der(:,j+1).*cte_r(:,1));
            dp(k+1, j+1, 2) = constante.*sum(spline_cur(:,k+1).*spline_ref_der(:,j+1).*cte_r(:,2));
            dp(k+1, j+1, 3) = constante.*sum(spline_cur(:,k+1).*spline_ref_der(:,j+1).*cte_r(:,3));
            dp(k+1, j+1, 4) = constante.*sum(spline_cur(:,k+1).*spline_ref_der(:,j+1).*cte_r(:,4));
            dp(k+1, j+1, 5) = constante.*sum(spline_cur(:,k+1).*spline_ref_der(:,j+1).*cte_r(:,5));
            dp(k+1, j+1, 6) = constante.*sum(spline_cur(:,k+1).*spline_ref_der(:,j+1).*cte_r(:,6));
            dp(k+1, j+1, 7) = constante.*sum(spline_cur(:,k+1).*spline_ref_der(:,j+1).*cte_r(:,7));
            dp(k+1, j+1, 8) = constante.*sum(spline_cur(:,k+1).*spline_ref_der(:,j+1).*cte_r(:,8));
            
            G(1) = G(1) + dp(k+1, j+1, 1);
            G(2) = G(2) + dp(k+1, j+1, 2);
            G(3) = G(3) + dp(k+1, j+1, 3);
            G(4) = G(4) + dp(k+1, j+1, 4);
            G(5) = G(5) + dp(k+1, j+1, 5);
            G(6) = G(6) + dp(k+1, j+1, 6);
            G(7) = G(7) + dp(k+1, j+1, 7);
            G(8) = G(8) + dp(k+1, j+1, 8);
        end
    end
    
    % Update
    p = inverse*G;
    
%     % Forwards compositional approach
%     p = -inverse*G;
        
    A = [p(5),p(3),p(1); p(4),-p(5)-p(6),p(2); p(7),p(8),p(6)];
    H = H*expm(A);    
    
    if sum(abs(p))< epsilon
        break;
    end
end 

iters

return