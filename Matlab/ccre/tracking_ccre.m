%   File: 'tracking_ccre.m'
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

function [H Warped_original] = tracking_ccre(ICur_gray, H, inverse, Template, size_bin, n_bins, Nx, size_template_x, size_template_y, cte_t, spline_ref, epsilon, max_iters)

% Runs for a maximum number of 'maxIters' iterations
for iters = 1:max_iters
    
    % Computing warped image and gradient
    Warped_original = warp(double(ICur_gray), H, size_template_x*2, size_template_y*2);
    Warped = Warped_original(:)./size_bin;
        
    % Computes joint distributions
    p_joint = computes(Warped', Template', Nx, n_bins);
    
    % Computes cumulative joint distribution
    p_zao = zeros(size(p_joint));
    
    for l = 1:n_bins+3
        for k = 1:n_bins+3
            p_zao(l, k) = sum(p_joint(l, k:end));
        end
    end
    
    soma_pzao = sum(p_zao);
    
    p_ref = sum(p_joint);
    p_cur = sum(p_joint');
    
    % Initialization
    G = zeros(8, 1);
    dp = zeros(n_bins+3, n_bins+3, 8);
    
    % Computes kernel values for every pixel in image
    spline_cur = computes_gradient(Warped, Nx, n_bins);
            
    % Populates g
    for k = 0:n_bins+2 
        for j = 0:n_bins+2 
            
            cte_part = (1/(Nx))*( log_r( p_zao(k+1,j+1)*inv_prob(p_cur(k+1)*soma_pzao(j+1))));
                    
            dp(k+1, j+1, 1) = cte_part.*sum(spline_cur(:,k+1).*spline_ref(:,j+1).*cte_t(:,1));
            dp(k+1, j+1, 2) = cte_part.*sum(spline_cur(:,k+1).*spline_ref(:,j+1).*cte_t(:,2));
            dp(k+1, j+1, 3) = cte_part.*sum(spline_cur(:,k+1).*spline_ref(:,j+1).*cte_t(:,3));
            dp(k+1, j+1, 4) = cte_part.*sum(spline_cur(:,k+1).*spline_ref(:,j+1).*cte_t(:,4));
            dp(k+1, j+1, 5) = cte_part.*sum(spline_cur(:,k+1).*spline_ref(:,j+1).*cte_t(:,5));
            dp(k+1, j+1, 6) = cte_part.*sum(spline_cur(:,k+1).*spline_ref(:,j+1).*cte_t(:,6));
            dp(k+1, j+1, 7) = cte_part.*sum(spline_cur(:,k+1).*spline_ref(:,j+1).*cte_t(:,7));
            dp(k+1, j+1, 8) = cte_part.*sum(spline_cur(:,k+1).*spline_ref(:,j+1).*cte_t(:,8));
            
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
    d = inverse*G;
    
    A = [d(5),d(3),d(1); d(4),-d(5)-d(6),d(2); d(7),d(8),d(6)];
    H = H*expm(A);      
    
    if sum(abs(d))< epsilon
        break;
    end
end

iters

return