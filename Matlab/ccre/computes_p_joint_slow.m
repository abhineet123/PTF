%   File: 'computes_p_joint_slow.m'
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

% Computing a vector that contains kernel values for each pixel on image
spline_values_cur = zeros(Nx, n_bins+3);
spline_values_ref = zeros(Nx, n_bins+3);

for i = 1:n_bins+3
    spline_values_ref(:,i) = comp_spline(i - 2 - Template);
    spline_values_cur(:,i) = comp_spline(i - 2 - Warped);
end

% Computing joint distribution
for i = 1:Nx
    
    cur = Warped(i);
    ref = Template(i);
    
    bin_cur = floor(cur); 
    bin_ref = floor(ref);
    
    % a e i m 
    % b f j n
    % c g k o 
    % d h l p
    
    % a
    p_joint(bin_cur+1, bin_ref+1) = p_joint(bin_cur+1, bin_ref+1) + spline_values_cur(i, bin_cur+1)*spline_values_ref(i, bin_ref+1);
    
    % b
    p_joint(bin_cur+2, bin_ref+1) = p_joint(bin_cur+2, bin_ref+1) + spline_values_cur(i, bin_cur+2)*spline_values_ref(i, bin_ref+1);
    
    % c
    p_joint(bin_cur+3, bin_ref+1) = p_joint(bin_cur+3, bin_ref+1) + spline_values_cur(i, bin_cur+3)*spline_values_ref(i, bin_ref+1); 
    
    % d
    p_joint(bin_cur+4, bin_ref+1) = p_joint(bin_cur+4, bin_ref+1) + spline_values_cur(i, bin_cur+4)*spline_values_ref(i, bin_ref+1); 


    % e
    p_joint(bin_cur+1, bin_ref+2) = p_joint(bin_cur+1, bin_ref+2) + spline_values_cur(i, bin_cur+1)*spline_values_ref(i, bin_ref+2);
    
    % f
    p_joint(bin_cur+2, bin_ref+2) = p_joint(bin_cur+2, bin_ref+2) + spline_values_cur(i, bin_cur+2)*spline_values_ref(i, bin_ref+2);
    
    % g
    p_joint(bin_cur+3, bin_ref+2) = p_joint(bin_cur+3, bin_ref+2) + spline_values_cur(i, bin_cur+3)*spline_values_ref(i, bin_ref+2); 
    
    % h
    p_joint(bin_cur+4, bin_ref+2) = p_joint(bin_cur+4, bin_ref+2) + spline_values_cur(i, bin_cur+4)*spline_values_ref(i, bin_ref+2); 


        % i
    p_joint(bin_cur+1, bin_ref+3) = p_joint(bin_cur+1, bin_ref+3) + spline_values_cur(i, bin_cur+1)*spline_values_ref(i, bin_ref+3);
    
    % j
    p_joint(bin_cur+2, bin_ref+3) = p_joint(bin_cur+2, bin_ref+3) + spline_values_cur(i, bin_cur+2)*spline_values_ref(i, bin_ref+3);
    
    % k
    p_joint(bin_cur+3, bin_ref+3) = p_joint(bin_cur+3, bin_ref+3) + spline_values_cur(i, bin_cur+3)*spline_values_ref(i, bin_ref+3); 
    
    % l
    p_joint(bin_cur+4, bin_ref+3) = p_joint(bin_cur+4, bin_ref+3) + spline_values_cur(i, bin_cur+4)*spline_values_ref(i, bin_ref+3); 


    % m
    p_joint(bin_cur+1, bin_ref+4) = p_joint(bin_cur+1, bin_ref+4) + spline_values_cur(i, bin_cur+1)*spline_values_ref(i, bin_ref+4);
    
    % n
    p_joint(bin_cur+2, bin_ref+4) = p_joint(bin_cur+2, bin_ref+4) + spline_values_cur(i, bin_cur+2)*spline_values_ref(i, bin_ref+4);
    
    % o
    p_joint(bin_cur+3, bin_ref+4) = p_joint(bin_cur+3, bin_ref+4) + spline_values_cur(i, bin_cur+3)*spline_values_ref(i, bin_ref+4); 
    
    % p
    p_joint(bin_cur+4, bin_ref+4) = p_joint(bin_cur+4, bin_ref+4) + spline_values_cur(i, bin_cur+4)*spline_values_ref(i, bin_ref+4); 

end

% Normalizing
p_joint = p_joint/Nx;
