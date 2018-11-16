%   File: 'run_sim.m'
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

%clear all
%close all
%clc
%input_parameters;

% Initialization for first image
nb_image = nb_first_image;
loads_image;

% Selects reference image (Template)
size_template_x = size_template_x/2; % in the rest of the code, this variable will represent only half the template size ...
size_template_y = size_template_y/2;

Template = double(ICur_gray(pos(2)-size_template_y+1:pos(2)+size_template_y, pos(1)-size_template_x+1: pos(1)+size_template_x));

%figure(3)
%imshow(uint8(Template))
%title('Reference image')

% Initializing tracking parameters
H = [ 1 0 pos(1)-size_template_x; 0 1 pos(2)-size_template_y; 0 0 1];
size_bin = 256/(n_bins);
Nx = numel(Template);

% Pre-computing Hessian matrix and its inverse
pre_computations;
fprintf(out_fid, 'frame%05d.jpg\t%15.9f\t%15.9f\t%15.9f\t%15.9f\t%15.9f\t%15.9f\t%15.9f\t%15.9f\n',...
    nb_first_image,...
    pos(1)-size_template_x, pos(2)-size_template_y,...
    pos(1)+size_template_x, pos(2)-size_template_y,...
    pos(1)+size_template_x, pos(2)+size_template_y,...
    pos(1)-size_template_x, pos(2)+size_template_y);

% Runs on dataset
for nb_image = nb_first_image +  1 : nb_last_image
    
    % Loads image
    loads_image
    
    % Tracking
    [H Warped] = tracking_mi(ICur_gray, H, Hessian_inverse, Template, size_bin, n_bins, Nx, size_template_x, size_template_y, cte_t, spline_ref, spline_ref_der, maxIters, epsilon);
    
    % Display
%     figure(1) ; clf ;
%     imshow(uint8(ICur)); axis image ;
%     hold on
    
    T_pos = H*[0 2*size_template_x 2*size_template_x 0 0; 0 0 2*size_template_y  2*size_template_y 0; ones(1,5)];
    T_pos(1,:) = T_pos(1,:)./T_pos(3,:);
    T_pos(2,:) = T_pos(2,:)./T_pos(3,:);
    
%     plot(T_pos(1,:), T_pos(2,:),'g-','LineWidth',2,'MarkerSize',6)
%     drawnow
    fprintf(out_fid, 'frame%05d.jpg\t%15.9f\t%15.9f\t%15.9f\t%15.9f\t%15.9f\t%15.9f\t%15.9f\t%15.9f\n',...
        nb_image,...
        T_pos(1,1), T_pos(2,1),...
        T_pos(1,2), T_pos(2,2),...
        T_pos(1,3), T_pos(2,3),...
        T_pos(1,4), T_pos(2,4));
    
    
    
    %     figure(2)
    %     imshow(uint8(Warped));
    %     title('Current back-warped image');
    
    % Saving results into a file
    % print(figure(2), strcat( './sauvegarde/', 'warped_', number, '.png' ), '-dpng', '-r150');
    % print(figure(1), strcat( './sauvegarde/', 'cross_', number, '.png' ), '-dpng', '-r150');
end
fclose(out_fid);


