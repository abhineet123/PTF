%   File: 'input_parameters.m'
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


% Image dataset parameters
addpath '../'

getParamLists;

actor_id = 0;
seq_id = 4;

actor = actors{actor_id+1}
seq_name = sequences{actor_id + 1}{seq_id + 1}

gt_data=importdata(sprintf('../../../Datasets/%s/%s.txt',actor,seq_name));
gt_data=gt_data.data;
n_frames=size(gt_data, 1)
out_file=sprintf('../../C++/MTF_LIB/log/tracking_data/%s/iclk_mat_mi_8_1.txt',seq_name);
out_fid=fopen(out_file, 'w');
fprintf(out_fid, 'frame ulx	uly	urx	ury	lrx	lry	llx	lly\n');

path_to_images = sprintf('../../../Datasets/%s/%s/',actor,seq_name)
file_name = 'frame';
image_format = '.jpg';
length_number = 5; % size of number characters in image name
nb_first_image = 1;
nb_last_image = n_frames;

% Storage directory
path_to_save = './sauvegarde/';
file_name_save = 'tracked_';

% Position of reference image on first frame

mean_x=int32(mean(gt_data(1, [1, 3, 5, 7])))
mean_y=int32(mean(gt_data(1, [2, 4, 6, 8])))
pos = [double(mean_x)   double(mean_y)]

size_x=int32(((gt_data(1, 3)-gt_data(1, 1)) + (gt_data(1, 5)-gt_data(1, 7)))/2)
size_y=int32(((gt_data(1, 8)-gt_data(1, 2)) + (gt_data(1, 6)-gt_data(1, 4)))/2)

% Select reference image size (in pixels, MUST BE PAIR)
size_template_x=double(size_x)
size_template_y=double(size_y)

% Number of histogram bins
n_bins = 8;

% Maximum number of iterations
maxIters = 100;

% Threshold for breaking optimization loop
epsilon = 0.001;