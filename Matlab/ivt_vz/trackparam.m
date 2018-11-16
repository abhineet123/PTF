% script: trackparam.m
%     loads data and initializes variables
%

% Copyright (C) Jongwoo Lim and David Ross.
% All rights reserved.

% DESCRIPTION OF OPTIONS:
%
% Following is a description of the options you can adjust for
% tracking, each proceeded by its default value.  For a new sequence
% you will certainly have to change p.  To set the other options,
% first try using the values given for one of the demonstration
% sequences, and change parameters as necessary.
%
% p = [px, py, sx, sy, theta]; The location of the target in the first
% frame.
% px and py are th coordinates of the centre of the box
% sx and sy are the size of the box in the x (width) and y (height)
%   dimensions, before rotation
% theta is the rotation angle of the box
%
% 'numsample',400,   The number of samples used in the condensation
% algorithm/particle filter.  Increasing this will likely improve the
% results, but make the tracker slower.
%
% 'condenssig',0.01,  The standard deviation of the observation likelihood.
%
% 'ff',1, The forgetting factor, as described in the paper.  When
% doing the incremental update, 1 means remember all past data, and 0
% means remeber none of it.
%
% 'batchsize',5, How often to update the eigenbasis.  We've used this
% value (update every 5th frame) fairly consistently, so it most
% likely won't need to be changed.  A smaller batchsize means more
% frequent updates, making it quicker to model changes in appearance,
% but also a little more prone to drift, and require more computation.
%
% 'affsig',[4,4,.02,.02,.005,.001]  These are the standard deviations of
% the dynamics distribution, that is how much we expect the target
% object might move from one frame to the next.  The meaning of each
% number is as follows:
%
%    affsig(1) = x translation (pixels, mean is 0)
%    affsig(2) = y translation (pixels, mean is 0)
%    affsig(3) = rotation angle (radians, mean is 0)
%    affsig(4) = x scaling (pixels, mean is 1)
%    affsig(5) = y scaling (pixels, mean is 1)
%    affsig(6) = scaling angle (radians, mean is 0)
%
% OTHER OPTIONS THAT COULD BE SET HERE:
%
% 'tmplsize', [32,32] The resolution at which the tracking window is
% sampled, in this case 32 pixels by 32 pixels.  If your initial
% window (given by p) is very large you may need to increase this.
%
% 'maxbasis', 16 The number of basis vectors to keep in the learned
% apperance model.

% Change 'title' to choose the sequence you wish to run.  If you set
% title to 'dudek', for example, then it expects to find a file called 
% dudek.mat in the current directory.
%
% Setting dump_frames to true will cause all of the tracking results
% to be written out as .png images in the subdirectory ./dump/.  Make
% sure this directory has already been created.

clear; close all;
title = 'tmt'; %'ming-hsuan_light'; % only matters for original ivt data

%data_dir = '/usr/data/vzhang/data/'; % for loading the original ivt dataset on lubicon
%data_dir = '/usr/data2/datasets/tracking/ivt/'; % for loading the original ivt dataset at home
%data_dir = '/usr/data/Datasets/TMT/nl_bookIII_s3/'; % for loading other dataset

% this is equivalent to source_id in MTF, referring to 
% the id of a sequence
 
addpath('../');
getParamLists;

root_dir = '../../../Datasets';
actor_id = 0;
seq_id = 3; 

actor = actors{actor_id+1};
seq_name = sequences{actor_id + 1}{seq_id + 1};

data_dir = [root_dir '/' actor '/' seq_name '/'];
full_title = [data_dir title];
dump_frames = 0;
load_raw = 1; % true if loading raw images rather than mat files
online = 1; % true if want the user to specify the template
read_from_gt = 1; % true if reading template from ground truth
save_result = 1; % true if wants to save the tracking results.
show_result = 1;

if ~online
    switch (title)
        case 'dudek';  p = [188,192,110,130,-0.08];
            opt = struct('numsample',600, 'condenssig',0.25, 'ff',1, ...
                'batchsize',5, 'affsig',[9,9,.05,.05,.005,.001]);
            % Use the following set of parameters for the ground truth experiment.
            % It's much slower, but more accuracte.
            %case 'dudek';  p = [188,192,110,130,-0.08];
            %     opt = struct('numsample',4000, 'condenssig',0.25, 'ff',0.99, ...
            %                 'batchsize',5, 'affsig',[11,9,.05,.05,0,0], ...
            %                 'errfunc','');
        case 'davidin300';  p = [160 106 62 78 -0.02];
            opt = struct('numsample',600, 'condenssig',0.75, 'ff',.99, ...
                'batchsize',5, 'affsig',[5,5,.01,.02,.002,.001]);
        case 'sylv';  p = [145 81 53 53 -0.2];
            opt = struct('numsample',600, 'condenssig',0.75, 'ff',.95, ...
                'batchsize',5, 'affsig',[7,7,.01,.02,.002,.001]);
        case 'trellis70';  p = [200 100 45 49 0];
            opt = struct('numsample',600, 'condenssig',0.2, 'ff',.95, ...
                'batchsize',5, 'affsig',[4,4,.01,.01,.002,.001]);
        case 'fish';  p = [165 102 62 80 0];
            opt = struct('numsample',600, 'condenssig',0.2, 'ff',1, ...
                'batchsize',5, 'affsig',[7,7,.01,.01,.002,.001]);
            %case 'toycan';  p = [137 113 30 62 0];
            %    opt = struct('numsample',600, 'condenssig',0.2, 'ff',1, ...
            %                 'batchsize',5, 'affsig',[7,7,.01,.01,.002,.001]);
        case 'car4';  p = [245 180 200 150 0];
            opt = struct('numsample',600, 'condenssig',0.2, 'ff',1, ...
                'batchsize',5, 'affsig',[5,5,.025,.01,.002,.001]);
        case 'car11';  p = [89 140 30 25 0];
            opt = struct('numsample',600, 'condenssig',0.2, 'ff',1, ...
                'batchsize',5, 'affsig',[5,5,.01,.01,.001,.001]);
        case 'mushiake'; p = [172 145 60 60 0];
            opt = struct('numsample',600, 'condenssig',0.2, 'ff',1, ...
                'batchsize',5, 'affsig',[10, 10, .01, .01, .002, .001]);
            %case 'dudekgt';  p = [188,192,110,130,-0.08];
            %   opt = struct('numsample',4000, 'condenssig',1, 'ff',1, ...
            %                 'batchsize',5, 'affsig',[6,5,.05,.05,0,0], ...
            %                'errfunc','');
        otherwise;  error(['unknown title ' title]);
    end
    
else
    % pick the initial template
    data = load_data(data_dir);
    imshow(data(:,:,1));
    if read_from_gt
        % read_from_gt, Read from ground_truth
        [xi, yi] = read_corners(load_gt(data_dir));
        draw_gt(xi, yi);
        theta = atan2( yi(2)- yi(1), xi(2) - xi(1));
        width = sqrt((xi(2)-xi(1))^2 + (yi(2)-yi(1))^2);
        height = sqrt((xi(3)-xi(2))^2 + (yi(3)-yi(2))^2);
        p = [mean(xi), mean(yi), width, height, theta];
    else
        % Let User specify
        %rect = getrect; %[xmin, ymin, width, height]
        %p = [rect(1)+rect(3)/2, rect(2)+rect(4)/2, rect(3), rect(4), 0];
        % select polygon
        disp('specify the corners clockwise');
        [BW, xi, yi] = roipoly(data(:,:,1));
        stats = regionprops(BW,'all');
        theta = atan2( yi(2)- yi(1), xi(2) - xi(1));
        width = sqrt((xi(2)-xi(1))^2 + (yi(2)-yi(1))^2);
        height = sqrt((xi(3)-xi(2))^2 + (yi(3)-yi(2))^2);
        p = [stats.Centroid(1), stats.Centroid(2), width, height, theta];
    end
    opt = struct('numsample',600, 'condenssig',0.01, 'ff',1, ...
        'batchsize',5, 'affsig',[9,9,0.04,0.04,0.005,0.001]);
end

if save_result
    outdir = './'; % Save to current directory
    outfileID = fopen([outdir seq_name '_ivt66.txt'], 'wt');
    fprintf(outfileID, 'frame ulx uly urx ury lrx lry llx lly\n');
    if online % write the template coordinates to the first frame
        fprintf(outfileID, 'frame00001.jpg\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n', ...
            xi(1), yi(1), xi(2), yi(2), xi(3), yi(3), xi(4), yi(4) );
    end
else
    outfileID = [];
end

% p = [px, py, sx, sy, theta]; The location of the target in the first
% frame.
% px and py are th coordinates of the centre of the box
% sx and sy are the size of the box in the x (width) and y (height)
%   dimensions, before rotation
% theta is the rotation angle of the box
if load_raw
    disp(['loading raw images, ' data_dir '...']);
    clear truepts;
    close;
elseif (~exist('datatitle') | ~strcmp(title,datatitle))
    if (exist('datatitle') & ~strcmp(title,datatitle))
        disp(['title does not match.. ' title ' : ' datatitle ', continue?']);
        pause;
    end
    disp(['loading ' title '...']);
    clear truepts;
    load([full_title '.mat'],'data','datatitle','truepts');
end

opt.dump = dump_frames;
if (opt.dump & exist('dump') ~= 7)
    error('dump directory does not exist.. turning dump option off..');
    opt.dump = 0;
end
