function [x, y] = read_corners(gt)
% This function reads corners from a cell array into two float arrays, 
% x and y
x = [gt{2}(1) gt{4}(1) gt{6}(1) gt{8}(1)];
y = [gt{3}(1) gt{5}(1) gt{7}(1) gt{9}(1)];