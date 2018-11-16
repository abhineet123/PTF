%% Specify the directory containing the ground truth in dirname variable
%% Saves the labels in a cell array.
function label = load_gt(dirname)
% dirname: Path of the data

fileID = fopen([dirname(1:end-1) '.txt']);
label = textscan(fileID,'%s %f %f %f %f %f %f %f %f','HeaderLines',1);
fclose(fileID);
