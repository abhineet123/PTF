%% Specify the directory containing the .pgm/.jpg files in dirname variable
%% Saves the images in a uint8 array called 'data'.
function data = load_data(dirname, filetype)
% dirname: Path of the data
% filetype: file extention

% Sort the image files with the specified extention
if nargin == 1 
    filetype = 'jpg';
end
ext = strcat('*.', filetype);
filenames = dir([dirname ext]);
filenames = sort({filenames.name});

% read the first to see how large it should be
im = imread([dirname filenames{1}]);
data = repmat(uint8(0),[size(im,1) size(im,2) length(filenames)]);
if size(im,3) == 1
    for ii = 1:length(filenames)
        data(:,:,ii) = imread([dirname filenames{ii}]);
    end
else
    for ii = 1:length(filenames)
        % if color, convert to grayscale
        data(:,:,ii) = rgb2gray(imread([dirname filenames{ii}]));
    end
end
