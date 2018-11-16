function drawbox(varargin)
% function drawbox(width,height, param, properties)
%                 ([width,height], param, properties)
%                 ([width,height], param, properties, outfid, fno)
%   param, properties are optional
%

%% Copyright (C) Jongwoo Lim and David Ross.
%% All rights reserved.


%----------------------------------------------------------
% Process the input.
%----------------------------------------------------------
if (length(varargin{1}) == 2)
  w = varargin{1}(1);
  h = varargin{1}(2);
  varargin(1) = [];
else
  [w,h] = deal(varargin{1:2});
  varargin(1:2) = [];
end

if (length(varargin) < 1 || any(length(varargin{1}) ~= 6))
  M = [0,1,0; 0,0,1];
else
  p = varargin{1};
  if (length(varargin) > 1 && strcmp(varargin{2},'geom'))
    p = affparam2mat(p);
    varargin(1:2) = [];
  else
    varargin(1) = [];
  end
  M = [p(1) p(3) p(4); p(2) p(5) p(6)];
end

%----------------------------------------------------------
% Draw the box.
%----------------------------------------------------------

%corners = [ 1,0,0; 1,w,0; 1,w,h; 1,0,h; 1,0,0 ]';
corners = [ 1,-w/2,-h/2; 1,w/2,-h/2; 1,w/2,h/2; 1,-w/2,h/2; 1,-w/2,-h/2 ]';
corners = M * corners;
% Save the corners to file
if length(varargin) > 4
    if varargin{end} ~= 0 && varargin{end} ~= 1 % do not write for frame 0 or 1
        fprintf(varargin{end-1}, 'frame%05d.jpg\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n', ...
            varargin{end}, corners(1,1), corners(2,1), corners(1,2), corners(2,2), corners(1,3), corners(2,3), corners(1,4), corners(2,4));
    end
    line(corners(1,:), corners(2,:), varargin{1:end-2});
else
    line(corners(1,:), corners(2,:), varargin{:});
% patch(corners(1,:), corners(2,:), 'y', ...
%       'FaceAlpha',0, 'LineWidth',1.5, varargin{:});
end

center = mean(corners(:,1:4),2);
hold_was_on = ishold; hold on;
if length(varargin) > 4
    plot(center(1),center(2),varargin{1:end-2});
else
    plot(center(1),center(2),varargin{:});
end
if (~hold_was_on) hold off; end
