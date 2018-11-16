function [slider_speed, txt_speed, pb_pause, pb_back, pb_next, pb_rewind, tb_rot, pb_exit] = createWidgets(surf_fig,...
    x_pos, y_pos, pb_width, pb_height, slider_width, slider_height, txt_height)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
slider_speed = uicontrol(surf_fig,'Style','slider','Min',1,'Max', 10, 'SliderStep', [0.1 0.1],...
    'Value',1, 'Position',[(pb_width-slider_width)/2 y_pos slider_width slider_height], 'Callback', @(h, e) figCallbackSpeed( h, e ));
y_pos=y_pos-txt_height;
txt_speed = uicontrol(surf_fig, 'Style','text','String','Speed',...
    'Position',[x_pos y_pos pb_width txt_height]);

y_pos=y_pos-pb_height;
pb_pause = uicontrol(surf_fig,'Style','pushbutton','String','Pause','Value',0,...
    'Position',[x_pos y_pos pb_width pb_height], 'Callback', @(h, e) figCallbackPause( h, e, slider_speed ));

y_pos=y_pos-pb_height;
pb_back = uicontrol(surf_fig,'Style','pushbutton','String','Back','Value',0,...
    'Position',[x_pos y_pos pb_width pb_height], 'Callback', @(h, e) figCallbackBack( h, e, pb_pause));

y_pos=y_pos-pb_height;
pb_next = uicontrol(surf_fig,'Style','pushbutton','String','Next','Value',0,...
    'Position',[x_pos y_pos pb_width pb_height], 'Callback', @(h, e) figCallbackNext( h, e, pb_pause));
y_pos=y_pos-pb_height;
pb_rewind = uicontrol(surf_fig,'Style','pushbutton','String','Rewind','Value',0,...
    'Position',[x_pos y_pos pb_width pb_height], 'Callback', @(h, e) figCallbackRewind( h, e, pb_pause));

rotate_surf=evalin('base', 'rotate_surf');
y_pos=y_pos-pb_height;
tb_rot = uicontrol(surf_fig,'Style','togglebutton','String','Rotate','Value',rotate_surf,'Max', 1, 'Min', 0,...
     'Position',[x_pos y_pos pb_width pb_height], 'Callback', @(h, e) figCallbackRot( h, e ));
 
% y_pos=y_pos-pb_height;
% uicontrol(surf_fig,'Style','pushbutton','String','Close',...
%      'Position',[x_pos y_pos pb_width pb_height], 'Callback', @(h, e) evalin('base', 'end_exec=1, close all;'));
 
y_pos=y_pos-pb_height;
pb_exit = uicontrol(surf_fig,'Style','pushbutton','String','Stop','Value',0,...
    'Position',[x_pos y_pos pb_width pb_height], 'Callback', @(h, e) figCallbackExit( h, e));
end

