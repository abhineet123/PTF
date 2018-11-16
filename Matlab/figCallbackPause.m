function figCallbackPause( h, e,  slider_speed )
%disp('here we are!\n');
pause_exec = evalin('base', 'pause_exec');
%fprintf('pause_exec before: %d\n', pause_exec);
assignin('base', 'pause_exec', 1-pause_exec)
pause_exec = evalin('base', 'pause_exec');
%fprintf('pause_exec after: %d\n', pause_exec);
if ~pause_exec
    set(h, 'String', 'Pause');
else
    set(h, 'String', 'Resume');
    assignin('base', 'speed_factor', 1);
    set(slider_speed, 'Value', 1);    
end
end

