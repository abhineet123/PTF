function figCallbackSpeedInc( h, e)
speed_factor = evalin('base', 'speed_factor');
speed_factor=speed_factor+1;
assignin('base', 'speed_factor', speed_factor);
set(h, 'String', sprintf('++%dx', speed_factor));
end

