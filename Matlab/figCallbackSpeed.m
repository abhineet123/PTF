function figCallbackSpeed( h, e)
speed_factor = evalin('base', 'speed_factor');
slider_val = get(h,'Value');

if slider_val>speed_factor
    speed_factor=speed_factor+1;
elseif slider_val<speed_factor
    speed_factor=speed_factor-1;
    if speed_factor<1
        speed_factor=1;
    end
end  

%fprintf('Setting speed_factor to: %d\n', speed_factor);
assignin('base', 'speed_factor', speed_factor);
end

