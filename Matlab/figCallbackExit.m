function figCallbackExit( h, e )
end_exec=evalin('base', 'end_exec');
if end_exec
    set(h, 'String', 'Bye!');
    close all;
else
    assignin('base', 'end_exec', 1);
    set(h, 'String', 'Close');
end
end

