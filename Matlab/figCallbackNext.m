function figCallbackNext( h, e, h_pause)
frame_id = evalin('base', 'frame_id');
file_end_id = evalin('base', 'file_end_id');
next_id=frame_id+1;
if next_id>file_end_id
    next_id=file_end_id;
end
assignin('base', 'frame_id', next_id);

assignin('base', 'pause_exec', 1);
set(h_pause, 'String', 'Resume');
end

