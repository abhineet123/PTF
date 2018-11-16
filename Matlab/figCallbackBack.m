function figCallbackBack( h, e,  h_pause)
frame_id = evalin('base', 'frame_id');
file_start_id = evalin('base', 'file_start_id');
prev_id=frame_id-1;
if prev_id<file_start_id
    prev_id=file_start_id;
end
assignin('base', 'frame_id', prev_id);

assignin('base', 'pause_exec', 1);
set(h_pause, 'String', 'Resume');
end

