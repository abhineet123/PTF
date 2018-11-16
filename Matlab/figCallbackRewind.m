function figCallbackRewind( h, e, h_pause )
frame_id_diff = evalin('base', 'frame_id_diff');
assignin('base', 'frame_id_diff', -frame_id_diff);
frame_id_diff = evalin('base', 'frame_id_diff');
if frame_id_diff==-1
    set(h, 'String', 'Forward');
else
    set(h, 'String', 'Rewind');
end

assignin('base', 'pause_exec', 0);
set(h_pause, 'String', 'Pause');
end

