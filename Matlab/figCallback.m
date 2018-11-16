function figCallback( h, e )
disp('here we are!\n');

pause_exec = evalin('base', 'pause_exec');
fprintf('pause_exec before: %d\n', pause_exec);
assignin('base', 'pause_exec', 1-pause_exec)
pause_exec = evalin('base', 'pause_exec');
fprintf('pause_exec after: %d\n', pause_exec);

% if e.Key=='p' || e.Key=='P'
%     pause_exec=1;
% end

end

