function figCallbackRot( h, e )
button_state = get(h,'Value');
if button_state == get(h,'Max')
	assignin('base', 'rotate_surf', 1);
elseif button_state == get(h,'Min')
	assignin('base', 'rotate_surf', 0);
end
end

