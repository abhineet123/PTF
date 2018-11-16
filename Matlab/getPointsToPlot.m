function [ x_vals, y_vals, n_iters, pt_cols, max_frame_id] = getPointsToPlot(tracker_hom_params,...
    max_iters, x_id, y_id)

max_frame_id=max(tracker_hom_params(:, 1));
min_frame_id=1;

fprintf('Getting points to plot with min_frame_id: %d and max_frame_id: %d \n',...
    min_frame_id, max_frame_id);

x_vals=zeros(max_frame_id, max_iters);
y_vals=zeros(max_frame_id, max_iters);
n_iters=zeros(max_frame_id, 1);
pt_cols=zeros(max_frame_id, max_iters, 3);
n_params=size(tracker_hom_params, 1);

hom_params_id=1;
while(tracker_hom_params(hom_params_id, 1)<min_frame_id+1)
    hom_params_id=hom_params_id+1;
end

for frame_id=min_frame_id:max_frame_id    
    n_iter=0;
    while(hom_params_id<=n_params && tracker_hom_params(hom_params_id, 1)==frame_id+1)
        n_iter=n_iter+1;
        x_vals(frame_id, n_iter)=tracker_hom_params(hom_params_id, x_id);
        y_vals(frame_id, n_iter)=tracker_hom_params(hom_params_id, y_id);      
        hom_params_id=hom_params_id+1;
    end
    %fprintf('Frame:\t%4d :: n_iter:\t%d\n', frame_id, n_iter);
    n_iters(frame_id)=n_iter;
    pt_cols(frame_id, 1, :)=[0, 1, 0];
    if n_iters(frame_id)>1
         pt_cols(frame_id,  n_iters(frame_id), :)=[1, 0, 0];
    end
    for i=2:n_iters(frame_id)-1
        pt_cols(frame_id, i, :)=[1-i/n_iters(frame_id), 1-i/n_iters(frame_id), 1-i/n_iters(frame_id)];
    end
    
end
end

