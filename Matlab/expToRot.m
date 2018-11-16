clear all;
root_dir = '../../Datasets/DFT/experimental_setup_moving_camera';
pose_fname=sprintf('%s/poseGroundtruth.txt', root_dir);
pose_data=importdata(pose_fname);

calib_fname=sprintf('%s/internalCalibrationMatrix.txt', root_dir);
calib_data=importdata(calib_fname);

n_data=size(pose_data, 2);
new_pose_data=zeros(12, n_data);
for data_id=1:n_data
curr_data=pose_data(1:3, data_id);
rot_mat=RotationMatrix(curr_data, 'exponentialMap'); 
new_pose_data(1:9, data_id) = reshape(rot_mat.matrix, [], 1);
new_pose_data(10:12, data_id) = pose_data(4:6, data_id);
fprintf('frame: %d:\n', data_id);
disp(rot_mat.matrix);
end
new_pose_fname=sprintf('%s/poseGroundtruth_new.txt', root_dir);
new_calib_fname=sprintf('%s/internalCalibrationMatrix_new.txt', root_dir);

dlmwrite(new_pose_fname,new_pose_data, 'delimiter','\t', 'precision','%12.8f');
dlmwrite(new_calib_fname,calib_data, 'delimiter','\t', 'precision','%12.8f');
