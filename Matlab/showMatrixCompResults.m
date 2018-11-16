times_xv=importdata('matrices/times_xv.txt');
times_cv=importdata('matrices/times_cv.txt');
times_eig=importdata('matrices/times_eig.txt');
times_arm=importdata('matrices/times_arm.txt');
times_mtl=importdata('matrices/times_mtl.txt');
times_np=importdata('matrices/times_np.txt');

div_factor=2.0;

xv_col='r';
cv_col='g';
eig_col='b';
arm_col='k';
mtl_col='c';
np_col='m';
plot_legend={'OpenCV', 'Eigen', 'Armadillo', 'MTL', 'Numpy', 'Xvision'};
plot_markers = 's^od*xv>';
line_width=1.5;
inv_id=1;
tr_id=2;
prod_id=3;
det_id=4;
mat_count=size(times_xv, 1);
plot_x=1:mat_count;
plot_x=plot_x*5;

prod_fig=figure;
title('Matrix Product Times');
set(prod_fig, 'Name', 'Product');
xlabel('Matrix Size');
ylabel('Time (secs)');
hold on;
grid on;
plot(plot_x, times_cv(:, prod_id), 'Color', cv_col, 'Marker', plot_markers(1), 'LineWidth', line_width, 'MarkerFaceColor', cv_col);
plot(plot_x, times_eig(:, prod_id), 'Color', eig_col, 'Marker', plot_markers(2), 'LineWidth', line_width, 'MarkerFaceColor', eig_col);
plot(plot_x, times_arm(:, prod_id), 'Color', arm_col, 'Marker', plot_markers(3), 'LineWidth', line_width, 'MarkerFaceColor', arm_col);
plot(plot_x, times_mtl(:, prod_id), 'Color', mtl_col, 'Marker', plot_markers(4), 'LineWidth', line_width, 'MarkerFaceColor', mtl_col);
plot(plot_x, times_np(:, prod_id), 'Color', np_col, 'Marker', plot_markers(5), 'LineWidth', line_width, 'MarkerFaceColor', np_col);
plot(plot_x, times_xv(:, prod_id), 'Color', xv_col, 'Marker', plot_markers(6), 'LineWidth', line_width, 'MarkerFaceColor', xv_col);
legend(plot_legend);

inv_fig=figure;
title('Matrix Inverse Times');
set(inv_fig, 'Name', 'Inverse');
xlabel('Matrix Size');
ylabel('Time (secs)');
hold on;
grid on;
plot(plot_x, times_cv(:, inv_id)./div_factor, 'Color', cv_col, 'Marker', plot_markers(1), 'LineWidth', line_width,'MarkerFaceColor', cv_col);
plot(plot_x, times_eig(:, inv_id)./div_factor, 'Color', eig_col, 'Marker', plot_markers(2), 'LineWidth', line_width,'MarkerFaceColor', eig_col);
plot(plot_x, times_arm(:, inv_id)./div_factor, 'Color', arm_col, 'Marker', plot_markers(3), 'LineWidth', line_width,'MarkerFaceColor', arm_col);
plot(plot_x, times_mtl(:, inv_id)./div_factor, 'Color', mtl_col, 'Marker', plot_markers(4), 'LineWidth', line_width,'MarkerFaceColor', mtl_col);
plot(plot_x, times_np(:, inv_id)./div_factor, 'Color', np_col, 'Marker', plot_markers(5), 'LineWidth', line_width,'MarkerFaceColor', np_col);
plot(plot_x, times_xv(:, inv_id)./div_factor, 'Color', xv_col, 'Marker', plot_markers(6), 'LineWidth', line_width,'MarkerFaceColor', xv_col);
legend(plot_legend);

tr_fig=figure;
title('Matrix Transpose Times');
set(tr_fig, 'Name', 'Transpose');
xlabel('Matrix Size');
ylabel('Time (secs)');
hold on;
grid on;
plot(plot_x, times_cv(:, tr_id)./div_factor, 'Color', cv_col, 'Marker', plot_markers(1), 'LineWidth', line_width,'MarkerFaceColor', cv_col);
plot(plot_x, times_eig(:, tr_id)./div_factor, 'Color', eig_col, 'Marker', plot_markers(2), 'LineWidth', line_width,'MarkerFaceColor', eig_col);
plot(plot_x, times_arm(:, tr_id)./div_factor, 'Color', arm_col, 'Marker', plot_markers(3), 'LineWidth', line_width,'MarkerFaceColor', arm_col);
plot(plot_x, times_mtl(:, tr_id)./div_factor, 'Color', mtl_col, 'Marker', plot_markers(4), 'LineWidth', line_width,'MarkerFaceColor', mtl_col);
plot(plot_x, times_np(:, tr_id)./div_factor, 'Color', np_col, 'Marker', plot_markers(5), 'LineWidth', line_width,'MarkerFaceColor', np_col);
plot(plot_x, times_xv(:, tr_id)./div_factor, 'Color', xv_col, 'Marker', plot_markers(6), 'LineWidth', line_width,'MarkerFaceColor', xv_col);
legend(plot_legend);

det_fig=figure;
title('Matrix Determinant Times');
set(det_fig, 'Name', 'Determinant');
xlabel('Matrix Size');
ylabel('Time (secs)');
hold on;
grid on;
plot(plot_x, times_cv(:, det_id)./div_factor, 'Color', cv_col, 'Marker', plot_markers(1), 'LineWidth', line_width,'MarkerFaceColor', cv_col);
plot(plot_x, times_eig(:, det_id)./div_factor, 'Color', eig_col, 'Marker', plot_markers(2), 'LineWidth', line_width,'MarkerFaceColor', eig_col);
plot(plot_x, times_arm(:, det_id)./div_factor, 'Color', arm_col, 'Marker', plot_markers(3), 'LineWidth', line_width,'MarkerFaceColor', arm_col);
plot(plot_x, times_mtl(:, det_id)./div_factor, 'Color', mtl_col, 'Marker', plot_markers(4), 'LineWidth', line_width,'MarkerFaceColor', mtl_col);
plot(plot_x, times_np(:, det_id)./div_factor, 'Color', np_col, 'Marker', plot_markers(5), 'LineWidth', line_width,'MarkerFaceColor', np_col);
legend(plot_legend);


