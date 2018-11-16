root_dir='../C++/MTF/log';
am = 'mi';
show_cum_hist = 0;
if show_cum_hist
    wts_file=sprintf('%s/PF/%s/pf_particle_cum_wts.txt', root_dir, am);
else
    wts_file=sprintf('%s/PF/%s/pf_particle_wts.txt', root_dir, am);
end
wts_data=importdata(wts_file);
n_data=size(wts_data, 1);
n_particles=size(wts_data, 2);
x_data=1:n_particles;
wts_fig=figure;
set(wts_fig, 'Position', [0, 50, 1600, 900]);
for data_id=1:n_data
    set(wts_fig, 'Name', sprintf('frame %d', data_id));
    plot(x_data, wts_data(data_id, :));
    grid on;
    drawnow;
    pause(0.15);
end
