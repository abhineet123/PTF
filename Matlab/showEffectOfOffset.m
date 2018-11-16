clear all;
close all;

offset = 100;
nbins = 100;
n = 10000;
n_sum = 1000;
n_sum_with_offset = n_sum + offset*n;
rand_vec = normrnd(0.5, 0.5, n, 1);
rand_vec_norm = rand_vec / sum(rand_vec);

[rand_vec_hist,~] = histcounts(rand_vec, nbins);

rand_vec_offset = (rand_vec_norm * n_sum) + offset;
rand_vec_offset_norm = rand_vec_offset / sum(rand_vec_offset);
[rand_vec_hist,edges] = histcounts(rand_vec, nbins);

ax_limits = [1 n 0 0.2];
plot_x = 1 : n;
figure, plot(plot_x, rand_vec_norm), axis(ax_limits);
figure, plot(plot_x, rand_vec_offset_norm), axis(ax_limits);

