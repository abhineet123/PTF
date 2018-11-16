clear all;
speed_ids=1:10;
speeds=[418.1630953, 734.8676074, 772.703762, 1064.440648, 387.2504594, 715.6162672, 155.2997269, 48.17332743, 181.2496618, 108.3014714]';
speeds=horzcat(speeds, zeros(10, 1));
set(0,'DefaultAxesFontName', 'Times New Roman');
set(0,'DefaultAxesFontSize', 14);
figure, hold on, grid on, title('Average speeds for 2DOF trackers');
bar(speed_ids, speeds);
ax = gca;
set(ax,'XTickLabel',{'FCLK', 'ICLK', 'FALK', 'IALK', 'ESM', 'AESM', 'NNIC', 'TLD', 'KCF', 'DSST'})
xlabel('search methods/trackers');
ylabel('Average FPS');