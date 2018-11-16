function getPointPlot(file, root_dir)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin<1
    file='list';
else if  nargin<2
        root_dir='Results';
    end
end
file_path=sprintf('%s/%s.txt', root_dir, file);
list_data=importdata(file_path);
plot_fname=list_data(1);
filenames=list_data(2:end);

line_styles={'-'  '--'  '-.'  ':'};
markers={ '+'  'o'  'D'  'x'  's'  'p'  '*'  'v'  '^'};
colors={'r'  'g'  'b' 'c' 'm' 'y' 'k'};

hold on


fig_sr=figure(1);
% grid on
% grid minor
title('Success Rate')
axis_sr=axes('parent',fig_sr);
%plt.legend(filenames)


fig_fps=figure(2);
% grid on
% grid minor
title('Average FPS')
axis_fps=axes('parent',fig_fps);
%plt.legend(filenames)

fig_drift=figure(3);

title('Average Drift')
axis_drift=axes('parent',fig_drift);
%plt.legend(filenames)

linestyle_id = 1;
marker_id = 1;
color_id = 1;
no_of_files=size(filenames, 1);

colors_size=size(colors, 2);
markers_size=size(markers, 2);
line_styles_size=size(line_styles, 2);

for i=1:no_of_files
    plot_fname=sprintf('%s/%s.txt', root_dir, filenames{i});
    if exist(plot_fname, 'file')
        plot_data=importdata(plot_fname);        
        parameters=plot_data.textdata(2:end, 1);
        header=strtrim(plot_data.textdata(1, 2:end));
        plot_data=plot_data.data;
    else
        fprintf('File %s does not exist\n', plot_fname)
    end
    
    success_rate=plot_data(:,1);
    avg_fps=plot_data(:, 2);
    avg_drift=plot_data(:, 3);
    
    data_count=size(success_rate, 1);   
    

    x=0:data_count-1;
    figure(1), hold
    plot(x, success_rate, 'LineStyle',line_styles{linestyle_id}, 'Color', colors{color_id}, 'Marker',markers{marker_id})
%     hold on
    figure(2), hold
    plot(x, avg_fps, 'LineStyle',line_styles{linestyle_id}, 'Color', colors{color_id}, 'Marker',markers{marker_id})
%     hold on
    figure(3), hold    
    plot(x, avg_drift, 'LineStyle',line_styles{linestyle_id}, 'Color', colors{color_id}, 'Marker',markers{marker_id})
%     hold on
    color_id = mod(color_id + 1, colors_size)+1;
    marker_id = mod(marker_id + 1, markers_size)+1;
%     linestyle_id = mod(linestyle_id + 1, line_styles_size)+1; 
end

grid on
grid minor
    
% filenames=filenames    
legend(axis_sr, filenames, 'Interpreter', 'none') 
legend(axis_fps, filenames, 'Interpreter', 'none') 
legend(axis_drift, filenames, 'Interpreter', 'none') 

% 
% %annotate_text=''
% %print 'annotate_text_list:\n'  annotate_text_list
% %for i in xrange(len(annotate_text_list)):
% %    annotate_text=annotate_text+str(i)+': '+annotate_text_list[i]+'\n'
% %
% %print 'annotate_text=\n'  annotate_text
% % saving success rate plot
% if use_sep_fig:
%     plt.figure(0)
% else:
%     plt.subplot(311)
% plt.legend(filenames  prop=fontP)
% plt.grid(True)
% %plt.figtext(0.01 0.01  annotate_text  fontsize=9)
% if use_sep_fig and plot_fname is not None:
%     plt.savefig('Results/'+plot_fname+'_success_rate'  dpi=96  ext='png')
% % saving fps plot
% if use_sep_fig:
%     plt.figure(1)
% else:
%     plt.subplot(312)
% plt.legend(filenames  prop=fontP)
% plt.grid(True)
% if use_sep_fig and plot_fname is not None:
%     plt.savefig('Results/'+plot_fname+'_avg_fps'  dpi=96  ext='png')
% % saving drift plot
% if use_sep_fig:
%     plt.figure(2)
% else:
%     plt.subplot(313)
% plt.legend(filenames  prop=fontP)
% plt.grid(True)
% if use_sep_fig and plot_fname is not None:
%     plt.savefig('Results/'+plot_fname+'_avg_drft'  dpi=96  ext='png')
% 
% % saving combined plot
% if not use_sep_fig and plot_fname is not None:
%     plt.savefig('Results/'+plot_fname  dpi=96  ext='png')
% 
% if show_plot:
%     plt.show()
% end

end

