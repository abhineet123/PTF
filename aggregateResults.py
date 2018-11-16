from Misc import *

filenames = [
    'fps_task_juice_tracker_ict_filter_',
    'fps_task_juice_tracker_nn_filter_',
    'fps_task_mugI_tracker_esm_filter_',
    'fps_task_mugI_tracker_ict_filter_',
    'fps_task_mugI_tracker_nn_filter_',
    'fps_task_mugII_tracker_esm_filter_',
    'fps_task_mugII_tracker_ict_filter_',
    'fps_task_mugII_tracker_nn_filter_',
    'fps_task_mugIII_tracker_esm_filter_',
    'fps_task_mugIII_tracker_ict_filter_',
    'fps_task_mugIII_tracker_nn_filter_',
    'error_task_bookI_tracker_esm_filter_',
    'error_task_bookI_tracker_ict_filter_',
    'error_task_bookI_tracker_nn_filter_',
    'error_task_bookII_tracker_esm_filter_',
    'error_task_bookII_tracker_ict_filter_',
    'error_task_bookII_tracker_nn_filter_',
    'error_task_bookIII_tracker_esm_filter_',
    'error_task_bookIII_tracker_ict_filter_',
    'error_task_bookIII_tracker_nn_filter_',
    'error_task_cereal_tracker_esm_filter_',
    'error_task_cereal_tracker_ict_filter_',
    'error_task_cereal_tracker_nn_filter_',
    'error_task_juice_tracker_esm_filter_',
    'error_task_juice_tracker_ict_filter_',
    'error_task_juice_tracker_nn_filter_',
    'error_task_mugI_tracker_esm_filter_',
    'error_task_mugI_tracker_ict_filter_',
    'error_task_mugI_tracker_nn_filter_',
    'error_task_mugII_tracker_esm_filter_',
    'error_task_mugII_tracker_ict_filter_',
    'error_task_mugII_tracker_nn_filter_',
    'error_task_mugIII_tracker_esm_filter_',
    'error_task_mugIII_tracker_ict_filter_',
    'error_task_mugIII_tracker_nn_filter_',
    'fps_task_bookI_tracker_esm_filter_',
    'fps_task_bookI_tracker_ict_filter_',
    'fps_task_bookI_tracker_nn_filter_',
    'fps_task_bookII_tracker_esm_filter_',
    'fps_task_bookII_tracker_ict_filter_',
    'fps_task_bookII_tracker_nn_filter_',
    'fps_task_bookIII_tracker_esm_filter_',
    'fps_task_bookIII_tracker_ict_filter_',
    'fps_task_bookIII_tracker_nn_filter_',
    'fps_task_cereal_tracker_esm_filter_',
    'fps_task_cereal_tracker_ict_filter_',
    'fps_task_cereal_tracker_nn_filter_',
    'fps_task_juice_tracker_esm_filter_'
]
#for filename in filenames:
#    aggregateDataFromFiles(filename, filename)
root_dir='G:/UofA/Thesis/Summer RA Report/Results/Features_nn'
# legend=['Grayscale', 'RGB', 'HSV', 'YCrCb', 'HLS', 'Lab']
# legend=['none', 'gabor', 'laplacian', 'sobel', 'scharr', 'canny', 'LoG', 'DoG']
# legend=['none', 'mean', 'majority', 'flatten']
# legend = ['none', 'mean', 'flatten']
# legend=['3x3', '5x5', '7x7', '9x9', '11x11', '13x13', '15x15']
# legend=['15$^\circ$', '30$^\circ$', '45$^\circ$', '60$^\circ$', '75$^\circ$', '90$^\circ$']
legend=['none', 'hoc']
# legend=['none', 'box', 'bilateral', 'gauss', 'median']
split=0
if split:
    keywords=['esm', 'ict', 'nn']
    # keywords=['ict', 'nn']
    # keywords=['RGB', 'HSV', 'YCrCb']
    splitFiles('list', keywords, root_dir=root_dir, plot=False)
else:
    title='NN'
    # xticks=['bookI', 'bookII', 'bookIII', 'cereal', 'juice', 'mugI', 'mugII', 'mugIII',
    #         'bus', 'highlighting', 'letter', 'newspaper']
    # filename='list_'+title.lower()
    filename='list'
    InteractivePlot(file=filename)
    # getPointPlot(file=filename, legend=None, xticks=None,
    #              root_dir=None,
    #              show_plot=True, title=None, use_sep_fig=True, plot_drift=False)
# getGroundTruthUpdates('G:/UofA/Thesis/#Code/Datasets/Human/nl_bookII_s3.txt')
