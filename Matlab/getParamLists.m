tracker_types = {
    'gt',...%0
    'esm',...%1
    'ic',...%2
    'nnic',...%3
    'pf',...%4
    'pw',...%5
    'ppw'...%6
    };
filter_types = {
    'none',...%0
    'gauss',...%1
    'box',...%2
    'norm_box',...%3
    'bilateral',...%4
    'median',...%5
    'gabor',...%6
    'sobel',...%7
    'scharr',...%8
    'LoG',...%9
    'DoG',...%10
    'laplacian',...%11
    'canny'%12
    };
inc_types = {'fc',...%0
    'ic',...%1
    'fa',...%2
    'ia'...%3
    };
pw_opt_types = {
    'pre',...%0
    'post',...%1
    'ind'%2
    };
error_types = {
    'mcd',...%0
    'cl',...%1
    'jaccard'%2
    };
grid_types = {
    'trans',...%0
    'rs',...%1
    'shear',...%2
    'proj',...%3
    'rtx',...%4
    'rty',...%5
    'stx',...%6
    'sty',...%7
    'trans2'%8
    };
appearance_models = {
    'ssd',...%0
    'scv',...%1
    'ncc'...%2
    'mi'...%3
    'ccre',...%4
    'hssd',...%5
    'jht',...%6
    'mi_old',...%7
    'ncc2',...%8
    'scv2',...%9
    'mi2',...%10
    'mssd',...%11
    'bmssd',...%12
    'bmi',...%13
    'crv',...%14
    'fkld',...%15
    'ikld',...%16
    'mkld',...%17
    'chis',...%18
    'ssim',...%19
    'fmaps'...%20
    };

datasets;
