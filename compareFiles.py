import subprocess
import filecmp
import os
dir_1 = '/root/mdp_log'
dir_2 = '/root/mdp_tf_log'
filenames = [
    ['features.txt', 'templates/features.txt', 0],
    ['scores.txt', 'templates/scores.txt', 0],
    ['indices.txt', 'templates/indices.txt', 0],
    ['overlaps.txt', 'templates/overlaps.txt', 0],
    ['ratios.txt', 'templates/ratios.txt', 0],
    ['angles.txt', 'templates/angles.txt', 0],
    ['bb_overlaps.txt', 'templates/bb_overlaps.txt', 0],
    ['similarity.txt', 'templates/similarity.txt', 0],
    ['scores.txt', 'templates/scores.txt', 0],
    ['roi.txt', 'templates/roi.txt', 0],
    ['patterns.txt', 'templates/patterns.txt', 0],
    ['lk_out.txt', 'lkcv/lk_out.txt', 1],
]
for files in filenames:
    path_1  = '{:s}/{:s}'.format(dir_1, files[0])
    path_2 = '{:s}/{:s}'.format(dir_2, files[1])

    if not os.path.isfile(path_1):
        print '{:s} does not exist'.format(path_1)
        continue
    if not os.path.isfile(path_2):
        print '{:s} does not exist'.format(path_2)
        continue
    subprocess.call('dos2unix -q {:s}'.format(path_1), shell=True)
    subprocess.call('dos2unix -q {:s}'.format(path_2), shell=True)
    if files[2]:
        subprocess.call('sed -i -e \'s/NaN/nan/g\' {:s}'.format(path_1), shell=True)
	
    if not filecmp.cmp(path_1, path_2):
        print 'Files {:s} and {:s} are different'.format(path_1, path_2)
