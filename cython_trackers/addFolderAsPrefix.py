import os
__author__ = 'Tommy'
root_dir='C:/Videos'

sub_dirs=[x[0] for x in os.walk(root_dir)]
del sub_dirs[0]
sub_dirs=[dir for dir in sub_dirs if not '#Misc' in dir]
sub_dirs=[dir for dir in sub_dirs if not 'Logitech' in dir]

print 'sub_dirs: ', sub_dirs
# n_dirs=len(sub_dirs)
for dir_path in sub_dirs:
    file_list=os.listdir(dir_path)
    dir=os.path.basename(dir_path)
    print 'dir: ', dir
    print 'file_list: ', file_list
    for file in file_list:
        file_parts=file.split('-')

        old_fname='{:s}/{:s}'.format(dir_path, file)
        new_fname='{:s}/{:s}-{:s}'.format(dir_path, dir, file_parts[-1])
        print 'Renaming {:s} to {:s}'.format(old_fname, new_fname)
        os.rename(old_fname, new_fname)



