import os
import shutil

# src_folder = 'C:/Users/Tommy/Desktop/Done'
# dst_folder = 'C:/Users/Tommy/Desktop/New/abb'

src_folder = 'E:/UofA/Thesis/Code/TrackingFramework/C++/MTF/log/success_rates'
dst_folder = 'E:/UofA/Thesis/Code/TrackingFramework/C++/MTF/log/success_rates/TMT'

duplicate_folder = 'duplicates'

src_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
dst_files = [f for f in os.listdir(dst_folder) if os.path.isfile(os.path.join(dst_folder, f))]
print 'Found {:d} source files'.format(len(src_files))
print 'Found {:d} test files'.format(len(dst_files))

duplicate_files = set(src_files) & set(dst_files)

if len(duplicate_files) > 0:
    print 'Found {:d} duplicate files:\n'.format(len(duplicate_files)), duplicate_files
else:
    print 'Found no duplicate files'

if not os.path.exists(duplicate_folder):
    os.mkdir(duplicate_folder)

for file in duplicate_files:
    shutil.move(os.path.join(dst_folder, file), os.path.join(duplicate_folder, file))




