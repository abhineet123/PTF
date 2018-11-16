import sys
import os
from Misc import getParamDict

params_dict = getParamDict()

db_root_dir = 'C:/Datasets'
seq_prefix = 'image'
seq_start_id = 1
actor_type = 1
actor_id = 4
start_id = 0
end_id = 1

arg_id = 1
if len(sys.argv) > arg_id:
    actor_id = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    start_id = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    end_id = int(sys.argv[arg_id])
    arg_id += 1
if len(sys.argv) > arg_id:
    db_root_dir = sys.argv[arg_id]
    arg_id += 1

print 'seq_prefix: {:s}'.format(seq_prefix)
print 'seq_start_id: {:d}'.format(seq_start_id)

if actor_type == 0:
    actors = params_dict['actors']
    sequences = params_dict['sequences']
else:
    actors = params_dict['mot_actors']
    sequences = params_dict['mot_sequences']

actor = actors[actor_id]

for seq_id in xrange(start_id, end_id+1):
    seq_name = sequences[actor][seq_id]
    # seq_name = 'hexagon_task_fast_right_2'
    if actor_type == 0:
        seq_root_dir = db_root_dir + '/' + actor + '/' + seq_name
    else:
        seq_root_dir = db_root_dir + '/' + actor + '/Images/' + seq_name

    src_file_names = [f for f in os.listdir(seq_root_dir) if os.path.isfile(os.path.join(seq_root_dir, f))]

    frame_id = seq_start_id
    file_count = 1
    n_files = len(src_file_names)
    print 'Renaming {:s} with {:d} files'.format(seq_name, n_files)
    for src_fname in src_file_names:
        filename, file_extension = os.path.splitext(src_fname)
        src_path = os.path.join(seq_root_dir, src_fname)
        dst_path = os.path.join(seq_root_dir, '{:s}{:06d}{:s}'.format(seq_prefix, frame_id, file_extension))
        while os.path.exists(dst_path):
            frame_id += 1
            dst_path = os.path.join(seq_root_dir, '{:s}_{:d}{:s}'.format(seq_prefix, frame_id, file_extension))
        os.rename(src_path, dst_path)
        frame_id += 1
        if file_count % 100 == 0 or file_count == n_files:
            print 'Done {:d}/{:d}'.format(file_count, n_files)
        file_count += 1
