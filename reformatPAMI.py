from Misc import getParamDict
from Misc import readDistGridParams
import os

if __name__ == '__main__':

    params_dict = getParamDict()
    param_ids = readDistGridParams()
    pause_seq = 0
    gt_col = (0, 0, 255)
    gt_thickness = 1
    sequences = params_dict['sequences']
    db_root_dir = '../Datasets'
    # img_name_fmt='img%03d.jpg'
    img_name_fmt = 'frame%05d.jpg'
    opt_gt_ssm = '0'
    use_opt_gt = 0

    actor = 'PAMI'
    sequences = sequences[actor]
    sequences = ['phone']
    n_seq = len(sequences)

    for seq_id in xrange(n_seq):
        seq_name = sequences[seq_id]
        # seq_name = 'nl_mugII_s1'

        print 'seq_name: ', seq_name

        src_dir = db_root_dir + '/' + actor + '/' + seq_name + '_gt'
        dst_fname = db_root_dir + '/' + actor + '/' + seq_name + '.txt'

        if not os.path.isdir(src_dir):
            print 'The source ground truth folder : {:s} does not exist'.format(src_dir)
            continue

        file_list = [each for each in os.listdir(src_dir) if each.endswith('.txt')]
        n_frames = len(file_list)
        print 'Ground truth folder has {:d} files'.format(n_frames)

        dst_file = open(dst_fname, 'w')
        dst_file.write('frame   ulx     uly     urx     ury     lrx     lry     llx     lly     \n')
        line_id = 1
        for frame_id in xrange(n_frames):
            src_fname = '{:s}/frame{:05d}.txt'.format(src_dir, frame_id + 1)
            if not os.path.isfile(src_fname):
                continue
            data_file = open(src_fname, 'r')
            lines = data_file.readlines()
            if len(lines) != 2:
                raise StandardError('invalid formatting for frame {:d} in file {:s}'.format(frame_id, src_fname))

            data_file.close()
            gt_y = lines[0]
            gt_x = lines[1]
            words_y = gt_y.split()
            words_x = gt_x.split()
            if len(words_x) != 4 or len(words_y) != 4:
                raise StandardError(
                    'invalid formatting for frame {:d} in lines {:s}\n{:s}'.format(frame_id, words_y, words_x))
            uly = float(words_y[0])
            ury = float(words_y[1])
            lry = float(words_y[2])
            lly = float(words_y[3])
            ulx = float(words_x[0])
            urx = float(words_x[1])
            lrx = float(words_x[2])
            llx = float(words_x[3])
            corner_str = '{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}\t{:5.2f}'.format(
                ulx, uly, urx, ury, lrx, lry, llx, lly)
            dst_file.write('frame{:05d}.jpg\t{:s}\n'.format(frame_id + 1, corner_str))
        dst_file.close()



