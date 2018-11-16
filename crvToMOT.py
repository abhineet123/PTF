root_dir = 'C:/Datasets/IDOT/orig/groundTruth'
seq_names = [
    '193402_Main_St_(US_51_Bus)_and_Empire_St_(IL_9)_in_Bloomington_20141023_11am',
    '243948_IL_126_@_Ridge_Rd._001_20150625_10am',
    '245837_FAI-74_E_of_St._Joseph_in_Champaign_County_20150630_09am',
    '251035_Princeton_34_&_26_T_20150812_08am',
    '251950_IL_8_(E.Washington_St)_&_Illini_Dr_-_Farmdale_Rd_20150818_12pm',
    '252707_FAI-74_E_of_Lincoln_Ave_in_Urbana_20150826_09am',
    '20150829_020000DST_ciceroPeterson', '20150829_020000DST_elstonIrvingPark',
    '20150918_150500DST_halsted', 'ILCHI_CHI003_20151010_075033_051',
    'ILCHI_CHI120_20151013_095039_099',
    'ILCHI_CHI164_20150930_125029_234',
    'intersection_4']
seq_id = 12
seq_name = seq_names[seq_id]
in_fname = '{:s}/{:s}.txt'.format(root_dir, seq_name)
in_data = open(in_fname, 'r').readlines()
out_fname = '{:s}/seq_{:d}.txt'.format(root_dir, seq_id + 1)
out_fid = open(out_fname, 'w')
line_id = 0
for line in in_data:
    # object_id x y width height frame_id if_lost if_occluded if_interpolated label
    gt_line = line.strip().split()
    if len(gt_line) < 6:
        raise StandardError("Invalid formatting on line {:d} of GT file: {:s}".format(line_id, line))
    obj_id = int(gt_line[0])
    x = int(gt_line[1])
    y = int(gt_line[2])
    width = int(gt_line[3])
    height = int(gt_line[4])
    frame_id = int(gt_line[5])
    out_fid.write('{:d},{:d},{:d},{:d},{:d},{:d},1,-1,-1,-1\n'.format(
        frame_id + 1, obj_id + 1, x, y, width, height))
    line_id += 1
    if line_id % 100 == 0:
        print 'Done {:d} lines'.format(line_id)
out_fid.close()
