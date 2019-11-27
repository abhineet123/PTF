from multiprocessing import Process
import visualizeWithMotion as vwm

if __name__ == '__main__':

    args1 = 'on_top=0 top_border=0 keep_borders=1 n_images=2 random_mode=1 auto_progress=1 fps=15 monitor_id=0 ' \
            'duplicate_window=0 second_from_top=1 win_offset_y=0 width=1920 height=1050 ' \
            'src_dirs=vids/20/2/14,vids/20/2/13 ' \
            'only_maximized=0 ' \
            'reversed_pos=2 video_mode=2 multi_mode=1 auto_progress_video=1 preserve_order=1 lazy_video_load=0'
    args2 = 'on_top=0 top_border=0 keep_borders=1 n_images=2 ' \
            'random_mode=1 auto_progress=1 transition_interval=30 monitor_id=0  ' \
            'duplicate_window=0 second_from_top=1 win_offset_y=0 width=1920 height=1050 ' \
            'src_dirs=vids/20/2/13**10,vids/20/2_patches**10,!20/9,!20/10,20,20/2*5,20/1*3,20/1/1_0*5,20/1/1_1*25,' \
            '20/1/1_8*8,20/1/1_9*20,20/1/1_3*4,20/1/1_5*15,20/1/1_2*15,20/1/1_4*20 ' \
            'only_maximized=0 reversed_pos=2'

    # args1 = args1.split(' ')
    # args2 = args2.split(' ')

    vwm1_thread = Process(target=vwm.main, args=(args1, ))
    vwm1_thread.start()

    vwm2_thread = Process(target=vwm.main, args=(args2, ))
    vwm2_thread.start()
