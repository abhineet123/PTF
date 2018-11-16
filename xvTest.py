__author__ = 'abhineet'
from xvInput import *
import numpy as np
import cv2

if __name__ == '__main__':
    track_window_name = 'Frames'

    img_source = 1

    camname = '/dev/video0'
    usbfmt = 'BA24'

    fname = '/home/abhineet/G/UofA/Thesis/Xvision/XVision2/src/sdf3.mpg'
    fname2 = '/home/abhineet/G/UofA/Thesis/#Code/Datasets/Human/nl_cereal_s3.mpg'

    [width, height] = initSource(img_source, fname, usbfmt)

    print 'width: ', width, 'height: ', height
    # img = np.zeros((height, width, 3)).astype(np.uint8)
    cv2.namedWindow(track_window_name)

    frame_id=1
    while True:
        # frame_id=getFrame(img)
        img = getFrame2(None)
        # print 'received frame ', frame_id
        frame_id+=1
        #print 'status: ', status
        cv2.imshow(track_window_name, img)
        if cv2.waitKey(1) == 27:
            break



