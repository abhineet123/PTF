from L1TrackingBPR_APGup import L1TrackingBPR_APGup
from numpy import matrix as MA
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import cv2
import matplotlib.cm as cm
import matplotlib
import numpy as np
import os


def tellme(s):
    print(s)
    plt.title(s, fontsize=10)
    plt.draw()


t = 0;
#Movie = cv2.VideoCapture("Videos/RobotNewSetup.avi")
#f,img = Movie.read()
img = cv2.imread('G:/UofA/Thesis/#Code/Datasets/Human/nl_bookI_s3/img1.jpg')
plt.imshow(img)
pts = [];
while len(pts) < 4:
    tellme('Select 4 corners with mouse anticlockwise starting with top left')
    pts = np.asarray(plt.ginput(4, timeout=-1))
    if len(pts) < 4:
        tellme('Too few points, starting over')
        time.sleep(1) # Wait a second
plt.close()


class para():
    def __init__(self):
        self.lambd = MA([[0.2, 0.001, 10]])
        self.angle_threshold = 50
        self.Lip = 8;
        self.Maxit = 5;
        self.nT = 10; # number of templates for the sparse representation
        self.rel_std_afnv = MA([[0.005, 0.003, 0.005, 0.003, 1, 1]]); # diviation of the sampling of particle filter
        self.n_sample = 100; # No of particles
        self.sz_T = MA([[12, 15]]); # Reshape each image so that they have the same image space representation
        self.init_pos = MA(
            [[int(pts[0, 1]), int(pts[1, 1]), int(pts[3, 1])], [int(pts[0, 0]), int(pts[1, 0]), int(pts[3, 0])]])
        self.path = 'G:/UofA/Thesis/#Code/Datasets/Human/nl_bookI_s3/img'
        self.results = 'ResultTracking'
        if not os.path.exists(self.results):
            os.makedirs(self.results)
        self.noZeros = '5'

    #		self.bDebug = 0; # debugging indicator
#		self.bShowSaveImag = 1 ; # indicator for result image show and save after tracking finished
def main():
    paraT = para()
    L1TrackingBPR_APGup(paraT)


if __name__ == '__main__':
    main()
