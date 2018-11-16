#from nntracker.trackers.TurnkeyTrackers import make_pf
'''
from BakerMatthewsICTracker import *
from CascadeTracker import *
from ESMTracker import *
from NNTracker import *
'''
#from nntracker.utility import *

import os
import cv
import cv2
import pdb
import numpy as np
from nntracker.utility import *
from nntracker.trackers.TurnkeyTrackers import *
import time
_step = [1]
_triger = ['fw']
_label = ['dlkt_homo_ncc']
class motion_experiments():
    def __init__(self, path_to_gts, path_to_imgs, path_to_write, trackers):
        self.img_path = path_to_imgs
        self.gt_path = path_to_gts
        self.write_path = path_to_write
        self.speeds = ['s4']
        self.seqs_name = ['bookI','bookII','cereal','mugI','mugII']
        self.trackers = trackers
        self.tracker = None
        self.fwobj = None
        self.triger = None
        self._step = None
        self.curr_working_path = None
        self.time = 0

    def run_all(self):
        for tracker in self.trackers:
            self.tracker = tracker
		    # check file existance
            curr_name = _label[self.trackers.index(tracker)]
            print curr_name
            self.curr_working_path = self.write_path+'/'+curr_name
            if not os.path.exists(self.curr_working_path):
                os.mkdir(self.curr_working_path)
            self.run_exp()

    def run_exp(self):
        for speed in self.speeds:
            for seq_name in self.seqs_name:
                # Set to write files
                print seq_name+'_'+speed
                self.time = 0
                if speed == '':
                    self.fwobj = open(self.curr_working_path+'/nl_'+seq_name+'.txt', 'w')
                    datas = open(self.gt_path+'/nl_'+seq_name+'_gt.txt').readlines()
                    img_path = self.img_path+'/nl_'+seq_name+'/'
                else:
                    self.fwobj = open(self.curr_working_path+'/nl_'+seq_name+'_'+speed+'.txt', 'w')
                    #datas = open(self.gt_path+'/nl_'+seq_name+'_'+speed+'_'+_label[0].split('_')[1]+'.txt').readlines()
                    datas = open(self.gt_path+'/nl_'+seq_name+'_'+speed+'.txt').readlines()[1:]
                    img_path = self.img_path+'/nl_'+seq_name+'_'+speed+'/'
                self.fwobj.write("frame\tulx\tuly\turx\tury\tlrx\tlry\tllx\tlly\n")
                img1 = cv2.imread(img_path+'frame%05d.jpg'%(1))
                img1 = to_grayscale(img1)
                img1 = cv2.GaussianBlur(np.asarray(img1), (5,5), 3)
                init_pts = datas[0].rstrip('\n')
                #init_pts = init_pts.split()[:]
                init_pts = init_pts.split()[1:]
                init_pts = np.array([[float(init_pts[0]),float(init_pts[1])],[float(init_pts[2]),float(init_pts[3])],[float(init_pts[4]),float(init_pts[5])],[float(init_pts[6]),float(init_pts[7])]]).T
                self.tracker.initialize(img1, init_pts)
		
                for indx in range(0,len(datas)):
                    img2 = cv2.imread(img_path+'frame%05d.jpg'%(indx+1))
                    img2_old = img2.copy()
                    img2 = to_grayscale(img2)
                    img2 = cv2.GaussianBlur(np.asarray(img2), (5,5), 3)
                    
                    time1 = time.time()
                    self.tracker.update(img2)
                    self.time += time.time() - time1
                    res_pts = self.tracker.get_region()
                    try:
                        draw_region(img2_old, res_pts, (255,0,0), 2)
                    except:
                        print 'draw results error'
                    #cv2.imshow('Trked', img2_old)
                    #cv.WaitKey(1)
                    self.fwobj.write('frame%05d.jpg'%(indx+1)+region_to_string(res_pts)+'\n')
                self.fwobj.close()
                   
    def to_track(self, init_pts, img1, img2):        
        self.tracker.initialize(img1, init_pts)
        self.tracker.update(img2)
        return self.tracker.get_region()

def region_to_string(region):
    output = ""
    for i in xrange(0,4):
        for j in xrange(0,2):
            output += "\t%.2f"%region[j,i]
    return output

def motion_measure(pts1, pts2):
    # Here we use corner differences
    res = 0.0
    for idx in range(len(pts1)):
        res += (pts1[idx][0]-pts2[idx][0])**2 + (pts1[idx][1]-pts2[idx][1])**2
    res = sqrt(res / 4)
    return res

if __name__ == '__main__':
    path_to_gts = '/home/xzhang6/Documents/'
    path_to_imgs = '/home/xzhang6/Documents/nina_dataset_wendy/nl'
    path_to_write = '/home/xzhang6/Documents/trk_res'
    trackers = []
    Model = 1
    trackers.append(make_dlkt(use_scv=False, MModel=Model, res=(100,100), iters=30))
    #trackers.append(make_dnn_bmic(use_scv=True, res=(50,50), bmic_iters=30, nn_samples=2000,  MModel=Model, gnn=False))
    #trackers.append(make_desm(use_scv=False, res=(50,50), iters=30, MModel=Model))
    #trackers.append(make_pf(use_scv=True, MModel=Model))
    exp = motion_experiments(path_to_gts, path_to_imgs, path_to_write, trackers)
    exp.run_all()
