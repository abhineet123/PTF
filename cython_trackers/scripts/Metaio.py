import math
import re

import cv2
import numpy as np

from nntracker.utility import *
# ---------- Transformation Between XGA boundary and Template boundary ---------- #
import pdb

class RegionScaling:
    def __init__(self, scale_factor):
        self._square = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]).T
        self.scale_matrix = np.matrix([[1,0,0],[0,1,0],[0,0,1/scale_factor]])

    def warp(self, w):
        return w * self.scale_matrix
    
    def region(self, r):
        w = self.warp(square_to_corners_warp(r))
        return apply_to_pts(w, self._square)

_scale_factor = 2.0
toTemplate = RegionScaling(1/_scale_factor)
toXGA = RegionScaling(_scale_factor)

# ---------- Parsing and Reading Metaio Files ---------- #

def parse_init_file(location):
    re_frame = re.compile("(?:Initialization data at frame )(\d+)")
    re_number = re.compile("([-+]?(?:\d+(?:\.\d*)?|\.\d+))")

    current_frame = None
    in_pts = []
    out_pts = []
    inits = {}

    for line in open(location):
        line = line.strip()
        m = re_frame.match(line)
        if m != None:
            if current_frame != None:
                inits[current_frame] = (np.array(in_pts).T, np.array(out_pts).T)
                in_pts = []
                out_pts = []
            current_frame = int(m.group(1))
        else:
            nums = map(float, re_number.findall(line))
            if nums != []:
                in_pts.append(nums[2:4])
                out_pts.append(nums[0:2])
    inits[current_frame] = (np.array(in_pts).T, np.array(out_pts).T)

    return inits

def open_benchmark(directory, seq_num):
    inits = parse_init_file(directory + "gtSeq%.2iinit.txt" % seq_num)
    #pdb.set_trace()
    video = cv2.VideoCapture(directory + "gtSeq%.2d.avi" % seq_num)

    return video, inits

# ---------- Running Trackers on Benchmarks ---------- #

def run_benchmark(video, inits, tracker):
    
    #cv2.namedWindow("Metaio Benchmark")
    corners = []
    frame = 0
    while True:
        succ, img = video.read()
        if not succ: break
        gray_img = cv2.GaussianBlur(np.asarray(to_grayscale(img)), (3,3), 0.75) 
        if frame in inits.keys():
            if tracker.is_initialized():
                tracker.set_region(toTemplate.region(inits[frame][1]))
            else:
                tracker.initialize(gray_img, toTemplate.region(inits[frame][1]))
        else:
            tracker.update(gray_img)
        if tracker.is_initialized():
            corners.append(toXGA.region(tracker.get_region()))
        frame += 1
        draw_region(img, tracker.get_region(), (255,0,0))
        draw_region(img, toXGA.region(tracker.get_region()), (0,255,0))
        #pdb.set_trace()
        #cv2.imshow("Metaio Benchmark", img)
        #cv2.waitKey(1)
    video.release()
    return corners

def load_and_run_benchmark(directory, seq_num, tracker):
    video, inits = open_benchmark(directory, seq_num)
    results = run_benchmark(video, inits, tracker)
    return results

def replay_results(vc, results_list):
    colour = [(255,0,0), (0,255,0), (0,0,255)]
    thickness = [3,2,1]
    
    cv2.namedWindow("replay")
    i = 0
    while True:
        succ, img = vc.read()
        if not succ: break

        annotated_img = img.copy()
        for j,results in enumerate(results_list):
            draw_region(annotated_img, results[i], colour[j], thickness[j])
        cv2.imshow("replay", annotated_img)
        cv2.waitKey(1)
        
        i += 1
