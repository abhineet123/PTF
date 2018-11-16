import cv2
import numpy as np
import timeit 

from nntracker.utility import *

def make_one_run(tracker, img, region, sigma = 8):
    tracker.initialize(img, region)
    def one_run():
        disturbance = np.random.normal(0, sigma, (2,4))
        disturbed_region = region + disturbance
        tracker.set_region(disturbed_region)
        tracker.update(img)
    return one_run

def time_tracker(tracker, img, region, sigma = 8, num_runs=1000):
    time = timeit.timeit(make_one_run(tracker, img, region, sigma), number=num_runs)
    return time / num_runs * 1000 # returns miliseconds taken
        
def time_search(desired_time, f, img, region, min_v, max_v, sigma=8):
    def evaluate_at(v):
        tracker = f(v)
        return time_tracker(tracker, img, region, sigma, 300)
    while (max_v - min_v) > 1:
        mid_v = (max_v + min_v)/2
        time = evaluate_at(mid_v)
        print "range = [%d, %d], mid = %d, time = %g" % (min_v, max_v, mid_v, time)
        if time > desired_time: max_v = mid_v
        if time < desired_time: min_v = mid_v

    best_v = (max_v + min_v) / 2
    best_time = time_tracker(f(best_v), img, region, sigma, 1000)

    print "--> best_v = %d, best_time = %g" % (best_v, best_time)
    
    return (best_v, best_time)

img = cv2.resize(np.asarray(to_grayscale(cv2.imread("/Users/travisdick/Desktop/Lenna.png"))), (256, 256))
ul = (256/2-50, 256/2-50)
lr = (256/2+50, 256/2+50)
region = rectangle_to_region(ul, lr)
    
