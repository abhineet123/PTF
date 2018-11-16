from nntracker.utility import *
from nntracker.trackers.FindGoodPatches import *

import cv
import cv2


class TrackerSchedule:
    def __init__(self, region, start_time, end_time):
        self.region = region
        self.start_time = start_time
        self.end_time = end_time

class ScheduledTracking:
    
    def __init__(self, images, tracker_maker, schedule = None, output_file=None):
        self.images = images
        self.tracker_maker = tracker_maker
        self.schedule = schedule
        if self.schedule == None: self.schedule = []
        self.output_file = output_file
        self.paused = False

    def click_handler(self, evt, x, y, arg, extra):
        if evt == cv2.EVENT_RBUTTONDOWN:
            self.paused = not self.paused
        if evt == cv2.EVENT_LBUTTONDOWN:
            print x, y, self.time

    def run(self, start=0):
        cv2.namedWindow("vis")
        cv2.setMouseCallback("vis", self.click_handler)
        if self.output_file != None:
            vw = cv2.VideoWriter(self.output_file, cv.CV_FOURCC('M','J','P','G'), 15, (800,600), 1)
        active_trackers = []
        self.time = start
        while self.time < len(self.images):
            if not self.paused:
                img = self.images[self.time]
                gray_img = to_grayscale(img)
                
                to_remove = []
                for (t, end_time) in active_trackers:
                    if end_time == self.time: to_remove.append((t, end_time))
                    else: t.update(gray_img)
                for e in to_remove: active_trackers.remove(e)

                for s in self.schedule:
                    if s.start_time == self.time:
                        tracker = self.tracker_maker()
                        tracker.initialize(gray_img, s.region)
                        active_trackers.append((tracker, s.end_time))

                self.time += 1
            
            annotated_img = img.copy()
            for (t, end_time) in active_trackers:
                draw_region(annotated_img, t.get_region(), (0,255,0), 2)
            
            cv2.imshow("vis", annotated_img)
            if self.output_file != None: vw.write(annotated_img)
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        
        
