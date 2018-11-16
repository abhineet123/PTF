
import cv2

from Homography import *
from InteractiveTracking import *

class PGMSequenceTracking(InteractiveTrackingApp):
    
    def __init__(self, file_names, tracker, name="vis"):
        InteractiveTrackingApp.__init__(self, tracker, name)
        self.file_names = file_names

    def run(self):
        img = cv2.imread(self.file_names[0])
        self.on_frame(img)
        self.paused = True
        i = 0
        while i < len(self.file_names):
            if not self.paused: 
                img = cv2.imread(self.file_names[i])
                i += 1
            self.on_frame(img)
            cv2.waitKey(1)
        self.cleanup()

