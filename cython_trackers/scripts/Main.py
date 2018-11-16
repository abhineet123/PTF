"""
A small standalone application for tracker demonstration. Depends
on OpenCV VideoCapture to grab frames from the camera.

Author: Travis Dick (travis.barry.dick@gmail.com)

"""

import os
import sys

from nntracker.utility import *
from nntracker.InteractiveTracking import *
from nntracker.trackers.TurnkeyTrackers import *

from nntracker.trackers.BMICTracker import BMICTracker
from nntracker.trackers.CascadeTracker import CascadeTracker
from nntracker.trackers.NNTracker import NNTracker


class StandaloneTrackingApp(InteractiveTrackingApp):
    """ A demo program that uses OpenCV to grab frames. """
    
    def __init__(self, vc, tracker, name="vis", start_paused=False):
        InteractiveTrackingApp.__init__(self, tracker, name, init_with_rectangle=False)
        self.vc = vc
        self.start_paused = start_paused
    
    def run(self):
        if self.start_paused:
            succ, img = self.vc.read()
            self.on_frame(img)
            self.paused = True
        while True:
            if not self.paused:
                (succ, img) = self.vc.read()
                if not succ: break
            if not self.on_frame(img): break
            cv2.waitKey(1)
        self.cleanup()


if __name__ == '__main__':
    if len(sys.argv) > 1: 
        vc = cv2.VideoCapture(os.path.expanduser(sys.argv[1]))
        start_paused = True
    else: 
        vc = cv2.VideoCapture(0)
        start_paused = False
    
    app = StandaloneTrackingApp(vc, [make_nn_bmic(use_scv=True), make_esm(use_scv=True)], start_paused = start_paused)
    app.run()
    app.cleanup()
