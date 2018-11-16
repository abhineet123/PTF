"""
Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import cv
import cv2
import numpy as np

from utility import *
import pdb

class InteractiveTrackingApp:
    def __init__(self, trackers, name="vis", init_with_rectangle=True, colours=[(255,0,0), (0,255,0), (0,0,255), (255,255,255), (0,0,0)],
                 thickness=[2,1,1,1,1]):
        """ An interactive window for initializing and visualizing tracker state.

        The on_frame method should be called for each new frame. Typically real
        applications subclass InteractiveTrackingApp and build in some application
        loop that captures frames and calls on_frame.
        
        Parameters:
        -----------
        tracker : TrackerBase
          Any class implementing the interface of TrackerBase. 

        name : string
          The name of the window. Due to some silliness in OpenCV this must
          be unique (in the set of all OpenCV window names).

        See Also:
        ---------
        StandaloneTrackingApp
        RosInteractiveTrackingApp
        """

        self.tracking = False
        self.trackers = trackers
        self.name = name
        self.init_with_rectangle = init_with_rectangle
        self.colours = colours
        self.thickness = thickness

        self.m_start = None
        self.m_end = None
        self.corner_pts = []

        self.gray_img = None
        self.paused = False
        self.img = None
        self.annotated_img = None
        cv2.namedWindow(self.name,0)
        cv2.setMouseCallback(self.name, self.mouse_handler)

    def display(self, img):
        self.annotated_img = img.copy()

        if self.tracking:
            for i,tracker in enumerate(self.trackers):
                if tracker.is_initialized():
                    draw_region(self.annotated_img, tracker.get_region(), self.colours[i], self.thickness[i])        	   
        if self.init_with_rectangle and self.m_start != None and self.m_end != None:
            ul = (min(self.m_start[0],self.m_end[0]), min(self.m_start[1],self.m_end[1]))
            lr = (max(self.m_start[0],self.m_end[0]), max(self.m_start[1],self.m_end[1]))             
            corners = np.array([ ul, [lr[0],ul[1]], lr, [ul[0],lr[1]]]).T            
            draw_region(self.annotated_img, corners, (255,0,0), 1)

        if not self.init_with_rectangle:
            for pt in self.corner_pts:
                cv2.circle(self.annotated_img, pt, 2, (255,0,0), 1)

        cv2.imshow(self.name, self.annotated_img)

    def mouse_handler(self, evt,x,y,arg,extra):
        if self.gray_img == None: return 
        if evt == cv2.EVENT_LBUTTONDOWN and self.m_start == None:
            if self.init_with_rectangle:
                self.m_start = (float(x),float(y))
                self.m_end = (x,y)
                self.paused = True
            else:
                self.corner_pts.append( (x,y) )
                if len(self.corner_pts) == 4:
		    print self.corner_pts
                    self.display(self.img)
                    cv2.waitKey(1)
                    corners = np.array(self.corner_pts, dtype=np.float64).T
                    if not self.tracking:
                        for tracker in self.trackers:
                            tracker.initialize(self.gray_img, corners)
                        self.tracking = True
                    else:
                        for tracker in self.trackers:
                            tracker.set_region(corners)
                    self.corner_pts = []
                    self.paused = False


        elif evt == cv2.EVENT_MOUSEMOVE and self.m_start != None:
            if self.init_with_rectangle:
                self.m_end = (x,y)
        elif evt == cv2.EVENT_LBUTTONUP:
            if self.init_with_rectangle:
                self.m_end = (float(x),float(y))
                ul = (min(self.m_start[0],self.m_end[0]), min(self.m_start[1],self.m_end[1]))
                lr = (max(self.m_start[0],self.m_end[0]), max(self.m_start[1],self.m_end[1]))
		print ul
		print lr
                if not self.tracking:
                    for tracker in self.trackers:
                        tracker.initialize_with_rectangle(self.gray_img, ul, lr)
                else: 
                    for tracker in self.trackers:
                        tracker.set_region(np.array([ ul, [lr[0],ul[1]], lr, [ul[0],lr[1]]]).T)
                self.m_start, self.m_end = None, None
                self.tracking = True
                self.paused = False

    def on_frame(self, img):
        if not self.paused:
            self.img = img
            self.gray_img = cv2.GaussianBlur(np.asarray(to_grayscale(img)), (5,5), 3)
            for tracker in self.trackers:
                tracker.update(self.gray_img)
#            pdb.set_trace()
        key = cv.WaitKey(33)
        # Since GTK encodes more information in the higher bits of the key
        # value, we need to get rid of those before we do the key comparison
        if key != -1: key %= 256 
        if self.img != None: self.display(self.img)
        #pdb.set_trace()
        if key == ord(' '): self.paused = not self.paused
        elif key == ord('c'): self.tracking = False
        elif key == ord('m'):
             corners = raw_input('please input values in this format: ulx uly,urx ury,lrx lry,llx lly ')
             corners = corners.split()
             new_corners = []
             for i in range(4):
                 xy_temp = [int(float(corners[2*i])), int(float(corners[2*i+1]))]
                 new_corners.append(xy_temp)
             corners = new_corners
             #for i in range(len(corners)):
             #    xy_temp  = corners[i].split()
             #    xy_temp = [int(float(j)) for j in xy_temp]
             #    corners[i] = xy_temp
             corners = np.array(corners,dtype=np.float64).T
             #for tracker in self.trackers:
             #    tracker.set_region(corners)
             #self.paused = False

             # tracker updating
             self.img = img
             self.gray_img = cv2.GaussianBlur(np.asarray(to_grayscale(img)), (5,5), 3)
             print self.tracking
             if not self.tracking:
                 for tracker in self.trackers:
                     tracker.initialize(self.gray_img,corners)
                 self.tracking = True
             else:
                 for tracker in self.trackers:
                     tracker.set_region(corners)
             self.paused = False
             self.corner_pts = []
             self.display(self.img)
        elif key > 0: return False
        return True

    def cleanup(self):
        cv2.destroyWindow(self.name)
