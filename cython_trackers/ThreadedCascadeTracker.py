"""
Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import threading

from TrackerBase import *

class ThreadedCascadeTracker(TrackerBase):
    
    def __init__(self, trackers, set_warp_directly=True):
        self.trackers = trackers
        self.set_warp_directly = True
        self.init_threads_mutex = threading.Lock()
        self.init_threads = []
        self.initialized = False

    def _set_state(self, tracker, state):
        if self.set_warp_directly: tracker.set_warp(state)
        else: tracker.set_region(state)

    def _get_state(self, tracker):
        if self.set_warp_directly: return tracker.get_warp()
        else: return tracker.get_region()

    def _finest_initialized_index(self):
        for i in xrange(len(self.trackers)-1, -1, -1):
            if self.trackers[i].is_initialized(): return i
        return None

    def initialize(self, img, region):
        for t in self.trackers:
            thread = _TrackerInitThread(t, img, region)
            thread.start()
            with self.init_threads_mutex:
                self.init_threads.append(thread)

    def update(self, img):
        i = self._finest_initialized_index()
        if i != None:
            state = self._get_state(self.trackers[i])
            for t in self.trackers:
                if t.is_initialized():
                    self._set_state(t, state)
                    t.update(img)
                    state = self._get_state(t)
    
        with self.init_threads_mutex:
            for thread in self.init_threads:
                if not thread.is_alive(): 
                    thread.join()
                    self.init_threads.remove(thread)

    def is_initialized(self):
        return self._finest_initialized_index() != None

    def set_warp(self, warp):
        for t in self.trackers:
            if t.is_initialized():
                t.set_warp(warp)
    
    def get_warp(self):
        i = self._finest_initialized_index()
        if i != None: return self.trackers[i].get_warp()

    def set_region(self, region):
        for t in self.trackers:
            if t.is_initialized():
                t.set_region(region)
    
    def get_region(self):
        i = self._finest_initialized_index()
        if i != None: return self.trackers[i].get_region()
    
class _TrackerInitThread(threading.Thread):
    
    def __init__(self, tracker, img, region):
        threading.Thread.__init__(self)
        self.tracker = tracker
        self.img = img
        self.region = region

    def run(self):
        self.tracker.initialize(self.img, self.region)
