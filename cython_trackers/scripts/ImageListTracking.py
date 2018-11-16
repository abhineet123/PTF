from nntracker.InteractiveTracking import *

class ImageListTracking(InteractiveTrackingApp):

    def __init__(self, image_list, tracker, name="vis"):
        InteractiveTrackingApp.__init__(self, tracker, name)
        self.images = image_list

    def run(self):
        img = self.images[0]
        self.on_frame(img)
        self.paused = True

        i = 0
        while i < len(self.images):
            if not self.paused:
                img = self.images[i]
                i += 1
            self.on_frame(img)
            cv2.waitKey(7)
        self.cleanup()
            
