import cv2
video  = cv2.VideoWriter('VS.avi', -1, 25, (640, 480));

for t in range(1900):
   img = cv2.imread('VideoResults/VS/{0:05d}.jpg'.format(t))
   video.write(img)
   cv2.imshow("webcam",img)
   if (cv2.waitKey(5) != -1):
       break
video.release()