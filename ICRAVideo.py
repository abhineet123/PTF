# Image Edit functionalaties 
import cv2
import numpy as np

# Stacking four images
img1 = cv2.imread('frame00001.jpg')
img2 = cv2.imread('frame00002.jpg')
img3 = cv2.imread('frame00003.jpg')
img4 = cv2.imread('frame00004.jpg')

vis = np.concatenate((img1,img2), axis=1)
vis1 = np.concatenate((img3,img4), axis=1)

final_image = np.concatenate((vis, vis1), axis=0)

# Putting a filled rectangle on a corner
#format cv2.rectangle(img,(startx,starty), (endx,endy),(R,G,B), -1)
BB = cv2.rectangle(img1, (680, 10), (790, 60), (255, 238, 51), -1)

#Putting Text
CC = cv2.putText(A,"IALK", (705,44), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)

# Maintain the relative ordering of BB and then CC