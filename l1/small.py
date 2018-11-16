import numpy
from numpy import array as NA
import cv2
import matplotlib.pyplot as plt

A = NA([[3,10,4],[1,2,4]])
B = NA([[1,2],[7,8],[1,3]]);
M = NA([[1,2,3,4,5,6]])
#ll = "/home/ankush/Edmonton/IMG_1214.JPG"
#img = cv2.imread(ll)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
#thresh  = 0.03*255;

#im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
#plt.imshow(im_bw) # expect true color
##plt.show()
#b = A[:,0]
#print b[0]
#A1 = numpy.nonzero((A[:,0]<24) & (A[:,1]>=2));
#A1= numpy.transpose(A1)
#print A1,A[:,0]<2
#A1 = [[0,1,2]]
#A[A1] = [[2,-2,-3]]
#B = A.flatten(1);
 
print numpy.dot(A,B)
