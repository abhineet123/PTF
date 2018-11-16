import numpy 
import cv2
import matplotlib.pyplot as plt
from numpy import matrix as MA
import matplotlib.cm as cm
import pylab
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import Image
import matplotlib.widgets as widgets

def onselect(eclick, erelease):
    if eclick.ydata>erelease.ydata:
        eclick.ydata,erelease.ydata=erelease.ydata,eclick.ydata
    if eclick.xdata>erelease.xdata:
        eclick.xdata,erelease.xdata=erelease.xdata,eclick.xdata
    ax.set_ylim(erelease.ydata,eclick.ydata)
    ax.set_xlim(eclick.xdata,erelease.xdata)
    fig.canvas.draw()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#filename="/home/ankush/Edmonton/IMG_1218.JPG"
#im = Image.open(filename)
#arr = np.asarray(im)
#plt_image=plt.imshow(arr)
#rs=widgets.RectangleSelector(
    #ax, onselect, drawtype='box',
    #rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
#plt.show()


def binary(img1, level):
	img1 = MA(img1)
	#import pdb;pdb.set_trace()
	tempIm = img1.flatten(1)
	for i in range(tempIm.shape[1]):
		if tempIm[0,i] < level:
			tempIm[0,i] = 0;
		else:
			tempIm[0,i] = 1;
	tempIm = (numpy.reshape(tempIm,(img1.shape[1],img.shape[0]))).T
	print tempIm
	return tempIm

#def test():
ll = "/home/ankush/Desktop/images.jpeg"
img = cv2.imread(ll)
b,g,r = cv2.split(img)
img1 = 0.2989*r + 0.5870 * g + 0.1140 * b;
# img1 = np.array(img1,dtype='float32')
# img1 = numpy.rint(img1)
img1 = img1/255;
	#IM1 = np.transpose(np.nonzero(img1>0.3))
#binaryIntermediate = MA(numpy.zeros((img1.shape[0],img1.shape[1])))
import pdb;pdb.set_trace()
# im_bw = cv2.threshold(img1, 128, 255, cv2.THRESH_BINARY)
tempIm = binary(img1,0.3)
# cv2.threshold(img1, 77, 255, 'THRESH_BINARY',binaryIntermediate);
# im2 = stats.threshold(img1, threshmin=0.3, threshmax=1, newval=0)
#pylab.imshow(im2)
# pylab.show()
#plt.imshow(im_bw,cmap = cm.binary_r)#,cmap = cm.Greys_r)
#plt.show()

#img2 = cv2.merge([r,g,b])
# plt.subplot(121);plt.imshow(img) # expects distorted color
# plt.imshow(img) # expect true color
#AA = np.array(img[:,:,0]);
# plt.show()
#img1 = AA[0:200,0:200]
#plt.imshow(img1) # expect true color
#plt.show()
#print AA[0,0], 
#cv2.imshow('bgr image',img) # expects true color
#cv2.imshow('rgb image',img2)


#img = cv2.imread(ll)
#cv2.imshow("window", img)
# tools.show_1_image_pylab(img)
#plt.subpploprint img.shape


	
#if __name__ == '__init__':
#	test()
