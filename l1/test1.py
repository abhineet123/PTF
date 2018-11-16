import cv2
c = cv2.VideoCapture("SmoothMARSROVER.avi")
f = True
t = 1;
while t<450:
	f,img = c.read() # After some minutes all frames returnes are empty and f is false
	if not f:
		break                 # This doesn't throws any exception
	try:
		cv2.imwrite('{0:05d}.jpg'.format(t),img)
		t = t+1
		print "Frame no (%d)",(t)
	except cv2.error as e:
		print e # print error: (-206) Unrecognized or unsupported array type
	k=cv2.waitKey(5)
	if k==27:
			break
