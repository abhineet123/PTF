# Detector test1.py
import cv2
import numpy as np
import itertools



img = cv2.imread("messi.jpeg")
template = cv2.imread("template.jpeg")
detector = cv2.FeatureDetector_create("SIFT")
descriptor = cv2.DescriptorExtractor_create("SIFT")
distance = 0.5
skp = detector.detect(img)
skp, sd = descriptor.compute(img, skp)

tkp = detector.detect(template)
tkp, td = descriptor.compute(template, tkp)
flann_params = dict(algorithm=1, trees=4)
flann = cv2.flann_Index(sd, flann_params)
idx, dist = flann.knnSearch(td, 1, params={})
del flann
dist = dist[:,0]/2500.0
dist = dist.reshape(-1,).tolist()
idx = idx.reshape(-1).tolist()
indices = range(len(dist))
indices.sort(key=lambda i: dist[i])
dist = [dist[i] for i in indices]
idx = [idx[i] for i in indices]
skp_final = []
for i, dis in itertools.izip(idx, dist):
    if dis < distance:
        skp_final.append(skp[i])
    else:
        break

h1, w1 = img.shape[:2]
h2, w2 = template.shape[:2]
nWidth = w1+w2
nHeight = max(h1, h2)
hdif = (h1-h2)/2
newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
newimg[hdif:hdif+h2, :w2] = template
newimg[:h1, w2:w1+w2] = img
#tkp = tkp_final
#skp = skp_fianl
import pdb;pdb.set_trace()
for i in range(min(len(tkp), len(skp))):
    pt_a = (int(tkp[i].pt[0]), int(tkp[i].pt[1]+hdif))
    pt_b = (int(skp[i].pt[0]+w2), int(skp[i].pt[1]))
    cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
