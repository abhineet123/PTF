import numpy as np                       
import math
import sys
'''
USAGE: python <path of tracked file> <path of ground truth>
     : Notice TH (threshold) is set at 5, you can change it if you want

OUTPUT: Average Drift
'''
def AverageDrift():
	
	trackerPath = sys.argv[1]
	GTPath =  sys.argv[2]
	TH = 5	
	load_Tracker = open(trackerPath, 'r').readlines()
	load_GT = open(GTPath, 'r').readlines()
	no_frames = len(load_Tracker) - 1
	E = 0
	I = 1
	
	while I < no_frames:
		Tracker = load_Tracker[I].strip().split()
		GT = load_GT[I].strip().split()
		err = 0
		# Alignment error 
		for p in range(1,9):
				err = (float(Tracker[p]) - float(GT[p]))**2 + err
		err = math.sqrt(err/4)
		if err < TH:E += err
		I += 1

	print E/float(no_frames)
	
if __name__ == '__main__':
	AverageDrift()
		
