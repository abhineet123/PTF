import math
import sys
import matplotlib.pyplot as plt
'''
USAGE: python <path of tracked file> <path of ground truth> <path of file containing inter frame motion values>
     : Notice TH (threshold) is set at 5, you can change it if you want
Format of third file:
frame Interframe_Motion	
frame00001.jpg 1.0
frame00002.jpg 2.0
frame00003.jpg 1.5
frame00004.jpg 1.6
frame00005.jpg …

OUTPUT: Speed Sensitivity (graph) in working directory
'''

def SpeedSensitivityPlot():
    trackerPath = sys.argv[1]
    GTPath =  sys.argv[2]
    Interframe = sys.argv[3]
    TH = 5
    load_Tracker = open(trackerPath, 'r').readlines()
    load_GT = open(GTPath, 'r').readlines()
    InterframeM = open(Interframe, 'r').readlines()
    no_frames = len(load_Tracker) - 1
    dict = {}
    I = 1
    while I < len(load_GT):
        Tracker = load_Tracker[I].strip().split()
        GT = load_GT[I].strip().split()
        Motion = float(InterframeM[I].strip().split()[-1])
        Err = 0
        for p in range(1,9):
            Err = (float(Tracker[p]) - float(GT[p])) ** 2 + Err
        Err = math.sqrt(Err/4)
	if Motion not in dict:
	    if Err > TH:
		dict[Motion] = [0,1]
	    else:
		dict[Motion] = [1,1]
	else:
	    if Err > TH:
		temp = dict[Motion]
	    	temp[1] = temp[1] + 1
		dict[Motion] = temp
	    else:
		temp = dict[Motion]
		temp[0] = temp[0] + 1
		temp[1] = temp[1] + 1
		dict[Motion] = temp
	I += 1
    x = dict.keys()
    x.sort()
    y = []	
    width = 0.05
    count = []
    for p in x:
        temp = dict[p]
        if temp[1] == 0:
            y.append(0.0)
            count.append(0)
        else:
            y.append(float(temp[0])/(temp[1]))
            count.append(temp[1]) 
    # Plot figure 
    plt.suptitle('Speed Sensitivity' , fontsize=15)
    plt.plot(x, y, linewidth=4.0) # Line property can be changed  
    plt.savefig('example.jpg')
    plt.close() 

if __name__ == '__main__':
    SpeedSensitivityPlot()
