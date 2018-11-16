#IC_S = [1, 1, 1, 1, 0.95, 0.9, 0.8, 0.72, 0.58, 0.48, 0.4, 0.37, 0.25, 0.22, 0.20, 0.18, 0.16, 0.15, 0.12, 0.1, 0.08]
#GNNIC_S = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.99, 0.97, 0.97, 0.96, 0.95, 0.942,  0.92, 0.89, 0.87 ,0.84]
#NNIC_S = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.99, 0.97, 0.96, 0.944, 0.942, 0.92, 0.88, 0.85, 0.82]
#GNN = [1, 1, 0.52, 0.41, 0.28, 0.22, 0.17, 0.09, 0.06, 0.05, 0.02, 0.01, 0.008, 0, 0, 0, 0, 0 ,0 , 0, 0]
#ANN = [1, 1, 0.53, 0.38, 0.26, 0.2, 0.07, 0.06, 0.055, 0.04, 0.02, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
#ESM_S = [1, 1, 1, 1, 1, 0.95, 0.92, 0.88, 0.78, 0.63, 0.58, 0.46, 0.42, 0.4, 0.35, 0.24, 0.22, 0.18, 0.15, 0.1, 0.07]
#RKLT_S = [1, 1 , 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.98, 0.96, 0.93, 0.91, 0.89]


#IC_A = [1, 1,1,1,1,0.82,0.61,0.4, 0.3]
#IC_S = [1, 1, 1, 1, 0.72, 0.54, 0.42, 0.25, 0.2 ]

#GNNIC_A = [1, 1,1,1,1,1,0.8, 0.76, 0.62]
#GNNIC_S = [1, 1, 1, 1, 1, 1, 1, 1, 0.92]

#NNIC_A = [1, 1,1,1,0.97,0.9,0.72,0.57, 0.5]
#NNIC_S = [1, 1, 1, 1, 1, 1, 1, 1, 0.95]

#ESM_A = [1, 1,1,1,0.98, 0.92, 0.78,0.57, 0.4]
#ESM_S = [1, 1, 1, 1, 1, 0.94, 0.9, 0.78, 0.68]

#RKLT_A = [1, 1, 1, 1, 1, 1, 0.80, 0.72, 0.67]
#RKLT_S = [1, 1, 1, 1, 1, 1 ,1 , 1, 0.98]

IC_C = [1,0.95,0.75,0.6,0.55,0.48,0.37,0.2]
NNIC_C = [1,0.95,0.75,0.6,0.55,0.49,0.52,0.4]
RKLT_C= [1,0.95,0.75,0.6,0.55,0.50,0.52,0.42]
GNNIC_C = [1,0.95,0.75,0.6,0.55,0.50,0.52,0.42]
ESM_C = [1,0.95,0.75,0.6,0.55,0.48,0.52,0.38]

#IC_J = [1,0.95,0.82,0.5,0.3,0.28,0.18,0.08]
#NNIC_J = [1,0.96,0.82,0.54,0.44,0.46,0.22,0.22]
#RKLT_J= [1,1,0.92,0.9,0.86,0.88,0.8,0.68]
#GNNIC_J = [1,0.98,0.84,0.67,0.52,0.50,0.42,0.30]
#ESM_J = [1,0.95,0.82,0.72,0.52,0.47,0.22,0.24]



X = [0, 2, 4, 6, 8, 10, 12, 14]
#X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#SSD = [0.45, 0.99, 0.45, 0.51, 0.52, 0.31]
#SCV = [1,1, 0.88, 0.66, 0.77,0.54]
#NCC = [1,1, 0.91, 0.64, 0.79,0.62]
#IVA = [0.89, 0.99, 0.62, 0.32, 0.50, 0.30]
#labels = ['bookI', 'bookII', 'mugI', 'mugII', 'cereal', 'juice']
#X = [1,2,3,4,5,6]
import matplotlib.pyplot as plt
import pylab
#from matplotlib import ax

#plt.plot(X, SSD, 'k',marker='D', linewidth=3.0,markersize=10, label='SSD')
#plt.plot(X, SCV, 'r',marker='*', linewidth=3.0,markersize=10, label='SCV')
#plt.plot(X, NCC, 'g',marker='>', linewidth=3.0,markersize=10, label='NCC')
#plt.plot(X, IVA, 'b',marker='8', linewidth=3.0,markersize=10, label='IVA')
#plt.legend(loc='lower left')
#plt.axis([1, 6, 0, 1])
#plt.xticks(X, labels)
#plt.grid()
#plt.xlabel('Sequences',fontsize=12)
#plt.ylabel('Success Rate',fontsize=12)


#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels)

IC_C = [1,1,1, 0.75, 0.5, 0.3, 0.25, 0.2]
IC_J = [1, 1, 0.9, 0.5, 0.31, 0.3, 0.20, 0.15]

NNIC_C = [1,1,1,1,1,0.94, 0.6, 0.51]
NNIC_J = [1, 1, 0.9, 0.52, 0.46, 0.47, 0.22, 0.22]

GNNIC_C = [1, 1,1,1,1,1, 0.73, 0.57] 
GNNIC_J = [1, 1, 0.92, 0.62, 0.6, 0.57, 0.51, 0.4]

ESM_C = [1,1,1, 0.82, 0.62, 0.55, 0.32, 0.27]
ESM_J =[1, 1, 0.92, 0.65, 0.6, 0.47, 0.22, 0.18]

RKLT_C = [1, 1,1,1,1,1, 0.76, 0.58]
RKLT_J =[1, 1, 0.95, 0.88, 0.84, 0.84, 0.8, 0.7]

plt.subplot(2, 1, 1)
plt.plot(X, IC_C, 'k',marker='o', linewidth=3.0,markersize=10, label='IC')
plt.plot(X, NNIC_C, 'r',marker='*',linewidth=3.0,markersize=10, label='NNIC')
plt.plot(X, GNNIC_C, 'g', marker='>',linewidth=3.0,markersize=10, label='GNNIC')
plt.plot(X, ESM_C, 'b',marker='<',linewidth=3.0,markersize=10, label='ESM')
plt.plot(X, RKLT_C, 'y', marker='4',linewidth=3.0,markersize=10, label='RKLT')
plt.axis([0, 14, 0, 1])
plt.legend(loc='lower left')
plt.xlabel('Sigma',fontsize=18)
plt.ylabel('Success Rate',fontsize=18)
plt.title('Cereal',fontsize=18)

plt.subplot(2, 1, 2)
plt.plot(X, IC_J, 'k',marker='o', linewidth=3.0,markersize=10, label='IC')
plt.plot(X, NNIC_J, 'r',marker='*',linewidth=3.0,markersize=10, label='NNIC')
plt.plot(X, GNNIC_J, 'g', marker='>',linewidth=3.0,markersize=10, label='GNNIC')
plt.plot(X, ESM_J, 'b',marker='<',linewidth=3.0,markersize=10, label='ESM')
plt.plot(X, RKLT_J, 'y', marker='4',linewidth=3.0,markersize=10, label='RKLT')
plt.axis([0, 14, 0, 1])

plt.legend(loc='lower left')
plt.xlabel('Sigma',fontsize=18)
plt.ylabel('Success Rate',fontsize=18)
plt.title('Juice',fontsize=18)


'''
points = [(1,2), (2,1), (3,3), (4,4), (5,6),(6,5),(7,7),(8,8)]
#points = [(232,0.84), (227,0.83), (234,0.81), (240,0.79), (252,0.72),(259,0.75),(305,0.54),(382,0.37)]
x_arr = []
y_arr = []
for x,y in points:
    x_arr.append(x)
    y_arr.append(y)

plt.plot(x_arr[0], y_arr[0], 'y',marker='^', linewidth=3.0,markersize=14, label='RKLT')
plt.plot(x_arr[1], y_arr[1], 'g',marker='>', linewidth=3.0,markersize=14, label='GNNIC')
plt.plot(x_arr[2], y_arr[2], 'b',marker='<', linewidth=3.0,markersize=14, label='ESM')
plt.plot(x_arr[3], y_arr[3], 'r',marker='*', linewidth=3.0,markersize=14, label='NNIC')
plt.plot(x_arr[4], y_arr[4], 'k',marker='o', linewidth=3.0,markersize=14, label='IC')
plt.plot(x_arr[5], y_arr[5], 'm',marker='p', linewidth=3.0,markersize=14, label='IVT')
plt.plot(x_arr[6], y_arr[6], 'c',marker='s', linewidth=3.0,markersize=14, label='L1')
plt.plot(x_arr[7], y_arr[7], '#feeefe',marker='d', linewidth=3.0,markersize=14, label='TLD')

#plt.axis([210, 400, 0.3, 0.9])

plt.axis([0, 9, 0, 9])
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()   
plt.legend(loc='upper left')
'''
'''
GNN_5k = [0.3, 0.65, 0.77, 0.82, 0.88, 0.94]
ANN_5k = [0.19, 0.65, 0.69, 0.73, 0.79, 0.81]
ANN_10k  = [0.19, 0.70, 0.79, 0.81, 0.91, 0.96]
ANN_40k = [0.19, 0.69, 0.81, 0.99,1, 1]

X = [1,3,5,7,9,11]

plt.plot(X, GNN_5k,  'b',marker='d', linewidth=3.0,linestyle='-',markersize=14, label='5000 SGANNS')
plt.plot(X, ANN_5k,  'r',marker='<', linewidth=3.0,linestyle='-',markersize=14, label='5000 ANN')
plt.plot(X, ANN_10k,  'g',marker='^', linewidth=3.0,linestyle='-',markersize=14, label='10000 ANN')
plt.plot(X, ANN_40k,  'm',marker='>', linewidth=3.0,linestyle='-',markersize=14, label='40000 ANN')

plt.legend(loc='lower right')
plt.xlabel('Success Threshold',fontsize=18)
#plt.xlabel('Number of Re-Iniializations',fontsize=18)
plt.ylabel('Overall Success',fontsize=18) 
#plt.ylabel('Overall Success',fontsize=18) 
plt.title('BookI Number of Sample Comparison')
plt.axis([0, 10, 0.2, 1])
#plt.title('Un-Normalized Rank Plot')
'''
#plt.grid()
plt.show()